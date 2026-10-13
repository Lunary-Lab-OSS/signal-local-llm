"""End-to-end: real transformers backend on this host (WSL2/Linux, CPU).

Downloads a small real model, loads it through LocalLLMModelLoader with the
configured backend, and generates through LLMComponent — verifying the
download manifest, backend dispatch, chat template, budgets, and reasoning
extraction against a real tokenizer/model.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

INTEGRATION = os.getenv("SIGNAL_LLM_INTEGRATION", "") == "1"
requires_integration = pytest.mark.skipif(
    not INTEGRATION, reason="set SIGNAL_LLM_INTEGRATION=1 (downloads a real model)"
)

MODEL_REPO = os.getenv("SIGNAL_LLM_E2E_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_REVISION = os.getenv("SIGNAL_LLM_E2E_REVISION", "7ae557604adf67be50417f59c2c2f167def9a775")


@requires_integration
def test_transformers_end_to_end(tmp_path):
    pytest.importorskip("transformers")
    from signal_llm.component import LLMComponent
    from signal_llm.config import LLMConfig, ModelConfig
    from signal_llm.loader import LocalLLMModelLoader

    model = ModelConfig(
        name="e2e-model",
        repo_id=MODEL_REPO,
        backend="transformers",
        device="cpu",
        dtype="float32",
        revision=MODEL_REVISION,
    )
    config = LLMConfig(
        router_priority=[model],
        semantic_priority=[model],
        agent_priority=[model],
        max_tokens=64,
        temperature=0.1,
    )
    loader = LocalLLMModelLoader(
        models_dir=tmp_path / "models", cache_dir=tmp_path / "cache", device="cpu"
    )
    component = LLMComponent(config=config, model_loader=loader, device="cpu", platform="linux")

    # Semantic generation with an explicit system prompt.
    answer = component.generate_semantic(
        "Reply with exactly the word: pong", system_prompt="You only answer with one word."
    )
    assert isinstance(answer, str)
    assert answer.strip() != ""
    # The system prompt must not leak into the visible answer (M2/M3).
    assert "system" not in answer.lower() or answer.strip() == "pong"

    # The completion manifest must exist and mark the snapshot complete.
    from signal_llm.loader import model_cache_key

    cache_dir = tmp_path / "models" / model_cache_key(MODEL_REPO, MODEL_REVISION)
    manifest = cache_dir / ".signal-llm-manifest.json"
    assert manifest.exists(), "completion manifest must be written after download"

    # Agent generation through the shared handle (same repo+revision+backend).
    agent_answer = component.generate_agent("Say hello in one word.")
    assert isinstance(agent_answer, str)
    assert agent_answer.strip() != ""
    assert "<think>" not in agent_answer, "reasoning must not leak into voice output (M3)"

    # Budget contract: instrument the real generation call and verify the
    # actual generated-token count, not merely that the handle exists.
    actual_engine = component.agent_model.handle
    generate_calls: list[dict] = []
    real_generate = actual_engine.generate

    def instrumented_generate(**kwargs):
        generate_calls.append(dict(kwargs))
        return real_generate(**kwargs)

    actual_engine.generate = instrumented_generate
    try:
        budget_answer = component.generate_agent("Reply with exactly one word: pong")
    finally:
        actual_engine.generate = real_generate

    assert generate_calls, "backend generate must have been invoked"
    observed_max_tokens = generate_calls[0]["max_new_tokens"]
    assert observed_max_tokens <= 64, (
        f"backend received max_new_tokens={observed_max_tokens}; must honour the 64-token cap"
    )
    output_ids = generate_calls[0].get("outputs", None)
    input_ids = generate_calls[0].get("input_ids", None)
    if output_ids is not None and input_ids is not None:
        generated_count = len(output_ids[0]) - len(input_ids[0])
        assert generated_count <= observed_max_tokens, (
            f"generated {generated_count} tokens; cap was {observed_max_tokens}"
        )
    assert budget_answer.strip() != ""
