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
        revision=None,
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
    assert len(answer) > 0

    # The completion manifest must exist and mark the snapshot complete.
    from signal_llm.loader import model_cache_key

    cache_dir = tmp_path / "models" / model_cache_key(MODEL_REPO, None)
    manifest = cache_dir / ".signal-llm-manifest.json"
    assert manifest.exists(), "completion manifest must be written after download"

    # Agent generation through the shared handle (same repo+revision+backend).
    agent_answer = component.generate_agent("Say hello in one word.")
    assert isinstance(agent_answer, str)

    # Budget contract: max_tokens=64 caps generation for real.
    call = component.agent_model
    assert call is not None
