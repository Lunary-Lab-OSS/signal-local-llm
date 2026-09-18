"""Tests for LLMComponent role loading, sharing, and generation (L06-L12)."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from signal_llm.component import (
    GenerationError,
    LLMComponent,
    LoadedModel,
    extract_answer,
)
from signal_llm.config import LLMConfig, ModelConfig
from signal_llm.loader import LoadMetadata, ModelLoadResult


def _model(name: str, repo: str, **kw) -> ModelConfig:
    return ModelConfig(name=name, repo_id=repo, **kw)


def _config(**overrides) -> LLMConfig:
    defaults = {
        "router_priority": [_model("router", "org/router", backend="transformers")],
        "semantic_priority": [_model("semantic", "org/semantic", backend="transformers")],
        "agent_priority": [_model("agent", "org/agent", backend="transformers")],
        "max_tokens": 512,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


class FakeLoader:
    """Loader returning per-repo canned handles."""

    def __init__(self):
        self.load_calls: list[dict[str, Any]] = []

    def load_llm_model(self, priority, draft_model_path=None, speculative=None):
        self.load_calls.append(
            {"priority": list(priority), "draft": draft_model_path, "spec": speculative}
        )
        requested = priority[0]
        handle = types.SimpleNamespace(
            backend_role=requested.name,
            generate=lambda prompt, **kw: f"[{requested.name}] {prompt}",
        )
        tokenizer = types.SimpleNamespace(chat_template=None)
        return ModelLoadResult(
            (handle, tokenizer),
            requested.name,
            LoadMetadata(
                replace(requested),
                requested.backend,
                Path(requested.local_path or "/tmp/fake").resolve(),
            ),
        )

    def _get_model_path(self, repo_id, revision=None):
        from pathlib import Path

        return Path("/tmp/fake") / (repo_id.replace("/", "--"))


def _component(config=None, loader=None) -> LLMComponent:
    return LLMComponent(
        config or _config(), model_loader=loader or FakeLoader(), device="cpu", platform="linux"
    )


# --------------------------------------------------------------------- #
# Answer extraction (L12)
# --------------------------------------------------------------------- #


def test_extract_answer_removes_complete_think_blocks() -> None:
    assert extract_answer("<think>reasoning</think>Answer", had_thinking=True) == "Answer"


def test_extract_answer_unterminated_think_returns_empty() -> None:
    assert extract_answer("<think>partial reasoning with no end", had_thinking=True) == ""


def test_extract_answer_strips_stray_closer() -> None:
    assert extract_answer("</think>Answer", had_thinking=True) == "Answer"


def test_extract_answer_multiple_blocks_all_removed() -> None:
    assert extract_answer("<think>a</think>One<think>b</think>Two", had_thinking=True) == "OneTwo"


def test_extract_answer_plain_text_without_thinking() -> None:
    assert extract_answer("plain answer", had_thinking=False) == "plain answer"


def test_extract_answer_complete_block_even_without_budget() -> None:
    # Complete blocks are always removed, regardless of the budget flag.
    assert extract_answer("<think>x</think>Answer", had_thinking=False) == "Answer"


def test_extract_answer_reasoning_only_output_is_empty() -> None:
    assert extract_answer("<think>only thoughts</think>", had_thinking=True) == ""


# --------------------------------------------------------------------- #
# Role gating and sharing (L06)
# --------------------------------------------------------------------- #


def test_semantic_disabled_agent_enabled_loads_only_agent() -> None:
    loader = FakeLoader()
    config = _config(agent_enabled=True, semantic_enabled=False)
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_all_models()
    loaded_roles = [call["priority"][0].name for call in loader.load_calls]
    assert "semantic" not in loaded_roles
    assert component.semantic_model is None
    assert component.agent_model is not None


def test_agent_disabled_semantic_enabled_loads_semantic_independently() -> None:
    loader = FakeLoader()
    config = _config(agent_enabled=False, semantic_enabled=True)
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_all_models()
    assert component.semantic_model is not None, (
        "a disabled agent role must not block semantic loading (L06)"
    )
    assert component.agent_model is None


def test_same_repo_same_revision_shares_single_load() -> None:
    loader = FakeLoader()
    shared = _model("shared", "org/big", backend="transformers")
    config = _config(
        semantic_priority=[shared],
        agent_priority=[_model("agent", "org/big", backend="transformers")],
    )
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_agent_model()
    component.load_semantic_model()
    loaded = [call["priority"][0].name for call in loader.load_calls]
    assert loaded == ["agent"], "identical semantic/agent requests must share one load"
    assert component.semantic_model is component.agent_model


def test_same_repo_different_revision_does_not_share() -> None:
    loader = FakeLoader()
    config = _config(
        semantic_priority=[_model("s", "org/big", backend="transformers", revision="v2")],
        agent_priority=[_model("a", "org/big", backend="transformers", revision="v1")],
    )
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_agent_model()
    component.load_semantic_model()
    assert len(loader.load_calls) == 2
    assert component.semantic_model is not component.agent_model


def test_different_backend_does_not_share() -> None:
    loader = FakeLoader()
    config = _config(
        semantic_priority=[_model("s", "org/big", backend="transformers")],
        agent_priority=[_model("a", "org/big", backend="vllm")],
    )
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_agent_model()
    component.load_semantic_model()
    assert len(loader.load_calls) == 2


def test_router_model_skipped_for_non_llm_router_types() -> None:
    for router_type in ("routellm", "coreml"):
        loader = FakeLoader()
        config = _config(router_type=router_type)
        component = LLMComponent(config, model_loader=loader, device="cpu")
        component.load_router_model()
        assert component.router_model is None, (
            f"router_type={router_type} must not load an LLM (L07)"
        )


def test_speculative_config_is_forwarded_to_loader() -> None:
    from signal_llm.config import SpeculativeDecodingConfig

    loader = FakeLoader()
    config = _config(
        speculative_decoding=SpeculativeDecodingConfig(enabled=True, num_speculative_tokens=7)
    )
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_agent_model()
    agent_call = next(c for c in loader.load_calls if c["priority"][0].name == "agent")
    assert agent_call["spec"] is config.speculative_decoding
    assert agent_call["spec"].num_speculative_tokens == 7


# --------------------------------------------------------------------- #
# Budget contract (L11)
# --------------------------------------------------------------------- #


def test_budget_hard_total_is_never_exceeded() -> None:
    component = _component(_config(max_tokens=100))
    budget = component._resolve_budget("agent", thinking_tokens=2000)
    assert budget["answer_tokens"] <= 100
    assert budget["answer_tokens"] + budget["thinking_tokens"] <= 100 + 0  # hard cap


def test_explicit_answer_override_is_capped_by_hard_total() -> None:
    component = _component(_config(max_tokens=50))
    budget = component._resolve_budget("semantic", answer_override=10_000, thinking_tokens=0)
    assert budget["answer_tokens"] <= 50


def test_negative_budget_raises() -> None:
    component = _component()
    with pytest.raises(GenerationError):
        component._resolve_budget("agent", answer_override=-3, thinking_tokens=0)


# --------------------------------------------------------------------- #
# Generation dispatch through backends (L09/L10/L12)
# --------------------------------------------------------------------- #


class _TemplateTokenizer:
    chat_template = "chatml"

    def __init__(self, output_holder: dict | None = None):
        self.template_calls: list[dict] = []
        self._holder = output_holder if output_holder is not None else {}

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, **kwargs})
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user = next(m["content"] for m in messages if m["role"] == "user")
        return f"<sys>{system}</sys><user>{user}</user><assistant>"

    def __call__(self, prompt, **kwargs):
        return {"input_ids": _FakeIds()}

    def decode(self, tokens, **kwargs):
        return self._holder.get("output", "")

    @property
    def eos_token_id(self):
        return 0


class _FakeIds:
    @property
    def shape(self):
        return (1, 3)

    def to(self, device):
        return self


class _RecordingTransformerModel:
    def __init__(self, output="<think>r</think>Final answer", holder=None):
        self.device = "cpu"
        self.generate_calls: list[dict] = []
        self._output = output
        self._holder = holder

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return _FakeGenerateOutput(self._output, holder=self._holder)


class _FakeGenerateOutput:
    def __init__(self, output_text, holder=None):
        self._text = output_text
        self._holder = holder

    def __getitem__(self, idx):
        if self._holder is not None:
            self._holder["output"] = self._text
        return [0, 0, 0, self._text]


def _make_transformers_component(monkeypatch, output="<think>r</think>Final answer"):
    loader = FakeLoader()
    component = LLMComponent(_config(), model_loader=loader, device="cpu")
    holder: dict = {}
    tokenizer = _TemplateTokenizer(holder)
    model = _RecordingTransformerModel(output, holder=holder)

    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    fake_torch.no_grad = _NoGrad
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    loaded = LoadedModel(
        handle=model,
        name="qwen3-test",
        backend="transformers",
        tokenizer=tokenizer,
        source_repo="org/big",
    )
    return component, model, tokenizer, loaded


def test_transformers_generation_applies_system_prompt_once(monkeypatch) -> None:
    component, _model, tokenizer, loaded = _make_transformers_component(monkeypatch)
    budget = component._resolve_budget("agent", thinking_tokens=0)
    result = component._generate(
        loaded, "translate this", budget, system_prompt="SENTINEL-SYSTEM-PROMPT"
    )
    assert len(tokenizer.template_calls) == 1
    messages = tokenizer.template_calls[0]["messages"]
    assert messages[0]["content"] == "SENTINEL-SYSTEM-PROMPT"
    assert result == "Final answer"


def test_transformers_generation_uses_default_system_prompt(monkeypatch) -> None:
    component, _model, tokenizer, loaded = _make_transformers_component(monkeypatch)
    budget = component._resolve_budget("agent", thinking_tokens=0)
    component._generate(loaded, "hello", budget)  # unset -> default prompt
    messages = tokenizer.template_calls[0]["messages"]
    assert messages[0]["content"] == "You are a concise voice assistant."


def test_unterminated_reasoning_is_not_leaked(monkeypatch) -> None:
    component, _model, _tokenizer, loaded = _make_transformers_component(
        monkeypatch, output="<think>the user should hear"
    )
    budget = component._resolve_budget("agent", thinking_tokens=64)
    result = component._generate(loaded, "hello", budget, system_prompt=None)
    assert result == "", "unterminated reasoning must not become the answer (L12)"


def test_generation_error_raised_for_unknown_model_type() -> None:
    component = _component()
    loaded = LoadedModel(handle=object(), name="x", backend="unknown")
    with pytest.raises(GenerationError, match="unsupported model type"):
        component._generate(
            loaded,
            "p",
            {
                "max_tokens": 5,
                "temperature": 0.1,
                "top_p": 0.9,
                "answer_tokens": 5,
                "thinking_tokens": 0,
                "hard_total": 10,
            },
        )


def test_chatml_fallback_preserves_system_prompt() -> None:
    component = _component()
    formatted = component._format_chat(
        tokenizer=None,
        loaded=LoadedModel(handle=object(), name="x", backend="x"),
        prompt="do the thing",
        system_prompt="SENTINEL",
        enable_thinking=False,
    )
    assert "<|im_start|>system\nSENTINEL<|im_end|>" in formatted
    assert "do the thing" in formatted


def test_none_system_prompt_omits_system_block() -> None:
    component = _component()
    formatted = component._format_chat(
        tokenizer=None,
        loaded=LoadedModel(handle=object(), name="x", backend="x"),
        prompt="p",
        system_prompt=None,
        enable_thinking=False,
    )
    assert "system" not in formatted
    assert "<|im_start|>user\np<|im_end|>" in formatted


# --------------------------------------------------------------------- #
# Adversarial-review regressions (load order + identity + think leak)
# --------------------------------------------------------------------- #


def test_load_all_models_with_identical_configs_loads_once() -> None:
    """load_all_models order must share one handle for identical configs."""
    loader = FakeLoader()
    shared = _model("shared", "org/big", backend="transformers")
    config = _config(
        semantic_priority=[shared],
        agent_priority=[_model("agent", "org/big", backend="transformers")],
    )
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.load_all_models()
    loaded = [call["priority"][0].name for call in loader.load_calls]
    # router (default LLM routing) + ONE shared agent/semantic handle.
    assert loaded == ["router", "agent"], f"identical semantic/agent must load once, got {loaded}"
    assert component.semantic_model is component.agent_model


def test_fallback_candidate_identity_is_used_for_sharing() -> None:
    """A fallback-loaded model must carry ITS identity, not priority[0]'s."""
    first = _model("primary", "org/primary", backend="vllm", device="cuda")
    fallback = _model("fallback", "org/fallback", backend="transformers", device="cpu")

    class FallbackLoader:
        def __init__(self):
            self.calls = 0

        def load_llm_model(self, priority, draft_model_path=None, speculative=None):
            self.calls += 1
            winner = fallback if self.calls > 0 else first
            handle = types.SimpleNamespace(generate=lambda p: "ok")
            handle.loaded_model_config = winner
            return ModelLoadResult(
                (handle, None),
                winner.name,
                LoadMetadata(replace(winner), winner.backend, Path("/tmp/fake")),
            )

        def _get_model_path(self, repo_id, revision=None):
            from pathlib import Path

            return Path("/tmp/fake")

    loader = FallbackLoader()
    component = LLMComponent(_config(), model_loader=loader, device="cpu")
    agent = component._load_role("agent", [first, fallback], require_nonempty=True)

    # Identity reflects the candidate that actually loaded.
    assert agent.source_repo == "org/fallback"
    assert agent.backend == "transformers"
    # And it does NOT match a request pinned to the failed first choice.
    assert agent.matches_request(first) is False
    assert agent.matches_request(fallback) is True


@pytest.mark.parametrize("current", [0, 1])
@pytest.mark.parametrize("requested_device", ["cuda", "auto", "cuda:0", "cuda:1"])
def test_indexed_actual_cuda_shares_only_matching_current_device(
    monkeypatch, current, requested_device
):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    shared = _model("shared", "org/shared", backend="transformers", device=requested_device)
    loader = FakeLoader()
    component = LLMComponent(
        _config(agent_priority=[shared], semantic_priority=[shared]),
        model_loader=loader,
        device="cuda",
    )
    component.agent_model = LoadedModel(
        handle=object(),
        name="shared",
        backend="transformers",
        source_repo="org/shared",
        device="cuda:0",
    )
    component.load_semantic_model()
    expected_share = requested_device == "cuda:0" or (
        requested_device in ("cuda", "auto") and current == 0
    )
    assert (component.semantic_model is component.agent_model) is expected_share
    assert len(loader.load_calls) == (0 if expected_share else 1)


@pytest.mark.parametrize("available", [True, False])
def test_cuda_sharing_fails_closed_when_current_device_unavailable(monkeypatch, available):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)

    def unavailable():
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(torch.cuda, "current_device", unavailable)
    loaded = LoadedModel(
        handle=object(), name="m", backend="transformers", source_repo="org/m", device="cuda:0"
    )
    assert not loaded.matches_request(_model("m", "org/m", backend="transformers", device="cuda"))


def test_unterminated_think_leaks_nothing_even_without_budget() -> None:
    """P1 residual: reasoning models can emit <think> when it was disabled."""
    assert extract_answer("<think>partial reasoning", had_thinking=False) == ""


def test_matches_request_distinguishes_quantization_and_device() -> None:
    loaded = LoadedModel(
        handle=object(),
        name="m",
        backend="transformers",
        source_repo="org/m",
        revision=None,
        device="cuda",
        quantized=True,
    )
    cpu_request = _model("m", "org/m", backend="transformers", device="cpu")
    cuda_quant = _model("m", "org/m", backend="transformers", device="cuda", quantized=True)
    assert loaded.matches_request(cuda_quant) is True
    assert loaded.matches_request(cpu_request) is False, (
        "a CUDA-quantized handle must not satisfy a CPU/unquantized request"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("private reasoning</think>Visible", "Visible"),
        ("private reasoning", ""),
        ("private reasoning</thi", ""),
    ],
)
def test_prefilled_reasoning_continuations(raw, expected):
    assert extract_answer(raw, had_thinking=True) == expected


@pytest.mark.parametrize(
    "change",
    [
        {"local_path": "/models/other"},
        {"dtype": "float32"},
        {"trust_remote_code": True, "revision": "a" * 40},
    ],
)
def test_sharing_requires_complete_identity(change):
    component = _component(
        _config(
            agent_priority=[_model("a", "org/shared", backend="transformers")],
            semantic_priority=[_model("s", "org/shared", backend="transformers", **change)],
        )
    )
    component.load_all_models()
    assert component.agent_model is not component.semantic_model


def test_missing_metadata_is_not_guessed():
    from signal_llm.loader import ModelLoadError

    component = _component(
        loader=types.SimpleNamespace(load_llm_model=lambda *a, **kw: (object(), "unknown"))
    )
    with pytest.raises(ModelLoadError, match="metadata"):
        component.load_agent_model()


def test_concurrent_lazy_load_and_close_once():
    from concurrent.futures import ThreadPoolExecutor

    component = _component()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: component.load_agent_model(), range(32)))
    assert len(component.model_loader.load_calls) == 1
    closed = []
    component.agent_model.handle.close = lambda: closed.append(True)
    component.semantic_model = component.agent_model
    component.close()
    component.close()
    assert closed == [True]
    with pytest.raises(GenerationError, match="closed"):
        component.generate_agent("hello")


def test_zero_budget_never_calls_backend(monkeypatch):
    component, model, _, loaded = _make_transformers_component(monkeypatch)
    budget = component._resolve_budget("agent", answer_override=0, thinking_tokens=0)
    assert component._generate(loaded, "hello", budget) == ""
    assert model.generate_calls == []


def test_temperature_zero_uses_greedy_decoding(monkeypatch):
    component, model, tokenizer, loaded = _make_transformers_component(monkeypatch)
    component.config.temperature = 0
    budget = component._resolve_budget("agent", thinking_tokens=0)
    component._generate(loaded, "hello", budget, system_prompt=None)
    assert model.generate_calls[0]["do_sample"] is False
    assert "temperature" not in model.generate_calls[0]
    assert tokenizer.template_calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_template_prefill_drives_extraction_not_budget(monkeypatch):
    component, _, tokenizer, loaded = _make_transformers_component(
        monkeypatch, output="private reasoning</think>Visible"
    )
    tokenizer.apply_chat_template = lambda *a, **kw: "assistant<think>"
    budget = component._resolve_budget("agent", thinking_tokens=0)
    assert component._generate(loaded, "hello", budget) == "Visible"


def test_backend_exception_redacted(monkeypatch, caplog):
    component, model, _, loaded = _make_transformers_component(monkeypatch)

    def fail(**kwargs):
        raise RuntimeError("PRIVATE-PROMPT")

    model.generate = fail
    budget = component._resolve_budget("agent", thinking_tokens=0)
    prompt = "PRIVATE-PROMPT"
    with pytest.raises(GenerationError) as error:
        component._generate(loaded, prompt, budget)
    import traceback

    assert "PRIVATE-PROMPT" not in "".join(traceback.format_exception(error.value))
    assert "PRIVATE-PROMPT" not in caplog.text


def test_raw_generate_adapter_rejected():
    component = _component()
    called = []
    loaded = LoadedModel(types.SimpleNamespace(generate=lambda p: called.append(p)), "raw", "raw")
    with pytest.raises(GenerationError, match="unsupported"):
        component._generate(loaded, "p", component._resolve_budget("agent", thinking_tokens=0))
    assert not called


def test_incompatible_draft_is_not_forwarded():
    from signal_llm.config import SpeculativeDecodingConfig

    component = _component(_config(speculative_decoding=SpeculativeDecodingConfig()))
    component.load_agent_model()
    assert component.model_loader.load_calls[-1]["draft"] is None


def test_draft_uses_actual_local_path():
    from signal_llm.config import SpeculativeDecodingConfig

    shared = _model("m", "org/m", backend="vllm", local_path="/models/draft")
    component = _component(
        _config(
            router_priority=[shared],
            agent_priority=[shared],
            speculative_decoding=SpeculativeDecodingConfig(),
        )
    )
    component.load_agent_model()
    assert component.model_loader.load_calls[-1]["draft"] == Path("/models/draft")


def test_shared_generation_is_serialized(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    component = _component()
    loaded = LoadedModel(object(), "m", "transformers")
    component.agent_model = component.semantic_model = loaded
    entered, release, second_started = (threading.Event() for _ in range(3))
    calls = []

    def generate(*args):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return "answer"

    monkeypatch.setattr(component, "_generate_transformers", generate)

    def semantic():
        second_started.set()
        return component.generate_semantic("second")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(component.generate_agent, "first")
        assert entered.wait(5)
        second = pool.submit(semantic)
        try:
            assert second_started.wait(5)
            assert len(calls) == 1
        finally:
            release.set()
        assert first.result() == second.result() == "answer"
    assert len(calls) == 2
