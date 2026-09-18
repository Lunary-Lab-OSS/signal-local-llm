"""Offline regressions for loader/component integration boundaries."""

from types import SimpleNamespace

import pytest

from signal_llm.component import LLMComponent, LoadedModel, extract_answer
from signal_llm.config import LLMConfig, ModelConfig
from signal_llm.loader import LocalLLMModelLoader


@pytest.mark.parametrize(
    ("backend", "device", "cuda", "expected"),
    [
        ("transformers", "auto", False, "cpu"),
        ("transformers", "auto", True, "cuda"),
        ("transformers", "cuda:1", True, "cuda:1"),
        ("vllm", "auto", True, "cuda"),
        ("exllamav2", "auto", True, "cuda"),
        ("mlx", "auto", False, "mps"),
    ],
)
def test_resolved_device_survives_immutable_handle(
    tmp_path, monkeypatch, backend, device, cuda, expected
):
    loader = LocalLLMModelLoader(tmp_path / "models", tmp_path / "cache", device=device)
    monkeypatch.setattr(loader, "_has_cuda", lambda: cuda)
    monkeypatch.setattr(loader, "_load_one", lambda *args: (object(), object()))
    model = ModelConfig(name="model", repo_id="org/model", backend=backend)
    result = loader.load_llm_model([model])
    assert result.metadata.device == expected
    config = LLMConfig(router_priority=[model], semantic_priority=[model], agent_priority=[model])
    component = LLMComponent(config, model_loader=loader)
    component.load_agent_model()
    assert component.agent_model.device == expected


def test_actual_engine_device_overrides_requested_default(tmp_path, monkeypatch):
    loader = LocalLLMModelLoader(tmp_path / "models", tmp_path / "cache", device="cuda")
    monkeypatch.setattr(loader, "_load_one", lambda *args: SimpleNamespace(device="cuda:1"))
    result = loader.load_llm_model(
        [ModelConfig(name="model", repo_id="org/model", backend="transformers")]
    )
    assert result.metadata.device == "cuda:1"


def test_unprefilled_closer_does_not_expose_reasoning():
    assert extract_answer("private reasoning</think>answer", had_thinking=False) == "answer"


@pytest.mark.parametrize("prefilled", [False, True])
def test_nested_reasoning_does_not_leak_after_inner_closer(prefilled):
    raw = "<think>outer<think>inner</think>still private</think>answer"
    assert extract_answer(raw, had_thinking=prefilled) == "answer"


def test_transformers_preserves_special_reasoning_delimiters():
    import torch

    class Tokenizer:
        chat_template = None
        eos_token = "<eos>"
        eos_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1]])}

        def decode(self, tokens, *, skip_special_tokens):
            assert tokens.tolist() == [2, 3, 4, 5, 0]
            if skip_special_tokens:
                return "private reasoninganswer"
            return "<think>private reasoning</think>answer<eos>"

    model = SimpleNamespace(
        device="cpu", generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5, 0]])
    )
    config = LLMConfig(
        router_priority=[],
        semantic_priority=[],
        agent_priority=[],
        router_type="coreml",
        semantic_enabled=False,
        agent_enabled=False,
    )
    component = LLMComponent(config, model_loader=object())
    loaded = LoadedModel(model, "model", "transformers", Tokenizer())
    budget = component._resolve_budget("agent", thinking_tokens=0)
    assert component._generate(loaded, "hello", budget) == "answer"
