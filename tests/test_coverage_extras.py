"""Coverage for loader/component/router branches not hit elsewhere."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from signal_llm.component import GenerationError, LLMComponent, LoadedModel
from signal_llm.config import LLMConfig, ModelConfig
from signal_llm.loader import LocalLLMModelLoader, ModelLoadError


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


# --------------------------------------------------------------------- #
# Loader download behaviour with a fake hub
# --------------------------------------------------------------------- #


class _FakeHubModule:
    def __init__(self):
        self.calls: list[dict] = []

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        (Path(kwargs["local_dir"]) / "config.json").write_text("{}", encoding="utf-8")
        return "sha"


@pytest.fixture
def fake_hub(monkeypatch):
    hub = _FakeHubModule()
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = hub.snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return hub


def test_download_forwards_token_and_revision(fake_hub, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    loader = LocalLLMModelLoader(models_dir=tmp_path / "m", cache_dir=tmp_path / "c", device="cpu")
    local = loader._get_model_path("org/m", "v9")
    loader._ensure_downloaded("org/m", local, "v9")
    call = fake_hub.calls[0]
    assert call["repo_id"] == "org/m"
    assert call["revision"] == "v9"
    assert call["token"] == "hf_test_token"
    # The huggingface_hub-1.x-removed argument must never be passed (L05/D).
    assert "local_dir_use_symlinks" not in call


def test_download_without_token_omits_it(fake_hub, tmp_path, monkeypatch) -> None:
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    loader = LocalLLMModelLoader(models_dir=tmp_path / "m", cache_dir=tmp_path / "c", device="cpu")
    local = loader._get_model_path("org/m", None)
    loader._ensure_downloaded("org/m", local, None)
    assert "token" not in fake_hub.calls[0]


# --------------------------------------------------------------------- #
# Component generation with a fake transformers stack
# --------------------------------------------------------------------- #


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        self.no_grad = _NoGrad
        self.cuda = types.SimpleNamespace(is_available=lambda: False)


class _RecordingTokenizer:
    chat_template = "chatml"

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return "<formatted>"

    def __call__(self, prompt, **kwargs):
        return types.SimpleNamespace(input_ids=_FakeIds())

    @property
    def eos_token_id(self):
        return 0


class _FakeIds:
    @property
    def shape(self):
        return (1, 3)

    def to(self, device):
        return self


class _FakeGenerateOutput:
    def __init__(self, output_text):
        self._text = output_text
        self.sliced = None

    def __getitem__(self, idx):
        self.sliced = idx
        return self._text


class _FakeTransformersModel:
    def __init__(self, output="hello"):
        self.device = "cpu"
        self.generate_kwargs: list[dict] = []
        self._output = output

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
        return _FakeGenerateOutput(self._output)


class _StaticTokenizer:
    """Tokenizer whose decode() returns a fixed string."""

    chat_template = "chatml"

    def __init__(self, decoded):
        self._decoded = decoded
        self.calls: list[Any] = []

    def apply_chat_template(self, messages, **kwargs):
        return "<formatted>"

    def __call__(self, prompt, **kwargs):
        self.calls.append(prompt)
        return {"input_ids": _FakeIds()}

    def decode(self, tokens, **kwargs):
        return self._decoded

    eos_token_id = 0


@pytest.fixture
def fake_torch(monkeypatch):
    module = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def test_transformers_backend_uses_configured_temperature(fake_torch, monkeypatch) -> None:
    config = _config(temperature=0.25, top_p=0.5)
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    component = LLMComponent(config, model_loader=loader, device="cpu")

    model = _FakeTransformersModel()
    tokenizer = _StaticTokenizer("the answer")
    loaded = LoadedModel(handle=model, name="m", backend="transformers", tokenizer=tokenizer)
    budget = component._resolve_budget("agent", thinking_tokens=0)
    result = component._generate(loaded, "hello", budget, system_prompt="sys")
    assert result == "the answer"
    call = model.generate_kwargs[0]
    assert call["temperature"] == 0.25, "config temperature must reach the backend (L16)"
    assert call["top_p"] == 0.5
    assert call["max_new_tokens"] == budget["answer_tokens"]


def test_transformers_backend_hard_total_caps_generation(fake_torch) -> None:
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    component = LLMComponent(_config(max_tokens=64), model_loader=loader, device="cpu")
    model = _FakeTransformersModel()
    tokenizer = _StaticTokenizer("x")
    loaded = LoadedModel(handle=model, name="m", backend="transformers", tokenizer=tokenizer)
    budget = component._resolve_budget("agent", thinking_tokens=500)
    assert budget["answer_tokens"] + budget["thinking_tokens"] <= 64
    component._generate(loaded, "hello", budget, system_prompt=None)
    assert model.generate_kwargs[0]["max_new_tokens"] <= 64


def test_phi4_reasoning_zeroes_thinking_budget(monkeypatch) -> None:
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    config = _config(agent_priority=[_model("phi-4-reasoning", "org/phi-4-reasoning")])
    component = LLMComponent(config, model_loader=loader, device="cpu")
    component.agent_model = LoadedModel(object(), "phi-4-reasoning", "transformers")
    budgets = []

    def generate(loaded, prompt, budget, **kwargs):
        budgets.append(budget)
        return "answer"

    monkeypatch.setattr(component, "_generate", generate)
    assert component.generate_agent("question", thinking_tokens=999) == "answer"
    assert budgets[0]["thinking_tokens"] == 0


# --------------------------------------------------------------------- #
# Chat-model detection (L10)
# --------------------------------------------------------------------- #


def test_tokenizer_without_template_is_authoritative() -> None:
    from signal_llm.component import _is_chat_model

    tokenizer = types.SimpleNamespace(chat_template=None)
    # Explicitly no template -> NOT a chat model, whatever the name says.
    assert _is_chat_model("qwen3-instruct", tokenizer) is False


def test_tokenizer_with_template_is_authoritative() -> None:
    from signal_llm.component import _is_chat_model

    tokenizer = types.SimpleNamespace(chat_template="{{messages}}")
    assert _is_chat_model("totally-unknown-name", tokenizer) is True


def test_name_markers_used_only_without_tokenizer() -> None:
    from signal_llm.component import _is_chat_model

    assert _is_chat_model("qwen3-something") is True
    assert _is_chat_model("random-model-xyz") is False


# --------------------------------------------------------------------- #
# GenerationError surfaces (L19)
# --------------------------------------------------------------------- #


def test_generate_router_unloaded_without_priority_raises() -> None:
    loader = types.SimpleNamespace(load_llm_model=lambda *a, **k: None)
    config = _config(router_type="routellm")
    component = LLMComponent(config, model_loader=loader, device="cpu")
    # routellm never loads an LLM; generate_router must fail clearly.
    with pytest.raises(GenerationError):
        component.generate_router("classify this")


def test_model_load_error_names_every_candidate(tmp_path) -> None:
    class FailingLoader(LocalLLMModelLoader):
        def _load_one(self, model_config, draft, spec):
            raise RuntimeError("boom")

    failing = FailingLoader(models_dir=tmp_path / "models", cache_dir=tmp_path / "cache")
    candidates = [
        _model("one", "org/one", backend="transformers"),
        _model("two", "org/two", backend="transformers"),
    ]
    with pytest.raises(ModelLoadError) as excinfo:
        failing.load_llm_model(candidates)
    assert "one" in str(excinfo.value) and "two" in str(excinfo.value)
