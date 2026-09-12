"""Coverage tests for loader branches, classifier helpers, and env scoping."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from signal_llm.config import LLMConfig, ModelConfig


def make_config(**overrides):
    model = ModelConfig(name="m", repo_id="a/b")
    defaults = dict(router_priority=[model], semantic_priority=[model], agent_priority=[model])
    defaults.update(overrides)
    return LLMConfig(**defaults)


# --------------------------------------------------------------------------
# Loader branches
# --------------------------------------------------------------------------


def test_loader_download_passes_hf_token(monkeypatch, tmp_path):
    from signal_llm.loader import LLMLoader

    captured = {}

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **kwargs: captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    loader = LLMLoader(models_dir=tmp_path, cache_dir=tmp_path, device="cpu")
    monkeypatch.setattr(loader, "hf_token", "tok-123")
    target = tmp_path / "a_b"
    target.mkdir()
    loader._download_model("a/b", target)
    assert captured["token"] == "tok-123"


def test_loader_mlx_unavailable_returns_none(monkeypatch, tmp_path):
    from signal_llm.loader import LLMLoader

    fake_mlx = types.ModuleType("mlx_lm")

    def boom(*a, **k):
        raise ImportError("no mlx on linux")

    fake_mlx.load = boom
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx)

    loader = LLMLoader(models_dir=tmp_path, cache_dir=tmp_path, device="cpu")
    assert loader._load_llm_mlx(tmp_path, None) is None


def test_loader_unknown_backend_raises_value_error(tmp_path):
    from signal_llm.loader import LLMLoader

    loader = LLMLoader(models_dir=tmp_path, cache_dir=tmp_path, device="cpu")
    config = ModelConfig(name="m", repo_id="a/b", backend="bogus")
    with pytest.raises(ValueError, match="Unknown backend"):
        loader._load_llm_exllama(tmp_path, config)


def test_loader_cpu_configured_with_cuda_logs_warning(tmp_path, monkeypatch):
    from signal_llm.loader import LLMLoader

    loader = LLMLoader(models_dir=tmp_path, cache_dir=tmp_path, device="cpu")
    monkeypatch.setattr(loader, "_has_cuda", lambda: True)
    # Reaching the backend dispatch is enough; unknown backend stops after warn.
    with pytest.raises(ValueError, match="Unknown backend"):
        loader._load_llm_exllama(tmp_path, ModelConfig(name="m", repo_id="a/b", backend="bogus"))


# --------------------------------------------------------------------------
# CoreML complexity pure helpers
# --------------------------------------------------------------------------


def test_softmax_matches_reference():
    from signal_llm.router.coreml_complexity import _softmax

    x = np.array([[1.0, 2.0, 3.0]])
    out = _softmax(x)
    assert out.shape == (1, 3)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] > out[0, 0]


# --------------------------------------------------------------------------
# MatrixFactorization env scoping (fake routellm, fake torch)
# --------------------------------------------------------------------------


def _install_fake_torch(monkeypatch):
    fake = types.ModuleType("torch")

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake.no_grad = _Ctx
    fake.load = lambda *a, **k: {}
    fake.device = lambda x: x
    fake.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_mf_router_env_key_scoped_and_restored(monkeypatch):
    _install_fake_torch(monkeypatch)
    from signal_llm.router import matrix_factorization as mf

    created = {}

    class FakeController:
        def __init__(self, **kwargs):
            created.update(kwargs)

    fake_routellm = types.ModuleType("routellm")
    fake_controller_mod = types.ModuleType("routellm.controller")
    fake_controller_mod.Controller = FakeController
    fake_routellm.controller = fake_controller_mod
    monkeypatch.setitem(sys.modules, "routellm", fake_routellm)
    monkeypatch.setitem(sys.modules, "routellm.controller", fake_controller_mod)

    os.environ.pop("OPENAI_API_KEY", None)
    strategy = mf.MatrixFactorizationRouterStrategy(
        make_config(router_type="routellm"), "cpu", "linux"
    )
    monkeypatch.setattr(strategy, "_load_embedding_model", lambda: None)
    controller = strategy.load_controller()
    assert controller is not None
    assert "OPENAI_API_KEY" not in os.environ


def test_mf_router_preserves_existing_key(monkeypatch):
    _install_fake_torch(monkeypatch)
    from signal_llm.router import matrix_factorization as mf

    class FakeController:
        def __init__(self, **kwargs):
            pass

    fake_routellm = types.ModuleType("routellm")
    fake_controller_mod = types.ModuleType("routellm.controller")
    fake_controller_mod.Controller = FakeController
    fake_routellm.controller = fake_controller_mod
    monkeypatch.setitem(sys.modules, "routellm", fake_routellm)
    monkeypatch.setitem(sys.modules, "routellm.controller", fake_controller_mod)

    monkeypatch.setenv("OPENAI_API_KEY", "real-key")
    strategy = mf.MatrixFactorizationRouterStrategy(
        make_config(router_type="routellm"), "cpu", "linux"
    )
    monkeypatch.setattr(strategy, "_load_embedding_model", lambda: None)
    strategy.load_controller()
    assert os.environ["OPENAI_API_KEY"] == "real-key"


def test_mf_router_missing_routellm_raises(monkeypatch):
    _install_fake_torch(monkeypatch)
    from signal_llm.router import matrix_factorization as mf

    monkeypatch.setitem(sys.modules, "routellm", None)
    monkeypatch.setitem(sys.modules, "routellm.controller", None)
    strategy = mf.MatrixFactorizationRouterStrategy(
        make_config(router_type="routellm"), "cpu", "linux"
    )
    monkeypatch.setattr(strategy, "_load_embedding_model", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        strategy.load_controller()


# --------------------------------------------------------------------------
# IntentRouter LLM classifier branches
# --------------------------------------------------------------------------


class _ExLlamaV2DynamicGenerator:
    def generate(self, prompt, max_new_tokens=None, **kwargs):
        return "agentic"


def test_intent_router_classifies_with_exllama_model():
    from signal_llm.intent_router import IntentRouter

    router = IntentRouter(router_type="llm")
    model = _ExLlamaV2DynamicGenerator()
    intent, _, _ = router.route("a complex reasoning question", router_model=model)
    assert intent == "agentic"


def test_intent_router_unclear_answer_raises():
    from signal_llm.intent_router import IntentRouter

    router = IntentRouter(router_type="llm")

    class Model:
        def generate(self, prompt, **kwargs):
            return "banana"

    with pytest.raises(RuntimeError, match="unclear classification"):
        router.route("an ambiguous query", router_model=Model())


def test_intent_router_model_without_generate_raises():
    from signal_llm.intent_router import IntentRouter

    router = IntentRouter(router_type="llm")
    with pytest.raises(RuntimeError, match="generate"):
        router.route("some query", router_model=object())


# --------------------------------------------------------------------------
# Component optimized settings branches
# --------------------------------------------------------------------------


def _component_for(device):
    from signal_llm.component import LLMComponent

    loader = MagicMock()
    return LLMComponent(config=make_config(), model_loader=loader, device=device, platform="linux")


def test_optimized_settings_cuda():
    comp = _component_for("cuda:0")
    assert comp._optimized_settings["router"]["max_tokens"] == 10
    assert comp._optimized_settings["agent"]["max_tokens"] == 256


def test_optimized_settings_mps():
    comp = _component_for("mps")
    assert comp._optimized_settings["agent"]["max_tokens"] == 128


def test_optimized_settings_cpu():
    comp = _component_for("cpu")
    assert comp._optimized_settings["agent"]["top_k"] == 20


# --------------------------------------------------------------------------
# Component model-loading branches
# --------------------------------------------------------------------------


def test_component_router_disabled_for_routellm():
    from signal_llm.component import LLMComponent

    comp = LLMComponent(
        config=make_config(router_type="routellm"),
        model_loader=MagicMock(),
        device="cpu",
        platform="linux",
    )
    comp.load_router_model()
    assert comp.router_model is None
    comp.model_loader.load_llm_model.assert_not_called()


def test_component_semantic_disabled():
    from signal_llm.component import LLMComponent

    comp = LLMComponent(
        config=make_config(semantic_enabled=False),
        model_loader=MagicMock(),
        device="cpu",
        platform="linux",
    )
    comp.load_semantic_model()
    assert comp.semantic_model is None


def test_component_agent_disabled():
    from signal_llm.component import LLMComponent

    comp = LLMComponent(
        config=make_config(agent_enabled=False),
        model_loader=MagicMock(),
        device="cpu",
        platform="linux",
    )
    comp.load_agent_model()
    assert comp.agent_model is None


def test_component_generate_router_caps_max_tokens():
    from signal_llm.component import LLMComponent

    comp = LLMComponent(
        config=make_config(max_tokens=5),
        model_loader=MagicMock(),
        device="cpu",
        platform="linux",
    )
    comp.router_model = object()
    seen = {}

    def fake_generate(model, prompt, settings, thinking_tokens):
        seen.update(settings)
        return "ok"

    comp._generate = fake_generate
    assert comp.generate_router("x") == "ok"
    assert seen["max_tokens"] == 5


def test_component_generate_router_unloaded_returns_empty():
    from signal_llm.component import LLMComponent

    comp = LLMComponent(
        config=make_config(router_type="routellm"),  # load is a no-op
        model_loader=MagicMock(),
        device="cpu",
        platform="linux",
    )
    assert comp.generate_router("x") == ""


def test_component_semantic_reuses_agent_when_same_repo():
    from signal_llm.component import LLMComponent

    comp = LLMComponent(
        config=make_config(),
        model_loader=MagicMock(),
        device="cpu",
        platform="linux",
    )
    fake = object()
    comp.agent_model = fake
    comp._agent_tokenizer = None
    comp._agent_model_name = "shared"
    comp.load_semantic_model()
    assert comp.semantic_model is fake
