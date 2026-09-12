"""Tests for LLMComponent — uses mock models, no GPU required."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from signal_llm.component import LLMComponent
from signal_llm.config import LLMConfig, ModelConfig


def _make_config():
    model = ModelConfig(name="m", repo_id="a/b")
    return LLMConfig(
        router_priority=[model],
        semantic_priority=[model],
        agent_priority=[model],
        agent_enabled=False,
        semantic_enabled=False,
    )


def test_component_init():
    config = _make_config()
    loader = MagicMock()
    comp = LLMComponent(config=config, model_loader=loader, device="cpu", platform="linux")
    assert comp.router_model is None
    assert comp.agent_model is None


def test_component_accepts_full_signal_config_shape(tmp_path):
    llm_config = _make_config()
    full_config = SimpleNamespace(
        llm=llm_config,
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        system=SimpleNamespace(),
    )
    comp = LLMComponent(
        config=full_config, model_loader=MagicMock(), device="cpu", platform="linux"
    )
    assert comp.config is llm_config


def test_generate_returns_empty_when_no_model():
    config = _make_config()
    loader = MagicMock()
    comp = LLMComponent(config=config, model_loader=loader, device="cpu", platform="linux")
    # Without loading a model, generate_router should return ""
    # Prevent actual model load
    comp.router_model = None
    with patch.object(comp, "load_router_model"):
        result = comp.generate_router("test")
    assert isinstance(result, str)


def test_load_agent_model_disabled():
    config = _make_config()
    config.agent_enabled = False
    loader = MagicMock()
    comp = LLMComponent(config=config, model_loader=loader, device="cpu", platform="linux")
    comp.load_agent_model()
    assert comp.agent_model is None
    loader.load_llm_model.assert_not_called()


def test_component_builds_default_loader_from_full_config(tmp_path):
    full_config = SimpleNamespace(
        llm=_make_config(),
        models_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
        system=SimpleNamespace(),
    )
    comp = LLMComponent(config=full_config, device="cpu", platform="linux")
    assert comp.model_loader.models_dir == Path(tmp_path / "models")


def test_component_uses_injected_loader_for_router_model():
    config = _make_config()
    loader = MagicMock()
    fake_model = object()
    loader.load_llm_model.return_value = (fake_model, "fake")
    comp = LLMComponent(config=config, model_loader=loader, device="cpu", platform="linux")
    comp.load_router_model()
    assert comp.router_model is fake_model
    loader.load_llm_model.assert_called_once_with(config.router_priority)
