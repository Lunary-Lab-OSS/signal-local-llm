"""Tests for the router strategy factory (R13)."""

from __future__ import annotations

import pytest

from signal_llm.config import LLMConfig, ModelConfig
from signal_llm.router.factory import RouterStrategyFactory


def _model(name: str) -> ModelConfig:
    return ModelConfig(name=name, repo_id=f"org/{name}", backend="transformers")


def _routellm_config(**overrides) -> LLMConfig:
    defaults = {
        "router_priority": [_model("r")],
        "semantic_priority": [_model("s")],
        "agent_priority": [_model("a")],
        "router_type": "routellm",
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def test_factory_creates_mf_strategy() -> None:
    strategy = RouterStrategyFactory(_routellm_config(), "cpu", "linux").create_strategy()
    assert type(strategy).__name__ == "MatrixFactorizationRouterStrategy"


def test_factory_normalises_router_name_case() -> None:
    config = _routellm_config(routellm_router_name="MF")
    strategy = RouterStrategyFactory(config, "cpu", "linux").create_strategy()
    assert type(strategy).__name__ == "MatrixFactorizationRouterStrategy"


def test_factory_creates_sota_strategy(tmp_path) -> None:
    config = _routellm_config(routellm_router_name="sota")
    strategy = RouterStrategyFactory(config, "cpu", "linux", models_dir=tmp_path).create_strategy()
    assert type(strategy).__name__ == "SotaRouterStrategy"
    assert strategy.models_dir == tmp_path


def test_factory_rejects_unknown_router_name() -> None:
    config = _routellm_config(routellm_router_name="psychic")
    with pytest.raises(ValueError, match="Unsupported router name"):
        RouterStrategyFactory(config, "cpu", "linux").create_strategy()


def test_factory_rejects_unknown_router_type() -> None:
    config = _routellm_config()
    object.__setattr__(config, "router_type", "psychic")
    with pytest.raises(ValueError, match="Unknown router_type"):
        RouterStrategyFactory(config, "cpu", "linux").create_strategy()


def test_factory_rejects_llm_router_type() -> None:
    config = _routellm_config(router_type="llm")
    with pytest.raises(ValueError, match="does not use router strategies"):
        RouterStrategyFactory(config, "cpu", "linux").create_strategy()
