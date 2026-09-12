"""Tests for router strategy factory and linguistic features."""

from __future__ import annotations

import pytest

from signal_llm.config import LLMConfig, ModelConfig


def make_config(**overrides):
    model = ModelConfig(name="m", repo_id="a/b")
    defaults = dict(router_priority=[model], semantic_priority=[model], agent_priority=[model])
    defaults.update(overrides)
    return LLMConfig(**defaults)


def test_router_factory_requires_routellm_type_for_strategies():
    pytest.importorskip("signal_llm.router")
    from signal_llm.router.factory import RouterStrategyFactory

    config = make_config(router_type="llm")
    with pytest.raises(ValueError, match="router_type='routellm'"):
        RouterStrategyFactory(config, device="cpu", platform="linux").create_strategy()


def test_router_factory_unknown_type_raises():
    from signal_llm.router.factory import RouterStrategyFactory

    config = make_config(router_type="bogus")
    with pytest.raises(ValueError, match="Unknown router_type"):
        RouterStrategyFactory(config, device="cpu", platform="linux").create_strategy()


def test_router_factory_rejects_unsupported_router_name():
    from signal_llm.router.factory import RouterStrategyFactory

    config = make_config(router_type="routellm", routellm_router_name="sw_ranking")
    with pytest.raises(ValueError, match="Unsupported router name"):
        RouterStrategyFactory(config, device="cpu", platform="linux").create_strategy()


def test_features_extractor_bounds():
    pytest.importorskip("textstat")
    # Import torch only if already loaded: re-importing after sys.modules pops
    # can double-register torch libraries in-process.
    if "torch" not in __import__("sys").modules:
        pytest.skip("torch not already loaded in this process")
    from signal_llm.router.features import LinguisticFeatureExtractor

    extractor = LinguisticFeatureExtractor()
    simple = extractor.extract(["hi"])
    complex_text = extractor.extract(
        [
            "Notwithstanding the aforementioned considerations, the quintessential "
            "characterization of antidisestablishmentarianism necessitates further evaluation"
        ]
    )
    for value in (*simple, *complex_text):
        assert 0.0 <= value <= 1.0001
    assert sum(complex_text) > sum(simple)
