"""Tests for LLMConfig and ModelConfig validation (L16/L17)."""

from __future__ import annotations

import pytest

from signal_llm.config import LLMConfig, ModelConfig, SpeculativeDecodingConfig


def _model(**overrides) -> ModelConfig:
    defaults = {"name": "m", "repo_id": "org/m"}
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _config(**overrides) -> LLMConfig:
    defaults = {
        "router_priority": [_model(name="r", repo_id="org/r")],
        "semantic_priority": [_model(name="s", repo_id="org/s")],
        "agent_priority": [_model(name="a", repo_id="org/a")],
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def test_minimal_config_is_valid() -> None:
    config = _config()
    assert config.router_type == "llm"
    assert config.temperature == 0.7


def test_model_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        _model(backend="tensorrt")


def test_model_config_rejects_bad_device() -> None:
    with pytest.raises(ValueError, match="device"):
        _model(device="tpu")


def test_model_config_accepts_indexed_cuda_device() -> None:
    assert _model(device="cuda:1").device == "cuda:1"


def test_model_config_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        _model(dtype="int4")


def test_model_config_rejects_negative_thinking_tokens() -> None:
    with pytest.raises(ValueError, match="thinking_tokens"):
        _model(thinking_tokens=-1)


def test_model_config_rejects_bool_thinking_tokens() -> None:
    with pytest.raises(ValueError, match="thinking_tokens"):
        _model(thinking_tokens=True)


def test_trust_remote_code_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned revision"):
        _model(trust_remote_code=True)
    assert _model(trust_remote_code=True, revision="a" * 40).trust_remote_code is True


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "abc123", "z" * 40, 123])
def test_remote_code_rejects_mutable_or_invalid_revision(revision):
    with pytest.raises(ValueError, match="immutable"):
        _model(trust_remote_code=True, revision=revision)


@pytest.mark.parametrize("device", ["cpu:0", "mps:0", "auto:1", "cudafoo", "cuda:x", "cuda:-1"])
def test_device_contract_rejects_unsupported_indices(device):
    with pytest.raises(ValueError, match="device"):
        _model(device=device)


def test_enabled_roles_require_nonempty_priority() -> None:
    with pytest.raises(ValueError, match="semantic_priority"):
        _config(semantic_priority=[])


def test_config_rejects_bool_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        _config(max_tokens=True)


def test_config_rejects_negative_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        _config(max_tokens=-5)


def test_config_rejects_nonfinite_temperature() -> None:
    with pytest.raises(ValueError, match="finite"):
        _config(temperature=float("nan"))


def test_config_rejects_out_of_range_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        _config(top_p=1.5)


def test_config_max_new_tokens_cannot_exceed_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        _config(max_tokens=100, max_new_tokens=200)


def test_config_rejects_unknown_router_type() -> None:
    with pytest.raises(ValueError, match="router_type"):
        _config(router_type="psychic")


def test_config_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="routellm_threshold"):
        _config(routellm_threshold=2.0)


def test_config_rejects_non_int_text_dim() -> None:
    with pytest.raises(ValueError, match="routellm_text_dim"):
        _config(routellm_text_dim=0)


def test_speculative_config_validates_tokens() -> None:
    assert SpeculativeDecodingConfig(num_speculative_tokens=8).num_speculative_tokens == 8
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeDecodingConfig(num_speculative_tokens=-1)


def test_config_rejects_non_modelconfig_entries() -> None:
    with pytest.raises(ValueError, match="ModelConfig"):
        _config(agent_priority=["not-a-model"])
