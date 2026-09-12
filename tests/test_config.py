from signal_llm.config import LLMConfig, ModelConfig, SpeculativeDecodingConfig


def test_model_config_defaults():
    cfg = ModelConfig(name="test", repo_id="org/model")
    assert cfg.quantized is False
    assert cfg.device == "auto"
    assert cfg.thinking_tokens == 0
    assert cfg.backend == "exllamav2"


def test_speculative_decoding_config_defaults():
    cfg = SpeculativeDecodingConfig()
    assert cfg.enabled is True
    assert cfg.num_speculative_tokens == 5


def test_llm_config_defaults():
    model = ModelConfig(name="m", repo_id="a/b")
    cfg = LLMConfig(
        router_priority=[model],
        semantic_priority=[model],
        agent_priority=[model],
    )
    assert cfg.max_tokens == 2048
    assert cfg.router_type == "llm"
    assert cfg.routellm_threshold == 0.5
    assert cfg.semantic_enabled is True
    assert cfg.agent_enabled is True
