"""
LLM Configuration Dataclasses
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single model"""

    name: str
    repo_id: str
    local_path: str | None = None
    quantized: bool = False
    device: str = "auto"
    dtype: str = "float16"
    thinking_tokens: int = 0
    revision: str | None = None
    backend: str = "exllamav2"


@dataclass
class SpeculativeDecodingConfig:
    """Speculative Decoding Configuration"""

    enabled: bool = True
    num_speculative_tokens: int = 5


@dataclass
class LLMConfig:
    """LLM Configuration"""

    router_priority: list[ModelConfig]
    semantic_priority: list[ModelConfig]
    agent_priority: list[ModelConfig]
    semantic_enabled: bool = True
    agent_enabled: bool = True
    max_tokens: int = 2048
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    router_thinking_tokens: int = 0
    semantic_thinking_tokens: int = 0
    agent_thinking_tokens: int = 512
    speculative_decoding: SpeculativeDecodingConfig | None = None
    router_type: str = "llm"
    routellm_router_name: str = "mf"
    routellm_threshold: float = 0.5
    routellm_embedding_model: str = "nomic-ai/modernbert-embed-base"
    routellm_embedding_device: str = "cuda"
    routellm_embedding_dtype: str = "bfloat16"
    routellm_embedding_compile: bool = True
    routellm_checkpoint_path: str | None = None
    routellm_text_dim: int | None = None
