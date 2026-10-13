"""Reject RouteLLM 0.2.0's provider-backed MF scorer without importing it.

MatrixFactorizationRouter has no embedding_model argument. MFModel.forward
calls OpenAI text-embedding-3-small. Substituting local embeddings is not
compatible with its trained projection, even when dimensions match.
"""

from typing import Any

from signal_llm.config import LLMConfig

from .base import RouterStrategy


class MatrixFactorizationRouterStrategy(RouterStrategy):
    def __init__(self, config: LLMConfig, device: str, platform: str):
        super().__init__(config, device, platform)
        self._embedding_model: Any | None = None

    def load_controller(self) -> Any:
        self.controller = None
        self._embedding_model = None
        if self.config.routellm_router_name != "mf":
            raise ValueError("MatrixFactorizationRouterStrategy only supports router_name='mf'")
        raise RuntimeError(
            "Local matrix-factorization routing is unsupported: RouteLLM 0.2.0 "
            "does not accept embedding_model and its MF scorer calls OpenAI. "
            "Use SOTA with a trained local checkpoint instead."
        )
