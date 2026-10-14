"""Matrix-factorization (``router_name='mf'``) strategy: intentionally disabled.

RouteLLM 0.2.0's ``MatrixFactorizationRouter`` has no ``embedding_model``
argument, and ``MFModel.forward`` calls the OpenAI ``text-embedding-3-small``
API. Substituting local embeddings is not compatible with its trained
projection, even when dimensions match. This strategy therefore fails fast,
before importing RouteLLM or touching credentials, so a configuration that
selects ``mf`` can never silently send prompts to a hosted provider. Use the
SOTA strategy with a trained local checkpoint for local routing.
"""

from typing import NoReturn

from .base import RouterStrategy


class MatrixFactorizationRouterStrategy(RouterStrategy):
    """Fail-fast placeholder for RouteLLM's provider-backed MF router."""

    def load_controller(self) -> NoReturn:
        """Always raise; local MF routing is not supported.

        Raises:
            ValueError: If the configuration does not select ``router_name='mf'``.
            RuntimeError: Always, for a configuration that does select it.
        """
        self.controller = None
        if self.config.routellm_router_name != "mf":
            raise ValueError("MatrixFactorizationRouterStrategy only supports router_name='mf'")
        raise RuntimeError(
            "Local matrix-factorization routing is unsupported: RouteLLM 0.2.0 "
            "does not accept embedding_model and its MF scorer calls OpenAI. "
            "Use SOTA with a trained local checkpoint instead."
        )
