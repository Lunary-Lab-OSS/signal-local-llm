"""
Router Strategy Factory

Factory for creating router strategies based on configuration.
"""

import logging
from pathlib import Path

from signal_llm.config import LLMConfig

from .base import RouterStrategy

logger = logging.getLogger(__name__)


class RouterStrategyFactory:
    """
    Factory for creating router strategies.

    Usage:
        factory = RouterStrategyFactory(config, device, platform)
        strategy = factory.create_strategy()
        controller = strategy.load_controller()
    """

    def __init__(self, config: LLMConfig, device: str, platform: str, models_dir=None):
        """
        Initialize factory.

        Args:
            config: LLM configuration (contains router settings)
            device: Device string (e.g., "cuda:0", "mps", "cpu")
            platform: Platform string (e.g., "windows", "macos")
            models_dir: Optional models directory path for loading checkpoints
        """
        self.config = config
        self.device = device
        self.platform = platform
        self.models_dir = models_dir

    def create_strategy(self) -> RouterStrategy:
        """
        Create router strategy based on configuration.

        The router name is normalised exactly once here (R13); downstream
        strategies receive the canonical value and never re-parse the raw
        configuration.

        Returns:
            RouterStrategy instance

        Raises:
            ValueError: If router type is unknown or unsupported
        """
        router_type = (self.config.router_type or "llm").strip().lower()

        if router_type == "routellm":
            router_name = (self.config.routellm_router_name or "mf").strip().lower()

            if router_name == "mf":
                from .matrix_factorization import MatrixFactorizationRouterStrategy

                logger.info("Using Matrix Factorization router strategy")
                return MatrixFactorizationRouterStrategy(self.config, self.device, self.platform)
            elif router_name == "sota":
                from .sota import SotaRouterStrategy

                logger.info("Using SOTA (Single-Tower Scalar Regression) router strategy")
                strategy = SotaRouterStrategy(self.config, self.device, self.platform)
                if self.models_dir:
                    strategy.models_dir = Path(self.models_dir)
                return strategy
            else:
                raise ValueError(
                    f"Unsupported router name '{router_name}' for router_type='routellm'. "
                    "Supported: 'mf' (Matrix Factorization), "
                    "'sota' (Single-Tower Scalar Regression)."
                )
        elif router_type == "llm":
            raise ValueError(
                "router_type='llm' does not use router strategies. "
                "Use router_type='routellm' to use the router strategy pattern."
            )
        else:
            raise ValueError(f"Unknown router_type '{router_type}'. Must be 'routellm' or 'llm'.")
