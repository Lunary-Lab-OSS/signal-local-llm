"""
Router Strategy Base Class

Abstract base class for all router strategies.
All router implementations must inherit from this class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from signal_llm.config import LLMConfig

logger = logging.getLogger(__name__)


class RouterStrategy(ABC):
    """
    Abstract base class for router strategies.

    All router implementations must:
    1. Inherit from this class
    2. Implement load_controller() to create and return a RouteLLM controller
    3. Handle their own model loading and device management
    """

    def __init__(self, config: LLMConfig, device: str, platform: str):
        """
        Initialize router strategy.

        Args:
            config: LLM configuration (contains router settings)
            device: Device string (e.g., "cuda:0", "mps", "cpu")
            platform: Platform string (e.g., "windows", "macos")
        """
        self.config = config
        self.device = device
        self.platform = platform
        self.controller: Any | None = None

    @abstractmethod
    def load_controller(self) -> Any:
        """
        Load and return the RouteLLM controller.

        This method should:
        - Load embedding models if needed
        - Create and configure the RouteLLM controller
        - Set self.controller to the created controller
        - Handle any router-specific initialization

        Returns:
            RouteLLM Controller instance
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if controller is loaded."""
        return self.controller is not None

    def get_name(self) -> str:
        """Get strategy name for logging."""
        return self.__class__.__name__

    def get_controller(self) -> Any:
        """
        Get the loaded controller.

        Returns:
            RouteLLM Controller instance

        Raises:
            RuntimeError: If controller is not loaded
        """
        if self.controller is None:
            raise RuntimeError("Router controller not loaded. Call load_controller() first.")
        return self.controller
