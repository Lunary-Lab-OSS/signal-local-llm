"""Router strategy exports with lazy backend imports."""

from .base import RouterStrategy
from .factory import RouterStrategyFactory

__all__ = [
    "CoreMLComplexityClassifier",
    "MatrixFactorizationRouterStrategy",
    "RouterStrategy",
    "RouterStrategyFactory",
    "SotaRouterStrategy",
]


def __getattr__(name):
    if name == "CoreMLComplexityClassifier":
        from .coreml_complexity import CoreMLComplexityClassifier

        return CoreMLComplexityClassifier
    if name == "MatrixFactorizationRouterStrategy":
        from .matrix_factorization import MatrixFactorizationRouterStrategy

        return MatrixFactorizationRouterStrategy
    if name == "SotaRouterStrategy":
        from .sota import SotaRouterStrategy

        return SotaRouterStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
