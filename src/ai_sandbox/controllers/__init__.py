"""Controllers that choose actions from observation contracts."""

from ai_sandbox.controllers.random_controller import RandomController
from ai_sandbox.controllers.rule_based_controller import RuleBasedController

__all__ = ["NeuralController", "RandomController", "RuleBasedController"]


def __getattr__(name: str):
    # Keep NumPy optional at import time for callers using only baseline controllers.
    if name == "NeuralController":
        from ai_sandbox.controllers.neural_controller import NeuralController

        return NeuralController
    raise AttributeError(name)
