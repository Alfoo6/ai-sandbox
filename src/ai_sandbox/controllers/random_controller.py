"""A uniformly random baseline controller with its own RNG."""

from random import Random

from ai_sandbox.contracts import Action, Observation


_ACTIONS = tuple(Action)


class RandomController:
    def __init__(self, seed: int) -> None:
        self._rng = Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        """Choose uniformly from all actions, without inspecting observation."""
        return self._rng.choice(_ACTIONS)
