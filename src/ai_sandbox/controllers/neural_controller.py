"""Translate local observations into actions using a supplied numeric network."""

from ai_sandbox.contracts import Action, Observation
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.observation_encoding import encode_observation


# Score indexes are a fixed policy contract, independent of enum iteration order.
ACTION_ORDER = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
    Action.STAY,
)
_ENCODED_INPUT_SIZE = 26  # The existing 5x5 cells followed by one energy value.


class NeuralController:
    """Choose the first highest-scoring action from a supplied network."""

    def __init__(self, network: TinyNeuralNetwork) -> None:
        if network.input_size != _ENCODED_INPUT_SIZE:
            raise ValueError(f"network input_size must be {_ENCODED_INPUT_SIZE}")
        if network.output_size != len(ACTION_ORDER):
            raise ValueError(f"network output_size must be {len(ACTION_ORDER)}")
        self._network = network

    def choose_action(self, observation: Observation) -> Action:
        scores = self._network.forward(encode_observation(observation))
        return ACTION_ORDER[max(range(len(ACTION_ORDER)), key=scores.__getitem__)]
