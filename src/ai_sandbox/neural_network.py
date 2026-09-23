"""One-hidden-layer numeric network; the planned policy will use 26 -> 12 -> 5."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TinyNeuralNetwork:
    """Compute raw scores with tanh(x @ W1 + b1) @ W2 + b2.

    W1 has shape (input_size, hidden_size), and W2 has shape
    (hidden_size, output_size). The network owns copies of all parameters.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        w1: ArrayLike,
        b1: ArrayLike,
        w2: ArrayLike,
        b2: ArrayLike,
    ) -> None:
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("layer sizes must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._w1 = np.array(w1, dtype=np.float64, copy=True)
        self._b1 = np.array(b1, dtype=np.float64, copy=True)
        self._w2 = np.array(w2, dtype=np.float64, copy=True)
        self._b2 = np.array(b2, dtype=np.float64, copy=True)

        expected_shapes = (
            ("w1", self._w1, (input_size, hidden_size)),
            ("b1", self._b1, (hidden_size,)),
            ("w2", self._w2, (hidden_size, output_size)),
            ("b2", self._b2, (output_size,)),
        )
        for name, parameter, expected in expected_shapes:
            if parameter.shape != expected:
                raise ValueError(f"{name} must have shape {expected}; got {parameter.shape}")

    @classmethod
    def random(
        cls,
        input_size: int = 26,
        hidden_size: int = 12,
        output_size: int = 5,
        seed: int = 42,
    ) -> "TinyNeuralNetwork":
        """Draw every weight and bias uniformly from [-0.1, 0.1)."""
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("layer sizes must be positive")
        rng = np.random.default_rng(seed)
        return cls(
            input_size,
            hidden_size,
            output_size,
            rng.uniform(-0.1, 0.1, (input_size, hidden_size)),
            rng.uniform(-0.1, 0.1, hidden_size),
            rng.uniform(-0.1, 0.1, (hidden_size, output_size)),
            rng.uniform(-0.1, 0.1, output_size),
        )

    def parameters(self) -> tuple[NDArray[np.float64], ...]:
        """Return independent copies of W1, b1, W2, and b2, in that order."""
        return tuple(parameter.copy() for parameter in (
            self._w1, self._b1, self._w2, self._b2
        ))

    def forward(self, inputs: ArrayLike) -> tuple[float, ...]:
        """Return immutable raw output scores without consuming randomness."""
        x = np.asarray(inputs, dtype=np.float64)
        if x.shape != (self.input_size,):
            raise ValueError(f"inputs must have shape ({self.input_size},); got {x.shape}")
        hidden = np.tanh(x @ self._w1 + self._b1)
        outputs = hidden @ self._w2 + self._b2
        return tuple(float(score) for score in outputs)
