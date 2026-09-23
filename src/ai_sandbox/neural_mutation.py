"""Seeded Gaussian mutation of neural-network parameters."""

import numpy as np

from ai_sandbox.neural_network import TinyNeuralNetwork


def mutate_network(
    parent: TinyNeuralNetwork,
    mutation_scale: float,
    seed: int,
) -> TinyNeuralNetwork:
    """Return a child with independent N(0, mutation_scale) noise per scalar."""
    if not np.isfinite(mutation_scale) or mutation_scale < 0:
        raise ValueError("mutation_scale must be a finite, non-negative number")

    rng = np.random.default_rng(seed)
    child_parameters = tuple(
        parameter + rng.normal(0.0, mutation_scale, size=parameter.shape)
        for parameter in parent.parameters()
    )
    return TinyNeuralNetwork(
        parent.input_size,
        parent.hidden_size,
        parent.output_size,
        *child_parameters,
    )
