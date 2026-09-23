"""Numeric encoding of local observations for future controllers."""

from ai_sandbox.constants import MAX_ENERGY
from ai_sandbox.contracts import Cell, Observation


_CELL_VALUES = {
    Cell.EMPTY: 0.0,
    Cell.FOOD: 1.0,
    Cell.OUT_OF_BOUNDS: -1.0,
}


def encode_observation(observation: Observation) -> tuple[float, ...]:
    """Return 25 row-major cell values followed by energy / MAX_ENERGY."""
    return (
        *(_CELL_VALUES[cell] for row in observation.cells for cell in row),
        observation.energy / MAX_ENERGY,
    )
