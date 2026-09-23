"""Tests for the numeric observation boundary."""

import unittest

from ai_sandbox.contracts import Cell, Observation
from ai_sandbox.episode import MAX_ENERGY
from ai_sandbox.observation_encoding import encode_observation


EMPTY_CELLS = ((Cell.EMPTY,) * 5,) * 5


class ObservationEncodingTests(unittest.TestCase):
    def test_row_major_cell_values_and_length(self) -> None:
        observation = Observation(
            cells=(
                (Cell.OUT_OF_BOUNDS, Cell.EMPTY, Cell.FOOD, Cell.EMPTY, Cell.EMPTY),
                (Cell.FOOD, Cell.EMPTY, Cell.EMPTY, Cell.OUT_OF_BOUNDS, Cell.EMPTY),
                (Cell.EMPTY, Cell.FOOD, Cell.EMPTY, Cell.EMPTY, Cell.EMPTY),
                (Cell.EMPTY, Cell.EMPTY, Cell.OUT_OF_BOUNDS, Cell.EMPTY, Cell.FOOD),
                (Cell.EMPTY, Cell.EMPTY, Cell.EMPTY, Cell.FOOD, Cell.EMPTY),
            ),
            energy=50,
        )

        encoded = encode_observation(observation)

        self.assertIsInstance(encoded, tuple)
        self.assertEqual(len(encoded), 26)
        self.assertEqual(encoded, (
            -1.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.5,
        ))

    def test_energy_normalization_uses_episode_maximum(self) -> None:
        self.assertEqual(MAX_ENERGY, 100)
        for energy, expected in ((100, 1.0), (50, 0.5), (0, 0.0)):
            with self.subTest(energy=energy):
                encoded = encode_observation(Observation(EMPTY_CELLS, energy))
                self.assertEqual(encoded, (0.0,) * 25 + (expected,))

    def test_deterministic_and_does_not_change_observation(self) -> None:
        observation = Observation(EMPTY_CELLS, 73)
        before = Observation(observation.cells, observation.energy)

        first = encode_observation(observation)
        second = encode_observation(observation)

        self.assertEqual(first, second)
        self.assertEqual(observation, before)

    def test_only_observation_is_required(self) -> None:
        # Construct directly, without a World, Episode, position, or seed.
        observation = Observation(EMPTY_CELLS, 0)
        self.assertEqual(encode_observation(observation), (0.0,) * 26)


if __name__ == "__main__":
    unittest.main()
