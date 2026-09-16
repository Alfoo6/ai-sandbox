"""Behavioral tests of the headless episode loop."""

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, call, patch

from ai_sandbox.contracts import Action, Cell, Observation
from ai_sandbox.controllers import RandomController
from ai_sandbox.episode import Episode
from ai_sandbox.runner import EpisodeResult, run_episode
from ai_sandbox.world import Position, World


class RunnerTests(unittest.TestCase):
    def test_runs_through_termination_and_returns_final_metrics(self) -> None:
        episode = Episode(World(Position(12, 12)))
        result = run_episode(episode, RandomController(42))
        self.assertTrue(episode.terminated)
        self.assertEqual(result.ticks_survived, 100)
        self.assertEqual(result.final_energy, 0)
        self.assertEqual(result, EpisodeResult(
            ticks_survived=episode.completed_ticks,
            resources_collected=episode.resources_collected,
            distance_travelled=episode.distance_travelled,
            final_energy=episode.energy,
        ))

    def test_observations_and_controller_actions_are_used_in_order(self) -> None:
        episode = Episode(World(Position(12, 12), {Position(13, 12)}))
        controller = RandomController(42)
        actions = [Action.EAST] + [Action.STAY] * 99
        trace = Mock()
        with (
            patch.object(episode, "observe", wraps=episode.observe) as observe,
            patch.object(controller, "choose_action", side_effect=actions) as choose,
            patch.object(episode, "step", wraps=episode.step) as step,
        ):
            trace.attach_mock(observe, "observe")
            trace.attach_mock(choose, "choose_action")
            trace.attach_mock(step, "step")
            result = run_episode(episode, controller)

        expected_calls = []
        self.assertEqual(choose.call_count, 100)
        for index, (chosen_call, action) in enumerate(zip(choose.call_args_list, actions)):
            observation = chosen_call.args[0]
            self.assertIsInstance(observation, Observation)
            self.assertEqual(observation.energy, 100 - index)
            self.assertIs(observation.cells[2][3], Cell.FOOD if index == 0 else Cell.EMPTY)
            expected_calls.extend([
                call.observe(), call.choose_action(observation), call.step(action)
            ])
        self.assertEqual(trace.mock_calls, expected_calls)
        self.assertEqual(result, EpisodeResult(100, 1, 1, 0))
        self.assertEqual(episode.world.agent_position, Position(13, 12))
        self.assertEqual(episode.world.food_positions, frozenset())

    def test_equivalent_episodes_and_seeds_produce_identical_results(self) -> None:
        food = {Position(11, 12), Position(13, 12), Position(12, 11), Position(12, 13)}
        first = Episode(World(Position(12, 12), food))
        second = Episode(World(Position(12, 12), food))
        self.assertEqual(
            run_episode(first, RandomController(17)),
            run_episode(second, RandomController(17)),
        )
        self.assertEqual(first.world.agent_position, second.world.agent_position)
        self.assertEqual(first.world.food_positions, second.world.food_positions)

    def test_already_terminated_episode_takes_no_further_actions(self) -> None:
        episode = Episode(World(Position(0, 0)))
        for _ in range(100):
            episode.step(Action.STAY)
        controller = RandomController(42)
        with patch.object(controller, "choose_action", side_effect=AssertionError("unexpected action")):
            self.assertEqual(run_episode(episode, controller), EpisodeResult(100, 0, 0, 0))

    def test_result_is_immutable(self) -> None:
        result = run_episode(Episode(World(Position(12, 12))), RandomController(42))
        for name in ("ticks_survived", "resources_collected", "distance_travelled", "final_energy"):
            with self.subTest(field=name), self.assertRaises(FrozenInstanceError):
                setattr(result, name, -1)

    def test_runs_headlessly_without_display_rendering_or_sleep(self) -> None:
        env = os.environ.copy()
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "SDL_VIDEODRIVER"):
            env.pop(name, None)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        script = """
import sys
from unittest.mock import patch

# Block graphical libraries before importing the runner in a fresh process.
sys.modules['pygame'] = None
sys.modules['tkinter'] = None
with patch('time.sleep', side_effect=AssertionError('unexpected sleep')):
    from ai_sandbox.controllers import RandomController
    from ai_sandbox.episode import Episode
    from ai_sandbox.runner import run_episode
    from ai_sandbox.world import Position, World
    result = run_episode(Episode(World(Position(12, 12))), RandomController(42))
    assert result.ticks_survived == 100
    assert result.final_energy == 0
"""
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
