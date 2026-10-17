"""Fast synthetic inference and trajectory checks; no real window or training."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ai_sandbox.behavioral_analysis import (
    BOUNDARY_POSITIONS, ENERGIES, OFFSETS, ObservedNeuralController,
    action_scores, analyze_champion, controlled_observation, controlled_probes,
    create_analysis_episode, manhattan, reduces_food_distance, summarize_trace,
    trace_episode, visible_food,
)
from ai_sandbox.contracts import Action, Cell
from ai_sandbox.controllers import NeuralController
from ai_sandbox.diverse_artifacts import save_network
from ai_sandbox.episode import Episode
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.runner import run_episode
from ai_sandbox.saved_champion import write_report
from ai_sandbox.scenarios import create_random_episode
from ai_sandbox.world import Position, World
from scripts.analyze_champion import main
from scripts.watch_saved_champion import main as watch_main


def fixed_network(scores=(0, 0, 1, 0, 0)):
    return TinyNeuralNetwork(26, 12, 5, np.zeros((26, 12)), np.zeros(12),
                             np.zeros((12, 5)), scores)


def synthetic_trace(positions, actions=None, food=None):
    rows = []
    for i, (before, after) in enumerate(zip(positions, positions[1:])):
        foods = [] if food is None else food[i]
        rows.append(dict(tick=i + 1, position=list(before), position_after=list(after),
                         action=actions[i] if actions else 'east', moved=before != after,
                         food_collected=False, visible_food=foods,
                         reduces_food_distance=reduces_food_distance(
                             foods, (after[0] - before[0], after[1] - before[1]))))
    return rows


class ProbeTests(unittest.TestCase):
    def test_all_24_food_offsets_and_energy(self):
        for offset in OFFSETS:
            observation = controlled_observation(Position(12, 12), offset, 75)
            self.assertEqual(visible_food(observation), [list(offset)])
            self.assertEqual(observation.cells[2][2], Cell.EMPTY)
            self.assertEqual(observation.energy, 75)
            self.assertFalse(any(c is Cell.OUT_OF_BOUNDS for row in observation.cells for c in row))
        self.assertEqual(len(OFFSETS), 24)
        for food in ((0, 0), (3, 0), (0, -1)):
            with self.assertRaises(ValueError):
                controlled_observation(Position(0, 0), food, 100)
        for energy in (0, 101, 25.5, True):
            with self.assertRaises(ValueError):
                controlled_observation(Position(12, 12), None, energy)

    def test_scores_and_ties_match_original_without_parameter_changes(self):
        network = fixed_network((3, 3, -2, 0, 1))
        before = network.parameters()
        observation = controlled_observation(Position(12, 12), (1, 0), 100)
        self.assertEqual(action_scores(network, observation), (3, 3, -2, 0, 1))
        observed = ObservedNeuralController(network)
        self.assertEqual(observed.choose_action(observation), Action.NORTH)
        self.assertEqual(observed.choose_action(observation), NeuralController(network).choose_action(observation))
        for first, after in zip(before, network.parameters()):
            np.testing.assert_array_equal(first, after)

    def test_energy_changes_are_real_and_adjacent(self):
        w1, b1, w2, b2 = fixed_network().parameters()
        w1[25, 0] = 1
        w2[0, 0] = 1
        b2[:] = [0, 0.5, -1, -1, -1]
        network = TinyNeuralNetwork(26, 12, 5, w1, b1, w2, b2)
        probes = controlled_probes(network)
        group = [p for p in probes if p['context'] == 'open' and p['food_offset'] == [1, 0]]
        self.assertEqual([p['energy'] for p in group], list(ENERGIES))
        self.assertEqual([p['action'] for p in group], ['north', 'north', 'south', 'south'])
        self.assertEqual([p['changed_from_previous_energy'] for p in group], [None, False, True, False])
        self.assertEqual(len([p for p in probes if p['context'] == 'open']), 100)
        self.assertEqual(len({p['action_scores'][0] for p in group}), 4)

    def test_boundary_geometry_and_collision(self):
        probes = controlled_probes(fixed_network((1, 0, 0, 0, 0)))
        self.assertEqual({p['context'] for p in probes}, set(BOUNDARY_POSITIONS))
        for probe in probes:
            position = Position(*probe['position'])
            food = probe['food_offset']
            observation = controlled_observation(position, None if food is None else tuple(food), probe['energy'])
            for y, row in enumerate(observation.cells):
                for x, cell in enumerate(row):
                    outside = not (0 <= position.x + x - 2 < 25 and 0 <= position.y + y - 2 < 25)
                    self.assertEqual(cell is Cell.OUT_OF_BOUNDS, outside)
            self.assertEqual(probe['collides_with_boundary'], position.y == 0)
            if position.y == 0 and food is not None:
                self.assertFalse(probe['reduces_food_distance'])
        self.assertEqual(len(probes), 484)

    def test_manhattan_and_visible_food_analysis(self):
        self.assertEqual(manhattan((-2, 1), (1, -1)), 5)
        self.assertTrue(reduces_food_distance([[1, 0]], (1, 0)))
        self.assertFalse(reduces_food_distance([[1, 0]], (0, 0)))
        self.assertFalse(reduces_food_distance([[1, 0]], (-1, 0)))
        self.assertIsNone(reduces_food_distance([], (1, 0)))
        # Moving toward a farther food is not enough if the minimum increases.
        self.assertFalse(reduces_food_distance([[0, 1], [2, 2]], (1, 0)))


class MetricTests(unittest.TestCase):
    def test_unique_boundary_and_blocked_excluding_stay(self):
        positions = [(1, 1), (0, 1), (0, 1), (0, 1), (1, 1)]
        trace = synthetic_trace(positions, ['west', 'west', 'stay', 'east'])
        summary = summarize_trace(trace, positions[0])
        self.assertEqual(summary['unique_positions'], 2)
        self.assertEqual(summary['boundary_movement_fraction'], 0.5)
        self.assertEqual(summary['attempted_blocked_moves'], 1)
        self.assertEqual(summary['aba_reversals'], 0)
        self.assertIsNone(summary['food_visible_closer_fraction'])
        self.assertEqual(summarize_trace([], (0, 0))['unique_positions'], 1)
        self.assertIsNone(summarize_trace([], (0, 0))['boundary_movement_fraction'])

    def test_reversals_runs_and_stationary_interruptions(self):
        a, b, c = (1, 1), (2, 1), (3, 1)
        positions = [a, b, a, b, a, a, b, a, c]
        summary = summarize_trace(synthetic_trace(positions), a)
        self.assertEqual(summary['aba_reversals'], 4)
        self.assertEqual(summary['oscillation_runs'], [dict(start_tick=2, end_tick=4, reversal_count=3)])
        isolated = summarize_trace(synthetic_trace([a, b, a]), a)
        self.assertEqual(isolated['aba_reversals'], 1)
        self.assertEqual(isolated['oscillation_runs'], [])

    def test_food_fraction_and_collection_trace(self):
        positions = [(1, 1), (2, 1), (3, 1), (4, 1)]
        trace = synthetic_trace(positions, food=[[[1, 0]], [[-1, 0]], []])
        summary = summarize_trace(trace, positions[0])
        self.assertEqual(summary['food_visible_decisions'], 2)
        self.assertEqual(summary['food_visible_closer_fraction'], 0.5)
        episode = Episode(World(Position(12, 12), {Position(13, 12)}))
        summary, trace = trace_episode(episode, fixed_network())
        self.assertEqual(trace[0]['tick'], 1)
        self.assertEqual(trace[0]['energy'], 100)
        self.assertEqual(trace[0]['energy_after'], 99)
        self.assertTrue(trace[0]['food_collected'])
        self.assertTrue(trace[0]['reduces_food_distance'])
        self.assertEqual(summary['resources_collected'], 1)
        self.assertEqual(summary['ticks_survived'], 100)
        self.assertEqual(summary['attempted_blocked_moves'], 88)

    def test_shared_runner_equivalence_and_no_food_isolation(self):
        original = create_random_episode(1027)
        empty = create_analysis_episode(1027, True)
        self.assertEqual(empty.world.agent_position, original.world.agent_position)
        self.assertEqual(empty.world.food_positions, frozenset())
        self.assertEqual(len(original.world.food_positions), 6)
        summary, trace = trace_episode(create_analysis_episode(1027), fixed_network())
        result = run_episode(create_random_episode(1027), NeuralController(fixed_network()))
        self.assertEqual(summary['ticks_survived'], result.ticks_survived)
        self.assertEqual(summary['resources_collected'], result.resources_collected)
        self.assertEqual(summary['successful_moves'], result.distance_travelled)
        self.assertEqual(trace[-1]['energy_after'], result.final_energy)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(TemporaryDirectory()))
        self.model = self.directory / 'model.npz'
        save_network(self.model, fixed_network())
        for target in ('ai_sandbox.neural_evolution.evolve',
                       'ai_sandbox.diverse_evolution.evolve_diverse',
                       'ai_sandbox.diverse_artifacts.evolve_diverse',
                       'ai_sandbox.neural_mutation.mutate_network'):
            self.stack.enter_context(patch(target, side_effect=AssertionError('training/mutation forbidden')))

    def test_deterministic_json_hash_and_unchanged_model(self):
        before = self.model.read_bytes()
        first = analyze_champion(self.model, 1041, True)
        second = analyze_champion(self.model, 1041, True)
        self.assertEqual(first, second)
        self.assertEqual(first['model_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(first['summary']['ticks_survived'], 100)
        self.assertEqual(first['summary']['resources_collected'], 0)
        self.assertEqual(first['summary']['food_visible_decisions'], 0)
        for name, report in [('one.json', first), ('two.json', second)]:
            write_report(self.directory / name, report)
        output = self.directory / 'one.json'
        self.assertEqual(output.read_bytes(), (self.directory / 'two.json').read_bytes())
        self.assertEqual(json.loads(output.read_text()), first)
        with self.assertRaises(FileExistsError):
            write_report(output, second)
        with self.assertRaises(FileExistsError):
            write_report(self.model, second)
        self.assertEqual(self.model.read_bytes(), before)

    def test_cli_and_overwrite_preflight(self):
        output = self.directory / 'report.json'
        args = ['--model', str(self.model), '--scenario', '1027', '--output', str(output), '--no-food']
        with redirect_stdout(StringIO()):
            main(args)
        self.assertTrue(json.loads(output.read_text())['food_disabled'])
        with patch('scripts.analyze_champion.analyze_champion') as analyze, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                main(args)
            analyze.assert_not_called()
        dangling = self.directory / 'dangling'
        dangling.symlink_to(self.directory / 'absent')
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(['--model', str(self.model), '--scenario', '0', '--output', str(dangling)])

    def test_debug_viewer_delegates_without_window(self):
        with patch('scripts.watch_saved_champion.run_visual_episode') as viewer:
            watch_main(['--model', str(self.model), '--scenario', '1041', '--debug'])
        episode, controller = viewer.call_args.args
        self.assertEqual(viewer.call_args.kwargs, {'scenario_seed': 1041})
        output = StringIO()
        with redirect_stdout(output):
            action = controller.choose_action(episode.observe())
        self.assertEqual(action, NeuralController(fixed_network()).choose_action(episode.observe()))
        for field in ('action=east', 'energy=100', 'visible_food=', 'scores=', 'north', 'stay'):
            self.assertIn(field, output.getvalue())
        with patch('scripts.watch_saved_champion.run_visual_episode') as viewer:
            watch_main(['--model', str(self.model), '--scenario', '1041'])
        self.assertIs(type(viewer.call_args.args[1]), NeuralController)


if __name__ == '__main__':
    unittest.main()
