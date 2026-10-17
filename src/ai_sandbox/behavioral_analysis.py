"""Inference-only probes and trace metrics; never alter policy or mechanics."""

from dataclasses import replace
import hashlib
from pathlib import Path
import platform

import numpy as np

from ai_sandbox.contracts import Action, Cell, Observation
from ai_sandbox.controllers import NeuralController
from ai_sandbox.controllers.neural_controller import ACTION_ORDER
from ai_sandbox.episode import Episode
from ai_sandbox.neural_network import TinyNeuralNetwork
from ai_sandbox.observation_encoding import encode_observation
from ai_sandbox.runner import episode_ticks
from ai_sandbox.saved_champion import load_champion
from ai_sandbox.scenarios import create_random_episode
from ai_sandbox.world import Position, World


ENERGIES = (100, 75, 50, 25)
OFFSETS = tuple((x, y) for y in range(-2, 3) for x in range(-2, 3) if (x, y) != (0, 0))
BOUNDARY_POSITIONS = {
    "open": Position(12, 12),
    "north": Position(12, 0), "south": Position(12, 24),
    "east": Position(24, 12), "west": Position(0, 12),
    "north_east": Position(24, 0), "north_west": Position(0, 0),
    "south_east": Position(24, 24), "south_west": Position(0, 24),
}
DELTAS = {
    Action.NORTH: (0, -1), Action.SOUTH: (0, 1),
    Action.EAST: (1, 0), Action.WEST: (-1, 0), Action.STAY: (0, 0),
}
METRIC_DEFINITIONS = {
    "ticks_survived": "Completed steps, including the fatal tick.",
    "resources_collected": "Number of steps collecting a food cell.",
    "unique_positions": "Distinct initial and post-step positions, including the terminal position.",
    "boundary_movement_fraction": "Successful moves ending at x or y = 0 or 24 / all successful moves; null if none.",
    "attempted_blocked_moves": "Non-stay decisions whose position does not change.",
    "aba_reversals": "Consecutive position triples A,B,A with A != B; stays/blocked steps interrupt these triples.",
    "oscillation_runs": "Maximal runs of at least two consecutive ABA reversals (ABAB or longer); start/end tick delimit reversal decisions, reversal_count counts overlapping triples.",
    "food_visible_decisions": "Pre-step observations containing at least one food cell.",
    "food_visible_closer_fraction": "Food-visible decisions with strictly smaller minimum Manhattan distance after actual movement to the SAME pre-step visible food set / food-visible decisions; null if none.",
}
LIMITATIONS = [
    "Raw scores are not probabilities or calibrated confidence; first maximum wins.",
    "Closer is a one-step geometric measure, not a guarantee of eventual collection.",
    "Reversals and oscillations are descriptive, not necessarily errors.",
    "Boundary movement counts destinations, not time occupancy or movement parallel to a wall.",
    "Single-food probes omit multi-food interactions; boundary comparisons require matching food offset and energy.",
    "Observed actions do not establish intent or an inferred strategy; one episode does not establish generalization.",
    "No-food removes food cues and energy replenishment together; it does not isolate their causal effects.",
    "Determinism assumes identical model bytes, path, code and runtime; cross-version numerical equality is not guaranteed.",
    "Using final-test scenarios to guide future changes makes them development data.",
]


def manhattan(first, second) -> int:
    """L1 distance between two (x, y) pairs."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def visible_food(observation: Observation) -> list[list[int]]:
    """Relative [dx, dy] pairs in row-major order, north negative y."""
    return [[x - 2, y - 2] for y, row in enumerate(observation.cells)
            for x, cell in enumerate(row) if cell is Cell.FOOD]


def reduces_food_distance(food, displacement) -> bool | None:
    """Compare minima over the unchanged pre-decision food set; None if empty."""
    if not food:
        return None
    return min(manhattan(displacement, p) for p in food) < min(
        manhattan((0, 0), p) for p in food
    )


def controlled_observation(position: Position, food: tuple[int, int] | None,
                           energy: int) -> Observation:
    """Build physically valid local geometry through the real observation path."""
    if type(energy) is not int or not 1 <= energy <= 100:
        raise ValueError("probe energy must be an integer from 1 to 100")
    if food is not None and food not in OFFSETS:
        raise ValueError("food must be a non-center radius-2 offset")
    foods = set() if food is None else {Position(position.x + food[0], position.y + food[1])}
    observation = Episode(World(position, foods)).observe()
    return replace(observation, energy=energy)


def action_scores(network: TinyNeuralNetwork, observation: Observation) -> tuple[float, ...]:
    """Untransformed outputs in ACTION_ORDER; no softmax or rounding."""
    return network.forward(encode_observation(observation))


class ObservedNeuralController:
    """Delegate decisions to the original controller and retain read-only diagnostics.

    Scores are recomputed through the same pure forward function, avoiding any
    change to the existing controller or network. Neither call consumes RNG state.
    """

    def __init__(self, network: TinyNeuralNetwork, *, debug: bool = False):
        self.network = network
        self.controller = NeuralController(network)
        self.debug = debug
        self.decision: dict = {}

    def choose_action(self, observation: Observation) -> Action:
        action = self.controller.choose_action(observation)
        self.decision = {
            "energy": observation.energy,
            "visible_food": visible_food(observation),
            "action": action.value,
            "action_scores": list(action_scores(self.network, observation)),
        }
        if self.debug:
            print(f"action={action.value} energy={observation.energy} "
                  f"visible_food={len(self.decision['visible_food'])} "
                  f"scores={dict(zip((a.value for a in ACTION_ORDER), self.decision['action_scores']))}",
                  flush=True)
        return action


def controlled_probes(network: TinyNeuralNetwork) -> list[dict]:
    """All feasible single-food offsets plus a no-food control at nine locations.

    Every condition uses all four energies. Impossible food/wall combinations
    are omitted, never silently relocated. Compare equal offsets and energies.
    """
    controller = ObservedNeuralController(network)
    results = []
    for context, position in BOUNDARY_POSITIONS.items():
        for food in (None, *OFFSETS):
            if food is not None and not (0 <= position.x + food[0] < World.SIZE
                                        and 0 <= position.y + food[1] < World.SIZE):
                continue
            previous_action = None
            for energy in ENERGIES:
                observation = controlled_observation(position, food, energy)
                action = controller.choose_action(observation)
                dx, dy = DELTAS[action]
                blocked = observation.cells[2 + dy][2 + dx] is Cell.OUT_OF_BOUNDS
                results.append({
                    "context": context, "position": [position.x, position.y],
                    "food_offset": None if food is None else list(food),
                    **controller.decision,
                    "collides_with_boundary": blocked,
                    "reduces_food_distance": reduces_food_distance(
                        controller.decision["visible_food"], (0, 0) if blocked else (dx, dy)),
                    "changed_from_previous_energy": None if previous_action is None else action != previous_action,
                })
                previous_action = action
    return results


def create_analysis_episode(seed: int, no_food: bool = False) -> Episode:
    """Reconstruct an isolated empty world at the seeded scenario's start."""
    episode = create_random_episode(seed)
    return Episode(World(episode.world.agent_position)) if no_food else episode


def summarize_trace(trace: list[dict], initial_position: list[int]) -> dict:
    """Compute metrics using all steps; do not collapse stationary positions."""
    positions = [tuple(initial_position)] + [tuple(row["position_after"]) for row in trace]
    reversals = [i for i in range(2, len(positions))
                 if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]]
    runs = []
    for tick in reversals:
        if runs and tick == runs[-1]["end_tick"] + 1:
            runs[-1]["end_tick"] = tick
            runs[-1]["reversal_count"] += 1
        else:
            runs.append(dict(start_tick=tick, end_tick=tick, reversal_count=1))
    moves = sum(row["moved"] for row in trace)
    boundary_moves = sum(row["moved"] and any(c in (0, World.SIZE - 1) for c in row["position_after"])
                         for row in trace)
    food_decisions = sum(bool(row["visible_food"]) for row in trace)
    return {
        "ticks_survived": len(trace),
        "resources_collected": sum(row["food_collected"] for row in trace),
        "unique_positions": len(set(positions)),
        "successful_moves": moves, "boundary_moves": boundary_moves,
        "boundary_movement_fraction": boundary_moves / moves if moves else None,
        "attempted_blocked_moves": sum(row["action"] != "stay" and not row["moved"] for row in trace),
        "aba_reversals": len(reversals),
        "oscillation_runs": [run for run in runs if run["reversal_count"] >= 2],
        "food_visible_decisions": food_decisions,
        "food_visible_closer_fraction": sum(row["reduces_food_distance"] is True for row in trace) / food_decisions if food_decisions else None,
    }


def trace_episode(episode: Episode, network: TinyNeuralNetwork) -> tuple[dict, list[dict]]:
    """Use the shared tick iterator; decision fields precede their numbered step."""
    controller = ObservedNeuralController(network)
    initial = episode.world.agent_position
    before = initial
    resources = episode.resources_collected
    trace = []
    for _ in episode_ticks(episode, controller):
        after = episode.world.agent_position
        displacement = (after.x - before.x, after.y - before.y)
        trace.append({
            "tick": episode.completed_ticks, "position": [before.x, before.y],
            **controller.decision,
            "position_after": [after.x, after.y], "energy_after": episode.energy,
            "food_collected": episode.resources_collected > resources,
            "moved": before != after,
            "reduces_food_distance": reduces_food_distance(controller.decision["visible_food"], displacement),
        })
        before, resources = after, episode.resources_collected
    return summarize_trace(trace, [initial.x, initial.y]), trace


def analyze_champion(model: Path, scenario: int, no_food: bool = False) -> dict:
    """One deterministic episode and controlled probes, without training or GUI."""
    model = model.resolve()
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    network = load_champion(model)
    summary, trace = trace_episode(create_analysis_episode(scenario, no_food), network)
    probes = controlled_probes(network)
    if hashlib.sha256(model.read_bytes()).hexdigest() != model_hash:
        raise ValueError(f"Model changed during analysis: {model}")
    return {
        "format_version": 1, "model_path": str(model), "model_sha256": model_hash,
        "scenario_seed": scenario, "food_disabled": no_food,
        "summary": summary, "trace": trace, "controlled_probes": probes,
        "experiment": {
            "action_order": [action.value for action in ACTION_ORDER],
            "coordinates": "x east, y south; relative food [dx,dy]; boundary cells 0 and 24 are valid",
            "trace_timing": "tick starts at 1; position, energy, visible_food and scores are pre-step; *_after and outcome flags are post-step",
            "probes": "Nine positions (open, four edge midpoints, four corners), every feasible non-center radius-2 food offset and no-food control, energies 100,75,50,25; changes compare adjacent energies in that order",
            "no_food": "Fresh Episode(World(seed scenario start)) with no food; original scenario unmodified",
            "metric_definitions": METRIC_DEFINITIONS, "limitations": LIMITATIONS,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
        },
    }
