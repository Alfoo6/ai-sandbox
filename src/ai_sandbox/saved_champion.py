"""Inference-only evaluation of saved v0.3.9 models."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from zipfile import BadZipFile

import numpy as np

from ai_sandbox.controllers import NeuralController, RandomController, RuleBasedController
from ai_sandbox.diverse_artifacts import MODEL_METADATA, load_network
from ai_sandbox.experiments import compare_paired_metric, run_experiment, summarize_results
from ai_sandbox.neural_network import TinyNeuralNetwork


def load_champion(path: Path) -> TinyNeuralNetwork:
    """Use the existing shape/metadata validator with a path-specific error."""
    try:
        return load_network(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile) as error:
        raise ValueError(f"Cannot load model {path}: {error}") from error


def evaluate_saved_champion(model: Path, seed_start: int, count: int) -> dict:
    """Evaluate fresh controllers/worlds on identical seeds, without training."""
    if type(seed_start) is not int or type(count) is not int or count <= 0:
        raise ValueError("seed-start must be an integer and count must be a positive integer")
    model = model.resolve()
    # Hash before loading and verify afterward so a replaced file cannot silently
    # produce a report claiming the identity of a different model.
    try:
        model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"Cannot load model {model}: {error}") from error
    network = load_champion(model)
    seeds = tuple(range(seed_start, seed_start + count))
    results = {
        "neural": run_experiment(seeds, lambda _: NeuralController(network)),
        "random": run_experiment(seeds, RandomController),
        "rule_based": run_experiment(seeds, RuleBasedController),
    }
    controllers = {}
    for name, episodes in results.items():
        controllers[name] = {
            "summary": asdict(summarize_results(episodes)),
            "episodes_collecting_resource": sum(e.resources_collected > 0 for e in episodes),
            "episodes_past_100_ticks": sum(e.ticks_survived > 100 for e in episodes),
            "episodes": [asdict(e) for e in episodes],
        }
    comparisons = {}
    for baseline in ("random", "rule_based"):
        for metric in ("ticks_survived", "resources_collected"):
            comparisons[f"neural_vs_{baseline}_{metric}"] = asdict(compare_paired_metric(
                results["neural"], results[baseline], lambda e: getattr(e, metric),
            ))
    if hashlib.sha256(model.read_bytes()).hexdigest() != model_hash:
        raise ValueError(f"Model changed during evaluation: {model}")
    return {
        "format_version": 1,
        "model_path": str(model),
        "model_sha256": model_hash,
        "scenario_seeds": list(seeds),
        "controllers": controllers,
        "paired_comparisons": comparisons,
        "experiment": {
            "architecture": MODEL_METADATA,
            "scenario_factory": "ai_sandbox.scenarios.create_random_episode",
            "baseline_controller_seed": "equals scenario seed; independent controller RNG",
            "neural_policy": "deterministic argmax; first action on ties",
            "paired_order": "first=neural, second=named baseline",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }


def write_report(output: Path, report: dict) -> None:
    """Exclusive creation protects existing reports, models and symlink targets."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def print_report(report: dict) -> None:
    for name, data in report["controllers"].items():
        summary = data["summary"]
        print(
            f"{name}: mean ticks={summary['ticks_survived']['mean']:.2f}, "
            f"median ticks={summary['ticks_survived']['median']:g}, "
            f"mean resources={summary['resources_collected']['mean']:.2f}, "
            f"collected food={data['episodes_collecting_resource']}/{summary['episodes']}, "
            f">100 ticks={data['episodes_past_100_ticks']}/{summary['episodes']}"
        )
    for name, comparison in report["paired_comparisons"].items():
        print(f"{name}: neural wins={comparison['first_wins']}, "
              f"baseline wins={comparison['second_wins']}, ties={comparison['ties']}")
