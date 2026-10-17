"""Simple experiment artifacts and inference-only network persistence."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from ai_sandbox.diverse_evolution import (
    DiverseConfig, DiverseGeneration, DiverseResult, GENERATION_SEED_SPAN,
    TRAINING_SEED_START, ValidationChampion, evolve_diverse,
)
from ai_sandbox.neural_network import TinyNeuralNetwork


MODEL_METADATA = {
    "format_version": 1,
    "input_size": 26,
    "hidden_size": 12,
    "output_size": 5,
    "activation": "tanh",
    "action_order": ["north", "south", "east", "west", "stay"],
    "encoding": "25 row-major local cells (-1,0,1), then energy/100",
}


def save_network(path: Path, network: TinyNeuralNetwork) -> None:
    """Atomically replace an inference model, not a resumable checkpoint."""
    if (network.input_size, network.hidden_size, network.output_size) != (26, 12, 5):
        raise ValueError("expected architecture 26 -> 12 -> 5")
    parameters = network.parameters()
    if not all(np.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("network parameters must be finite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, metadata=json.dumps(MODEL_METADATA),
                 **dict(zip(("w1", "b1", "w2", "b2"), parameters)))
    temporary.replace(path)


def load_network(path: Path) -> TinyNeuralNetwork:
    """Load validated architecture metadata and arrays without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata != MODEL_METADATA:
            raise ValueError("unsupported model metadata")
        parameters = tuple(archive[name] for name in ("w1", "b1", "w2", "b2"))
        if not all(np.isfinite(parameter).all() for parameter in parameters):
            raise ValueError("network parameters must be finite")
        return TinyNeuralNetwork(26, 12, 5, *parameters)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_diverse_experiment(config: DiverseConfig, output_dir: Path) -> DiverseResult:
    """Stream metrics and improved champions into a new directory; never overwrite a run."""
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config.json", {
        "format_version": 1,
        "configuration": asdict(config),
        "architecture": MODEL_METADATA,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scenario_schedule": {
            "algorithm": "SeedSequence([schedule_seed, generation]); choice without replacement",
            "training_seed_start": TRAINING_SEED_START,
            "generation_seed_span": GENERATION_SEED_SPAN,
            "validation_namespace": [200, 999],
            "reserved_final_test_namespace": [1000, 1099],
        },
        "candidate_policy": "current-batch winner; highest validation mean ticks; earliest on ties",
    })
    started = perf_counter()
    print(f"Writing artifacts to {output_dir}", flush=True)
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as history:
        def record_generation(record: DiverseGeneration, selected: ValidationChampion | None) -> None:
            elapsed = perf_counter() - started
            if selected is not None:
                # Generation-specific files let history refer to an exact retained model.
                save_network(output_dir / f"champion-generation-{selected.generation:05d}.npz", selected.network)
            entry = asdict(record) | {"elapsed_seconds": elapsed}
            history.write(json.dumps(entry, allow_nan=False) + "\n")
            history.flush()
            message = (
                f"Generation {record.training.generation + 1}/{config.generations} "
                f"(index {record.training.generation}) | current batch best={record.training.best_fitness:.2f} "
                f"mean={record.training.mean_fitness:.2f}"
            )
            if record.validation is not None:
                summary = record.validation.summary
                message += (
                    f" | validation mean ticks={summary.ticks_survived.mean:.2f}"
                    f" median={summary.ticks_survived.median:g}"
                    f" resources={summary.resources_collected.mean:.2f}"
                    f" food={record.validation.episodes_collecting_resource}/{summary.episodes}"
                    f" >100={record.validation.episodes_past_100_ticks}/{summary.episodes}"
                    f" selected={selected is not None}"
                )
            print(f"{message} | elapsed={elapsed:.2f}s", flush=True)

        result = evolve_diverse(config, on_generation=record_generation)
    champion = result.best_validation_champion
    _write_json(output_dir / "result.json", {
        "status": "complete",
        "elapsed_seconds": perf_counter() - started,
        "completed_generations": len(result.generations),
        "final_batch_training": asdict(result.generations[-1].training),
        "best_validation_generation": champion.generation if champion else None,
        "best_validation": asdict(champion.validation) if champion else None,
        "champion_file": f"champion-generation-{champion.generation:05d}.npz" if champion else None,
        "resumable_checkpoint": False,
    })
    print(f"Completed. Summary: {output_dir / 'result.json'}", flush=True)
    return result
