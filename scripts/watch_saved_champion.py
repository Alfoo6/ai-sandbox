"""Watch a saved neural champion on one scenario without retraining."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai_sandbox.behavioral_analysis import ObservedNeuralController
from ai_sandbox.controllers import NeuralController
from ai_sandbox.saved_champion import load_champion
from ai_sandbox.scenarios import create_random_episode
from ai_sandbox.visual_demo import run_visual_episode


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="print pre-step neural decisions to the terminal")
    args = parser.parse_args(argv)
    try:
        network = load_champion(args.model)
    except ValueError as error:
        parser.error(str(error))
    controller = ObservedNeuralController(network, debug=True) if args.debug else NeuralController(network)
    run_visual_episode(
        create_random_episode(args.scenario), controller,
        scenario_seed=args.scenario,
    )


if __name__ == "__main__":
    main()
