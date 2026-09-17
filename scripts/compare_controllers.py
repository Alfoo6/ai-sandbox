"""Run the fixed v0.1 scenario once with each baseline controller."""

from ai_sandbox.controllers import RandomController, RuleBasedController
from ai_sandbox.runner import run_episode
from ai_sandbox.scenarios import create_demo_episode


def main() -> None:
    print(
        f"{'controller':<20} {'ticks':>5} {'resources':>9} "
        f"{'distance':>8} {'energy':>6}"
    )
    for controller in (RandomController(seed=42), RuleBasedController(seed=42)):
        result = run_episode(create_demo_episode(), controller)
        print(
            f"{type(controller).__name__:<20} "
            f"{result.ticks_survived:>5} "
            f"{result.resources_collected:>9} "
            f"{result.distance_travelled:>8} "
            f"{result.final_energy:>6}"
        )


if __name__ == "__main__":
    main()
