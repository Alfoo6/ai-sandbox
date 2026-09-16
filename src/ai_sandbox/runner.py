"""Minimal headless episode execution."""

from dataclasses import dataclass

from ai_sandbox.controllers import RandomController
from ai_sandbox.episode import Episode


@dataclass(frozen=True)
class EpisodeResult:
    ticks_survived: int
    resources_collected: int
    distance_travelled: int
    final_energy: int


def run_episode(episode: Episode, controller: RandomController) -> EpisodeResult:
    """Advance the supplied episode to termination and snapshot its metrics."""
    while not episode.terminated:
        observation = episode.observe()
        action = controller.choose_action(observation)
        episode.step(action)

    return EpisodeResult(
        ticks_survived=episode.completed_ticks,
        resources_collected=episode.resources_collected,
        distance_travelled=episode.distance_travelled,
        final_energy=episode.energy,
    )
