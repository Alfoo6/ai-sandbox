"""Minimal headless episode execution."""

from dataclasses import dataclass
from collections.abc import Iterator

from ai_sandbox.controllers import RandomController
from ai_sandbox.episode import Episode


@dataclass(frozen=True)
class EpisodeResult:
    ticks_survived: int
    resources_collected: int
    distance_travelled: int
    final_energy: int


def episode_ticks(episode: Episode, controller: RandomController) -> Iterator[None]:
    """Advance one complete simulation tick per iteration, without pacing or I/O."""
    while not episode.terminated:
        observation = episode.observe()
        action = controller.choose_action(observation)
        episode.step(action)
        yield


def run_episode(episode: Episode, controller: RandomController) -> EpisodeResult:
    """Advance the supplied episode to termination and snapshot its metrics."""
    for _ in episode_ticks(episode, controller):
        pass

    return EpisodeResult(
        ticks_survived=episode.completed_ticks,
        resources_collected=episode.resources_collected,
        distance_travelled=episode.distance_travelled,
        final_energy=episode.energy,
    )
