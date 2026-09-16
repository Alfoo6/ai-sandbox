"""State and event tests using SDL's dummy display, without pixel assertions."""

from itertools import count
import os
import random
import unittest
from unittest.mock import patch

import pygame

from ai_sandbox.controllers import RandomController
from ai_sandbox.rendering.pygame_renderer import PygameRenderer, WINDOW_SIZE
from ai_sandbox.runner import episode_ticks, run_episode
from ai_sandbox.visual_demo import create_demo_episode, run_visual_episode
from ai_sandbox.world import Position


class VisualDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(pygame.quit)
        pygame.display.init()
        pygame.font.init()

    def test_demo_world_has_the_requested_layout(self) -> None:
        episode = create_demo_episode()
        self.assertEqual(episode.world.agent_position, Position(12, 12))
        self.assertEqual(episode.world.food_positions, {
            Position(11, 12), Position(13, 12), Position(12, 11),
            Position(12, 13), Position(10, 10), Position(14, 14),
        })

    def test_rendering_does_not_change_state_or_reproducibility(self) -> None:
        episode = create_demo_episode()
        controller = RandomController(42)
        renderer = PygameRenderer(pygame.Surface(WINDOW_SIZE))
        global_rng_state = random.getstate()
        for _ in episode_ticks(episode, controller):
            before = (
                episode.observe(), episode.world.agent_position,
                episode.world.food_positions, episode.completed_ticks,
                episode.resources_collected, episode.distance_travelled,
                episode.terminated,
            )
            for paused in (False, True):
                renderer.draw(episode, "RandomController", paused)
            self.assertEqual(before, (
                episode.observe(), episode.world.agent_position,
                episode.world.food_positions, episode.completed_ticks,
                episode.resources_collected, episode.distance_travelled,
                episode.terminated,
            ))
        self.assertEqual(random.getstate(), global_rng_state)
        reference = create_demo_episode()
        reference_controller = RandomController(42)
        self.assertEqual(
            run_episode(episode, controller),
            run_episode(reference, reference_controller),
        )
        self.assertEqual(episode.world.agent_position, reference.world.agent_position)
        self.assertEqual(episode.world.food_positions, reference.world.food_positions)
        self.assertEqual(
            controller.choose_action(episode.observe()),
            reference_controller.choose_action(reference.observe()),
        )

    def test_iterator_does_not_advance_until_requested(self) -> None:
        episode = create_demo_episode()
        ticks = episode_ticks(episode, RandomController(42))
        self.assertEqual(episode.completed_ticks, 0)
        next(ticks)
        self.assertEqual(episode.completed_ticks, 1)

    def test_space_pauses_and_resumes(self) -> None:
        episode = create_demo_episode()
        space = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        frames = []
        times = count(0, 200)
        with (
            patch("pygame.event.get", side_effect=[[space], [], [space], [escape]]),
            patch("pygame.time.get_ticks", side_effect=lambda: next(times)),
            patch("pygame.time.Clock"),
            patch.object(PygameRenderer, "draw", side_effect=lambda ep, name, paused: frames.append(
                (ep.completed_ticks, paused)
            )),
        ):
            run_visual_episode(episode, RandomController(42))
        self.assertEqual(frames, [(0, True), (0, True), (1, False)])

    def test_closing_or_escape_exits_before_advancing(self) -> None:
        for event in (
            pygame.event.Event(pygame.QUIT),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
        ):
            with self.subTest(event=event):
                episode = create_demo_episode()
                with patch("pygame.event.get", return_value=[event]):
                    run_visual_episode(episode, RandomController(42))
                self.assertEqual(episode.completed_ticks, 0)
                self.assertFalse(pygame.display.get_init())

    def test_visual_loop_matches_headless_and_keeps_final_state_visible(self) -> None:
        episode = create_demo_episode()
        controller = RandomController(42)
        final_frames = []
        times = count(0, 200)

        def events() -> list[pygame.event.Event]:
            if len(final_frames) == 3:
                return [pygame.event.Event(pygame.QUIT)]
            return []

        def draw(ep, name, paused) -> None:
            if ep.terminated:
                final_frames.append(ep.completed_ticks)

        with (
            patch("pygame.event.get", side_effect=events),
            patch("pygame.time.get_ticks", side_effect=lambda: next(times)),
            patch("pygame.time.Clock"),
            patch.object(PygameRenderer, "draw", side_effect=draw),
        ):
            run_visual_episode(episode, controller)
        expected = run_episode(create_demo_episode(), RandomController(42))
        self.assertEqual(run_episode(episode, controller), expected)
        self.assertEqual(final_frames, [expected.ticks_survived] * 3)


if __name__ == "__main__":
    unittest.main()
