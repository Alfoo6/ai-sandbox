# AI Sandbox

A learning and experimentation project for autonomous agent behavior.

## v0.1 scope

- One agent in a 25×25 grid with blocked edges and no wrapping, advancing in
  fixed discrete simulation steps.
- Local square vision radius 2. Observations include local cells and current
  energy, but no global coordinates.
- Actions: north, south, east, west, stay. Invalid movement leaves the agent in
  place and still consumes the tick.
- Initial and maximum energy: 100. Each tick costs 1 energy. Collecting food
  restores 25 energy, capped at the maximum; food does not respawn in v0.1.
- Tick order: observe → choose action → apply movement → collect food → subtract
  energy → check termination.
- Episodes end at zero energy or 500 completed ticks. The fatal tick counts as a
  completed tick.
- Interchangeable random and rule-based controllers only; both are memoryless.
- Reset clears all episode state.
- Explicit, separate RNGs for the world and controller; evaluate controllers across
  the same seeds for reproducible comparisons.
- Metrics: ticks survived, resources collected, distance travelled, final energy.
  Ticks survived includes the fatal tick; distance travelled counts only
  successful movement.

## Architecture

Observations → controller → actions → world update. Controllers depend on the
observation/action contract, never world internals. The world defines mechanics;
controllers choose behavior.

One shared simulation loop supports visual and headless execution. Rendering only
reads simulation state and must never influence it, including simulation timing
or randomness. Prefer simple implementations and minimal dependencies.

## Visual demo

Run `uv run ai-sandbox` (or `python -m ai_sandbox.visual_demo` in an environment
with the project installed). Pygame provides the window, grid drawing, and keyboard
events; no assets or additional graphical dependencies are needed.

The demo starts at (12, 12) with food at (11, 12), (13, 12), (12, 11), (12, 13),
(10, 10), and (14, 14), using `RandomController(seed=42)`. Green circles are food,
the gold square is the agent, and shaded cells show its radius-2 vision.

The display advances at five simulation ticks per second. SPACE pauses or resumes;
ESC or closing the window exits. The final state stays visible after termination.
Visual and headless execution use the same simulation iterator; drawing and pacing
do not consume controller randomness or alter discrete tick mechanics.

Run tests with `uv run python -m unittest discover -s tests -v`.

## Future work (outside v0.1)

Neural networks, reinforcement learning, genetic algorithms, multiple agents,
reproduction, and other advanced systems are deferred.
