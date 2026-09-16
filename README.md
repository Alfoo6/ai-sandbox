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

## Future work (outside v0.1)

Neural networks, reinforcement learning, genetic algorithms, multiple agents,
reproduction, and other advanced systems are deferred.
