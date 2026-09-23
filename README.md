# AI Sandbox

A learning and experimentation project for autonomous agent behavior.

Current status: v0.1 simulation and visual demo, v0.2 reproducible baselines and
experiments, and v0.3 neuroevolution with a held-out evaluation are implemented.

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
- A memoryless `RandomController` baseline and a headless episode runner.
- A Pygame viewer that reads the same simulation loop as headless execution.
- Metrics: ticks survived, resources collected, distance travelled, final energy.
  Ticks survived includes the fatal tick; distance travelled counts only
  successful movement.

## v0.2 baselines and experiments

- A memoryless `RuleBasedController` moves toward visible food and explores with
  its own seeded RNG when no food is visible.
- The structural `Controller` protocol accepts interchangeable controllers through
  `choose_action(observation) -> Action`.
- Seeded random scenarios create fresh worlds with six food cells. The multi-scenario
  headless experiment runner evaluates controllers on the same seeds.
- Experiments report per-episode metrics, aggregate mean/median/min/max, and paired
  per-scenario resource or survival wins and ties.

## v0.3 neuroevolution

- Numeric encoding turns the local 5×5 cells, in row-major order, into 25 values:
  empty `0.0`, food `1.0`, and out of bounds `-1.0`. Normalized energy
  (`energy / 100`) is the final value, for 26 inputs total.
- `TinyNeuralNetwork` has one tanh hidden layer. The first policy uses 26 inputs,
  12 hidden neurons, and 5 raw action scores: 389 weights and biases in total.
  `NeuralController` maps the scores to north, south, east, west, and stay, choosing
  the first action on a score tie.
- Fitness is the arithmetic mean of ticks survived on the training scenarios.
  Food has no direct bonus: collecting it restores energy and may extend survival.
- Mutation adds independent Gaussian noise to every weight and bias. The
  generational loop ranks by fitness, preserves elites unchanged, and fills the
  remaining slots with mutated children of elites in deterministic round-robin
  order. Generation 0 evaluates the initial random population.
- A separate held-out evaluation compares one evolved champion with the random
  and rule-based baselines on unseen scenario seeds.

## Architecture

World → Observation → Controller → Action → World. Controllers depend on the
observation/action contract, never World or Episode internals or global coordinates.
The neural controller receives only local 5×5 cells and normalized energy. It does
not receive absolute food positions, food direction, distances, scenario seeds, or
controller history. The world defines mechanics; controllers choose behavior. The
project explores behavior learned from local experience rather than hard-coding
intelligent decisions into the environment.

One shared simulation loop supports visual and headless execution. Rendering only
reads simulation state and must never influence it, including simulation timing
or randomness. Scenario generation uses a local seeded RNG; random and rule-based
controllers use their own seeded RNGs; network initialization and mutation use
local NumPy generators derived from the training seed. These operations do not
depend on Python or NumPy module-level global RNG state. Prefer simple
implementations and minimal dependencies.

## Visual demo

Run `uv run ai-sandbox` (or `python -m ai_sandbox.visual_demo` in an environment
with the project installed). Pygame provides the window, grid drawing, and keyboard
events; no assets or additional graphical dependencies are needed.

The demo starts at (12, 12) with food at (11, 12), (13, 12), (12, 11), (12, 13),
(10, 10), and (14, 14), using `RuleBasedController(seed=42)`. Green circles are food,
the gold square is the agent, and shaded cells show its radius-2 vision.

The display advances at five simulation ticks per second. SPACE pauses or resumes;
ESC or closing the window exits. The final state stays visible after termination.
Visual and headless execution use the same simulation iterator; drawing and pacing
do not consume controller randomness or alter discrete tick mechanics.

Run tests with `uv run python -m unittest discover -s tests -v`. At this
checkpoint, the full suite contains 130 passing tests.

## Reproducing experiments

- `uv run python scripts/run_baseline_experiment.py` compares the random and
  rule-based baselines on scenario seeds 0–99.
- `uv run python scripts/run_neuroevolution.py` runs the first 30-generation
  training demonstration.
- `uv run python scripts/run_evolved_evaluation.py` reproduces that training once,
  then compares its champion with both baselines on held-out seeds 100–199.

The first training run uses population size 40, 8 elites, scenario seeds
`(0, 1, 2, 3, 4)`, mutation scale `0.05`, 30 generations, and training seed `42`.
Fitness is mean ticks survived. Generation 0's best fitness was `105.0`; the
champion reached `125.0` by generation 10 and remained at `125.0` through
generation 29. Its ticks on the five training scenarios were
`(150, 100, 125, 150, 100)`. Improvement on these reused seeds alone does not
establish general learning.

## Held-out v0.3 result

Training used seeds **0–4**. The held-out test used **100–199**, which did not
influence training or champion selection. Each controller ran on fresh equivalent
worlds for the same 100 held-out seeds.

| Controller | Mean ticks | Median ticks | Mean resources | Collected food | Survived past 100 ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RandomController` | 108.70 | 100 | 0.45 | 35/100 | 34/100 |
| `RuleBasedController` | 124.86 | 123.5 | 1.51 | 75/100 | 73/100 |
| Evolved `NeuralController` | 104.49 | 100 | 0.22 | 17/100 | 13/100 |

The champion's mean survival fell from `125.0` on training seeds to `104.49` on
held-out seeds, a gap of `20.51` ticks. It sometimes found food on unseen
scenarios, but most held-out episodes ended at the 100-tick no-food baseline.
Its held-out survival and resource collection were below both current baselines.
This is descriptive evidence of a train/test generalization gap for this champion,
not a formal statistical conclusion.

## Roadmap

The main open question is how to improve generalization beyond a very small fixed
training set. Possible investigations include more diverse training scenarios,
changing scenario sets during evolution, improving experimental methodology with
train/validation/test separation, population size, mutation scale, generation
count, alternative fitness designs, visualizing the evolved champion's behavior,
and champion persistence. None is established as the solution yet. Later work may
compare evolutionary learning with reinforcement learning; reinforcement learning
is not implemented. Multiple agents and reproduction remain future work.
