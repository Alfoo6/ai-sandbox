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

Run tests with `uv run python -m unittest discover -s tests -v`.

## Watch the evolved champion (v0.3.7)

```bash
uv run python scripts/watch_evolved.py --scenario 0
uv run python scripts/watch_evolved.py --scenario 3
uv run python scripts/watch_evolved.py --scenario 100
uv run python scripts/watch_evolved.py --scenario 150
```

The scenario defaults to `0`. Each launch reconstructs the same champion once:
population 40, 8 elites, training seeds `(0, 1, 2, 3, 4)`, mutation scale `0.05`,
30 generations, training seed `42`. Training takes place before the window opens
and verifies final best fitness `125.0`; the selected viewing seed never affects
training. No saved champion file is required.

Compare training seed `0` or `3` with unseen held-out seed `100` or `150` to
observe how the same policy behaves on different food layouts. The existing
viewer shows the scenario seed, ticks, energy, resources, and Running/Paused/Finished
status at five ticks per second. SPACE pauses/resumes; ESC or closing the window
exits. The final frame remains visible after termination. HUD information is only
for the observer; the controller still receives its original local observation.

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
count, alternative fitness designs, and champion persistence. None is established
as the solution yet. Later work may
compare evolutionary learning with reinforcement learning; reinforcement learning
is not implemented. Multiple agents and reproduction remain future work.

## Repeated evolution study (v0.3.8)

Run `uv run python scripts/run_evolution_study.py`. The visible `CONFIG` in that
script fixes population size **40**, elite count **8**, training scenario seeds
**0–4**, mutation scale **0.05**, **30** generations (0–29), evolution seeds
**42, 43, 44**, and validation scenario seeds **200–249**.

Training scenarios determine fitness and champion selection. Each evolution seed
starts a fresh population and RNG state; multiple runs reveal sensitivity to
initialization and mutation randomness. Every unchanged champion is evaluated on
the same validation scenarios, alongside random and rule-based reference baselines
whose controller RNG seed equals the scenario seed. The study retains champions,
training summaries, and per-episode validation metrics in memory.

The report shows per-run training fitness, validation survival and resources, and
episode counts. Aggregate means and **population standard deviations (`ddof=0`)**
are computed across per-run validation means, not pooled scenario episodes. This
convention describes the observed runs (one run has SD zero); three runs do not
establish statistical significance.

Validation supports exploratory comparisons and may guide later choices, so its
results are **not final test results**. Training and validation seeds must be
unique within each set and disjoint between sets. Scenario seeds **1000–1099**
are reserved for future final testing after experimental choices are fixed; this
framework rejects them in both scenario sets. The older held-out 100–199 results
above remain a historical single-champion evaluation. No mechanics, network,
fitness, mutation, evolution algorithm, or viewer behavior changes in this block.

## Diverse-scenario evolution (v0.3.9)

This is a separate experimental training path; the original `evolve()` and all
fixed-scenario experiments remain unchanged. Each generation samples a fresh batch
using a local NumPy RNG derived from `(scenario_schedule_seed, generation_index)`.
Generation `g` samples without replacement from
`[10000 + g*1000000, 10000 + (g+1)*1000000)`. Distinct generation blocks prevent
scenario-seed reuse; the batch does not depend on evolution RNG consumption or the
requested run length. Use the same schedule seed and batch size to compare runs
on identical training scenarios. Different seeds can still produce similar worlds.

All individuals see the same batch, including elites preserved unchanged by the
existing selection function. Every individual is reevaluated on the next batch.
Breeding uses only that generation's scores; fitness remains mean survival ticks.
**Training scores from different batches are not directly comparable**, and best
training fitness need not increase. No all-time training champion is selected.

Validation uses fixed seeds **200–249** by default, in a separate namespace
`[200, 1000)`. The Python configuration can select a subset of that namespace.
Training seeds start at 10000; final-test seeds **1000–1099** are never evaluated.
At every validation interval of completed generations, and at the final generation,
only the current batch winner is validated. Training ties use population order.
The best validation candidate is replaced only by a strictly higher mean survival
on the fixed validation set; ties retain the earliest candidate. Validation never
changes parent selection or mutation. `--validation-interval 0` disables validation
and produces no validation champion model.

### Manual experiments

These commands are for manual execution. The 200-generation seed-42 run has now
completed; see v0.3.10 below for loading its selected champion. Defaults are a short pilot: 5 generations,
12 individuals, 3 elites, 5 scenarios per generation, mutation scale 0.05,
evolution seed 42, schedule seed 123, validation interval 2.

```bash
uv run python scripts/run_diverse_evolution.py --output-dir results/diverse-pilot
```

A longer exploratory experiment (200 generations, 20 scenarios per generation):

```bash
uv run python scripts/run_diverse_evolution.py \
  --generations 200 --population 40 --elites 8 \
  --scenarios-per-generation 20 --mutation-scale 0.05 \
  --seed 42 --scenario-schedule-seed 123 --validation-interval 10 \
  --output-dir results/diverse-200-seed42
```

For a persistent terminal, if tmux is installed:

```bash
tmux new -s ai-sandbox
cd /home/alfons/projects/ai-sandbox
# Run either command above inside this session.
# Detach with Ctrl-b, then d. Reattach later:
tmux attach -t ai-sandbox
```

Detaching leaves the job running; it is not a checkpoint or a resume mechanism.
Progress prints and flushes after each generation: completed/total generations,
current-batch best and mean fitness, validation metrics when available, and elapsed
time. There is no output during an individual generation's evaluation.

### Saved artifacts and interpretation

Without `--output-dir`, a unique timestamped directory is created under `results/`.
That directory tree is ignored by Git. An explicitly supplied directory must not
already exist; use a new name for each experiment. Custom paths outside `results/`
may need their own ignore rule.

- `config.json`: configuration, RNG seeds, validation seeds, schedule definition,
  architecture/encoding/action metadata, and Python/NumPy versions.
- `generations.jsonl`: one flushed JSON record per completed generation, with
  zero-based generation index, actual training scenario seeds, individual fitnesses,
  best/mean training fitness, elapsed seconds, and periodic validation summaries
  plus all validation episode metrics. `selected_for_validation_champion` records
  whether this generation improved the retained candidate.
- `champion-generation-NNNNN.npz`: parameters and explicit metadata for each newly
  retained validation champion. Files use NumPy arrays without pickle; load one
  using `ai_sandbox.diverse_artifacts.load_network(Path(...))` to obtain a network
  suitable for `NeuralController`. The existing viewer is unchanged.
- `result.json`: written only on successful completion, with final-batch training
  summary, best validation metrics/generation, and the selected model filename.
  Model and JSON replacements are atomic per file. On interruption, completed
  history and already saved models remain, but there is no completed result file;
  an abrupt termination can leave a partial last JSONL line or temporary file.

These are inference models, **not resumable training checkpoints**: population and
RNG state are not saved. The process also retains generation summaries in memory.
Periodic validation examines only batch winners, not every population member, and
repeated candidate selection can overfit the validation set. A single run does not
establish generalization or statistical significance; repeat independent evolution
seeds on the same schedule and validation set before drawing conclusions. Final
testing remains separate from training; v0.3.10 adds manual saved-model testing. Changing batches may reduce overfitting
but can also introduce noisy selection; improvement has not been demonstrated.

Cost scales roughly as `generations × population × scenarios_per_generation`,
plus one validation episode per validation seed at each validation event. The pilot
requires **300 training + 150 validation episodes**; the longer command requires
**160000 training + 1000 validation episodes**. At the 500-tick episode cap, those
are at most 225000 and 80500000 simulation ticks respectively. Runtime depends on
hardware and policy survival; use the pilot's elapsed times for a local estimate.
No timing benchmark was run for v0.3.9.


## Saved champion evaluation and visualization (v0.3.10)

Both new tools load a v0.3.9 `.npz` directly, without retraining. Identify the
selected champion from `result.json`'s `champion_file`, not the largest filename
or the final training batch. For example, from the project root:

```bash
uv run python -c 'import json; from pathlib import Path; p = Path("results/diverse-200-seed42"); r = json.loads((p / "result.json").read_text()); print(p / r["champion_file"])'
```

The completed 200-generation experiment reports `best_validation_generation: 169`
(zero-based index, completed generation **170**) and selects
`results/diverse-200-seed42/champion-generation-00169.npz`.
Its evolution seed is 42, schedule seed 123, and validation seeds are 200–249.

Manual final evaluation (100 scenarios, **300 episodes** across three controllers):

```bash
uv run python scripts/evaluate_saved_champion.py \
  --model results/diverse-200-seed42/champion-generation-00169.npz \
  --seed-start 1000 --count 100 \
  --output results/diverse-200-seed42/final-test-1000-1099.json
```

Manual visualization, initially using validation scenario 200:

```bash
uv run python scripts/watch_saved_champion.py \
  --model results/diverse-200-seed42/champion-generation-00169.npz \
  --scenario 200
```

The existing viewer displays the selected scenario seed in its HUD; SPACE
pauses/resumes, ESC exits, and the final frame stays visible. Viewing defaults to
scenario 0. Evaluation defaults to seed-start 1000 and count 100; model and output
are required. The output parent directory must exist, and existing files are
never overwritten (choose another filename for a repeat).

Evaluation uses fresh equivalent worlds for each controller on every requested
seed. Random and rule-based controller RNG seeds equal the scenario seed, matching
the existing baseline convention; the neural policy is deterministic. Reports
include mean/median survival, mean resources, counts collecting any resource and
surviving strictly past 100 ticks, full per-episode metrics, and paired comparisons.
Each comparison matches scenario seeds: `first_wins` counts neural wins,
`second_wins` counts baseline wins, and `ties` counts equal results. Survival and
resource comparisons are separate; these counts describe outcomes, not effect
sizes, confidence intervals, or significance tests.

JSON records the absolute model path, SHA-256 of its file bytes, architecture,
scenario seeds, seeding conventions, and Python/NumPy versions. There are no
wall-clock timestamps or durations. Identical model bytes and seeds under the same
code/runtime give deterministic results and serialization. Moving the model
changes its nonessential absolute path; runtime-version metadata can also differ
between environments. Cross-version numerical reproducibility is not guaranteed.
The model loader validates architecture, parameter shapes, and finite values and
owns independent parameter copies. Evaluation never writes to the model.

Validation seeds 200–249 influenced champion selection, so their scores are not
an unbiased final test. Final-test seeds **1000–1099 are now authorized for manual
saved-champion evaluation**, after fixing the model and experimental choices;
they remain excluded from diverse training and validation. No final evaluation
has been run as part of this implementation. Using test results or test viewing
to tune another champion would turn those seeds into development data and require
a new untouched test set. The evaluator accepts other ranges for exploratory
use; it does not label arbitrary seeds as unseen or enforce historical disjointness.

One champion from one evolution seed cannot establish robustness across training
runs. Validation selection can overfit, scenario seeds can generate similar worlds,
and paired wins alone do not establish generalization or statistical significance.
This adds inference tools only; simulation, controllers, rendering, and the v0.1
mechanics are unchanged. Saved models remain inference artifacts, not resumable
training checkpoints. Older `watch_evolved.py` and `run_evolved_evaluation.py`
still retrain their historical champion; use the new saved-model scripts here.

## Saved champion behavioral analysis (v0.3.11)

This inference-only tool examines boundary following and decisions near visible
food before any fitness change or further training. It loads the saved champion
without modifying weights, controllers, or world rules. Each invocation runs one
headless episode (at most 500 ticks) plus 484 controlled observation probes.
It does not import Pygame. From the project root:

```bash
uv run python scripts/analyze_champion.py \
  --model results/diverse-200-seed42/champion-generation-00169.npz \
  --scenario 1027 --output results/diverse-200-seed42/behavior-1027.json

uv run python scripts/analyze_champion.py \
  --model results/diverse-200-seed42/champion-generation-00169.npz \
  --scenario 1041 --output results/diverse-200-seed42/behavior-1041.json

uv run python scripts/analyze_champion.py \
  --model results/diverse-200-seed42/champion-generation-00169.npz \
  --scenario 1027 --no-food \
  --output results/diverse-200-seed42/behavior-1027-no-food.json
```

The output parent must exist. Existing reports (including symlinks) are rejected,
with exclusive creation also protecting against races. JSON includes the absolute
model path and SHA-256, scenario seed, food-disabled flag, summary, full trace,
controlled probes, definitions, limitations, and runtime versions. No timestamps
are included; identical inputs under the same code/runtime serialize identically.
Moving the model changes the recorded path. Cross-version numerical equality is
not guaranteed. No-food builds a fresh empty `World` at the seeded scenario's
starting position; it never removes resources from the original scenario or
changes global mechanics. All current scenarios start at (12,12), so no-food
trajectories are identical across seeds for a fixed policy.

### Probes and action scores

Probes use `World` and `Episode.observe()` to generate valid 5×5 geometry, then
replace only the observation's energy with 100, 75, 50, or 25. Open-space probes
cover all 24 non-center food offsets, with exactly one food cell. Boundary probes
use the four edge midpoints and four corners, covering every feasible single-food
offset at each position. Every position also has a no-food control. Food outside
the world is omitted, never relocated. Compare rows with the **same food offset
and energy** to study boundary sensitivity; there is no non-center food offset
valid at every corner. The positions are on the edges (0 or 24), not one cell
inward. Out-of-bounds cells are beyond those edges; boundary cells are traversable.

Each probe records all five raw scores, action, boundary collision, one-step
food-distance reduction, and whether the action differs from the previous energy
in the order 100 → 75 → 50 → 25 (null for the first energy). Score order is
**north, south, east, west, stay**; the first maximum wins ties. Scores are the
untransformed outputs of `tanh(input @ W1 + b1) @ W2 + b2`. They can be negative,
do not sum to one, and are **not probabilities or calibrated confidence**.
The analysis delegates action selection to the original `NeuralController` and
recomputes scores through its same pure forward function for diagnostics.

### Trace and metric definitions

Tick numbers start at 1. Position, energy, visible food offsets `[dx, dy]`, action,
and scores describe the state **before** that step. `position_after`,
`energy_after`, movement, and collection flags describe its outcome. Positive x
is east; positive y is south. Food offsets are ordered north-to-south, west-to-east.

- `ticks_survived`: completed steps, including the fatal tick.
- `resources_collected`: number of steps collecting food.
- `unique_positions`: distinct initial and post-step positions, including the
  terminal position.
- `successful_moves` and `boundary_moves`: moves changing position, and the subset
  ending with x or y equal to 0 or 24. `boundary_movement_fraction` divides the
  latter by the former. This counts movement destinations, not time occupancy or
  specifically movement parallel to the wall. Stays and blocked moves are excluded.
- `attempted_blocked_moves`: non-stay actions that leave position unchanged.
- `aba_reversals`: consecutive position triples A-B-A with A different from B.
  Stationary steps interrupt reversals; they are not removed from the trajectory.
- `oscillation_runs`: maximal groups of at least two consecutive reversal events,
  corresponding to A-B-A-B or longer alternating sequences. Each entry gives the
  first and last reversal decision ticks and number of overlapping reversals.
  An isolated A-B-A counts as a reversal but not an oscillation run.
- `food_visible_decisions`: pre-step observations containing any food.
- `food_visible_closer_fraction`: fraction of food-visible decisions for which
  actual displacement strictly reduces the minimum Manhattan distance to the
  **same pre-step visible food set**. Manhattan distance is `abs(dx) + abs(dy)`.
  Recompute the minimum over that set at the destination, including collected
  food; do not substitute the next observation's food. Stay and blocked moves
  do not reduce distance. Undefined fractions (zero denominator) are JSON null.

A reversal may be useful, including after collecting food. Neither reversals nor
oscillations automatically indicate errors. Moving closer for one step does not
guarantee eventual food collection. Single-food probes omit interactions between
multiple food cells. No-food removes both food cues and energy replenishment,
so it cannot separate those causal effects. Observed boundary movement is evidence
about actions, not proof of an inferred boundary-following strategy or intent.
One episode cannot establish generalization; using final-test scenarios 1027 and
1041 to guide future training makes them development data and requires a new
untouched final test set.

### Optional viewer diagnostics and verification

For manual viewing only, add `--debug` to the existing saved-model viewer:

```bash
uv run python scripts/watch_saved_champion.py \
  --model results/diverse-200-seed42/champion-generation-00169.npz \
  --scenario 1027 --debug
```

Debug mode prints each pre-step action, energy, visible-food count, and named raw
scores to the terminal. This avoids expanding the fixed-size HUD or altering the
renderer. The shared viewer and pacing remain unchanged; the HUD identifies the
diagnostic wrapper as `ObservedNeuralController`. Without `--debug`, the original
controller and visualization remain unchanged. Printing can affect wall-clock
pacing but never the discrete simulation results.

```bash
uv run python -m unittest discover -s tests -p 'test_behavioral_analysis.py' -v
uv run python -m unittest discover -s tests -v
uv run python scripts/analyze_champion.py --help
uv run python scripts/watch_saved_champion.py --help
git diff --check
```

Behavioral tests use synthetic networks and trajectories, forbid evolution calls
in report/viewer tests, verify unchanged weights/model bytes, and mock the viewer
entrypoint so they never open a real window.
