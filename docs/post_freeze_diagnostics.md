# Post-freeze diagnostics D1-D8

The original 120 control episodes and eight oracle forecast trajectories remain
the frozen benchmark. These diagnostics were added after inspecting those
results. They do not replace the original estimates or uncertainty intervals.

## Inputs and commands

Use the later release source, a Python environment with the PyTorch extra, and
the extracted versioned research-data archive. Keep output outside the source
checkout. `ARCHIVE` below means the extracted archive root; replace all uppercase
paths with actual paths for your computer.

The supplementary seed spread, raw-energy reconciliation and crossed paired
bootstrap can be rebuilt without training or GPU computation:

```text
python scripts/analyze_seed_spread.py --control-dir ARCHIVE/results/control --analysis-dir ARCHIVE/results/aggregate --output OUTSIDE_CHECKOUT/seed-analysis
```

This retains the original calculation and uses command-line paths. Its raw
source column uses filenames relative to the supplied control directory,
whereas the original execution used internal workspace-relative paths.

The original execution used an internal input layout. The staging helper maps
the public archive into that layout by copying hash-verified frozen members:

```text
python scripts/stage_research_inputs.py --archive-root ARCHIVE --output OUTSIDE_CHECKOUT/frozen-evidence
python scripts/diagnostics.py --id D1 --frozen-evidence OUTSIDE_CHECKOUT/frozen-evidence --checkpoints ARCHIVE/models --output OUTSIDE_CHECKOUT/diagnostics --workers 3 --device cuda
python scripts/summarize_diagnostics.py --root OUTSIDE_CHECKOUT/diagnostics --manifest ARCHIVE/data/training/manifest.json
```

D1 completes 21 episodes before D2 reads its selected, elite and candidate
plans. D1's `branch_result.json` provides prospective descriptive criteria.
Branch (b) requires investigation of the control/observer/delay coupling before
interpreting poor control as predictor error. Branch (c) reports mixed behavior.
Check this result before constructing an explanatory narrative.

Run the next command with `ID` equal to D3, D4, D5, D6, D7 and D8 in that order:

```text
python scripts/diagnostics_additional.py --id ID --frozen-evidence OUTSIDE_CHECKOUT/frozen-evidence --checkpoints ARCHIVE/models --data-manifest ARCHIVE/data/training/manifest.json --output OUTSIDE_CHECKOUT/diagnostics --device cuda
```

The D4 runner first performs a separate one-epoch cost pilot. It stops if its
conservative estimate exceeds four hours on the GPU host. It runs four
new training conditions at seed 101. D8 reuses the weight-0.01 timing record
from D4 for one representative training and oracle evaluation, avoiding
duplicate work. Original offline timings were not recorded.

To rebuild the diagnostic figures after the runs finish:

```text
python scripts/render_diagnostic_assets.py --diagnostic-dir OUTSIDE_CHECKOUT/diagnostics --output OUTSIDE_CHECKOUT/diagnostic-figures
```

The figures contain a source-hash/plotted-value JSON companion. D1's trace
example uses N03 and PINO seed 101; endpoint temperatures are shown against
plan selection time, with the endpoint 130 seconds later because the known
10-second held-input interval precedes the 12-step plan.

An additional unplanned, descriptive readback reconciles the selected CEM
objective against its tracking, variance, slew and energy terms:

```text
python scripts/analyze_plan_preferences.py --diagnostic-dir OUTSIDE_CHECKOUT/diagnostics --protocol configs/frozen/protocol_v2.yaml --output OUTSIDE_CHECKOUT/plan-objective-readback
```

This reads saved candidates and predictions without new optimization or plant
runs. Candidate energy/objective rank association is not a physical response
derivative or evidence of a causal failure mechanism.

| ID | Question and design | Interpretation boundary |
|---|---|---|
| D1 | Same selected plan: 3 scenarios × (ROM + 6 checkpoints), 90 cycles each | Clone starts after the held-input interval; later closed-loop reoptimization is excluded |
| D2 | Training versus selected, elite and all candidate power, slew and plan energy | Correlated occurrences; empirical range coverage is not a distributional guarantee |
| D3 | Eight 16×16 oracle trajectories with identical forcing/material draws to the 32×32 test | Frozen weights; checks resolution sensitivity without retraining |
| D4 | Residual weights 0, 0.001, 0.01, 0.1; new seed-101 fits and 3 control scenarios | Exploratory single-seed comparison; not a retuning of the frozen benchmark |
| D5 | CEM 64/256 candidates × 3/6 iterations for ROM and FNO101 in 3 scenarios | Tests optimization-budget sensitivity within these settings |
| D6 | Five feasible SLSQP starts across 12 scenarios | Original SLSQP already used a shifted warm start and two starts |
| D7 | 16, 32, 64, 128 grids; field L2 comparison on the 64-grid reference | Piecewise-constant prolongation or area averaging; 64 is not an exact solution |
| D8 | 36-trajectory generation and one representative training/prediction measurement | Post-hoc timing on the tested host; CUDA event interval includes idle gaps |

## Records and reproducibility

Each diagnostic writes separate arrays/CSVs, configuration hashes and source
identity. `post-freeze/executed_sources` in the research archive preserves the
executed scripts and core source. Receipts retain dirty-tree status rather
than presenting a later release commit as the source of every earlier run.
For D1 the original control trace is compared with the instrumented rerun;
instrumented decision timings are not benchmark timing results.

Avoid rerunning into an existing diagnostic directory when changing any input.
D1 skips complete episodes, and D4 reuses existing training records. Start a
new output directory for changed configurations. Different training or CUDA
stacks can produce different weights even with the same nominal seed.
