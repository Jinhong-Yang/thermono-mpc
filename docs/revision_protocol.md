# Version 0.1.2 controlled computation

The revision computation has its own data, models, experiment receipts and
analyses. It does not replace the 120 frozen control episodes, six original
checkpoints or historical D1-D11 evidence. The installable software contains the
reusable interfaces; the separate computation archive supplies the larger
research inputs and results.

## Entry paths

For a short, complete reuse example without PyTorch:

```text
python examples/predictor_workflow.py --output outputs/predictor_workflow
```

For the numerical collection, extract the version-specific computation archive
into a new directory and follow its `REPRODUCE_R1.txt`. The archive's
`verify_revision_archive.py` checks its file inventory with the Python standard
library. `reproduce_revision.py --mode audit --output NEW_DIRECTORY` recomputes
the reported quantities from saved traces and arrays in a separate directory.
The `full` mode regenerates the added data, trains the declared models and runs
the controlled studies. It needs the recorded scientific/GPU dependencies and
many hours; it is separate from the short CPU example.

An audit of saved records is not a fresh neural inference or a rerun of training.
The full rerun retains its own checkpoints and results because GPU/software
changes need not reproduce identical weight bytes.

## Declared comparisons

| Study | Design and scope |
|---|---|
| Training and control | Three data arms × two neural variants × ten training seeds, evaluated on twelve original and six new control scenarios: 60 models and 1,080 episodes |
| Known-input forecasting | Sixteen independent trajectories and thirteen overlapping forecast windows per trajectory per neural model: 12,480 saved windows; trajectory is the statistical unit |
| Grid/material | 78 episodes with a fixed 32×32 scoring plant and observer; native32 versus restriction16/prolongation32 prediction paths, actual materials and nominal/oracle material channels |
| SLSQP | 48 episodes crossing physical/normalized coordinates with one/two local starts; the physical objective, constraints and candidate acceptance set are fixed |
| Physical references | PID, ROM-CEM and normalized two-start SLSQP on the same eighteen scenarios; existing compatible reference traces are retained |
| Active-limit stress | 216 episodes crossing two materials, three probe-noise/dropout conditions, four independent scenario seeds and nine controllers |
| Serial latency | A separate 36-episode serial study; concurrent diagnostic elapsed times do not establish a latency ranking |

The machine-readable protocol fixes the settings and random seeds before the
new training and confirmatory control studies. Its SHA-256 is
`291220bdb515edec3a4e0300e8040977a57b8fef29d97185906214fdc21890e1`.
Raw outcomes and explicit failure records are retained rather than filtering
seeds or scenarios according to their performance.

## Data lineage and interventions

The original development collection has 24 training, six validation and six
reserved-calibration whole trajectories. Each expanded training arm preserves
the original 24 and adds 72; validation and calibration files are identical.
The calibration collection remains unused. Uniform and targeted additions
share their material and initial-state draws. Targeted excitation changes only
the declared target-power sampling rule before the original slew limiter.

The original eight forecast-test trajectories and eight additional trajectories
are separate from development data. Closed-loop scenarios produce independent
feedback trajectories. Training uses supervised validation loss for checkpoint
selection, with no test-based seed, model or hyperparameter selection. The same
40-epoch cap does not equalize optimizer-update counts across 24/96-trajectory
arms; actual updates and elapsed training time are reported.

Control intervals resample complete training seeds and scenarios. Forecast
intervals resample complete trajectories, together with the training-seed axis
where applicable. Correlated windows and controller calls are not independent
replicates. The sixteen trajectories and eighteen designed scenarios do not
constitute an industrial deployment population.

The reported 95% intervals are pointwise percentile intervals from 10,000
resamples (random seed 53117), without multiple-comparison adjustment. Paired
training-arm contrasts resample matched seed/scenario differences; forecast
windows are averaged within each trajectory before resampling. Each arm uses a
fixed training collection across its ten model-training seeds. This measures
training-seed variation conditional on that collection, not variation across
independently regenerated training datasets.

The serial timing study runs one episode per configuration on N03/S01/F01 after
the other declared experiments and their initial analyses finish. Twelve
configurations comprise PID, ROM-CEM, four SLSQP scaling/start settings and the
six original neural checkpoints. Configuration order is shuffled with seed
53117 within each scenario. The first decision is not a process/GPU cold-start
measurement, and model loading is excluded. Subsequent-cycle statistics include
fallback decisions and summarize the remaining 89 decisions in each episode.
Reported mean episode medians and mean episode 95th percentiles average three
scenario-specific values; they are not pooled percentiles or repeated trials of
one scenario. These shared-desktop measurements are not worst-case execution-time
or hard-real-time guarantees.

The recorded serial study has six decisions above 0.5 s, all in normalized
two-start SLSQP (6/270 decisions; maximum 672.78 ms). These synchronous decisions
are measured without deadline rejection. Accuracy improvement therefore does
not imply compliance with the soft acceptance budget. All 36 numerical control
traces match their specified reference traces exactly, excluding elapsed time.

## Recorded training-intervention outcomes

These are newly trained models, separate from the six original checkpoints.
Each mean uses ten training seeds and eighteen control scenarios. The low-energy
column counts episodes below the descriptive 0.2 MJ cutoff, out of 180 per model
variant; it is not an efficiency or safety criterion.

| Training arm | FNO mean RMSE (K) | PINO mean RMSE (K) | Low-energy episodes, FNO / PINO |
|---|---:|---:|---:|
| original24 | 23.48 | 25.16 | 59 / 78 |
| uniform96 | 23.63 | 21.59 | 52 / 36 |
| targeted96 | 15.00 | 14.00 | 0 / 0 |

Targeted96 minus uniform96 has paired differences of -8.62 K (pointwise 95%
interval -14.44 to -2.80) for FNO and -7.59 K (-13.09 to -2.66) for PINO. The
six new scenarios give targeted-arm means of 14.74/14.01 K. Uniform expansion
alone has paired intervals spanning zero. The targeted-arm mean errors remain
above the matched ROM-CEM/scaled-SLSQP values of 11.60/10.79 K; there is no
general neural-controller superiority claim.

Nonmonotone local response counts decrease from 66/90 and 63/90 in uniform96 to
8/90 and 6/90 in targeted96. Strict plan-ranking reversals nevertheless persist
in 89/90 and 87/90 targeted-arm calls. The workflow therefore measures a useful
closed-loop improvement and exposes an unresolved diagnostic failure. Different
arms visit different states, and prediction from estimated state is compared
with replay from aligned scoring state. These descriptive local changes do not
identify a single necessary or sufficient failure mechanism.

The numerical archive's `analysis/expanded/analysis.json`, per-seed summaries
and complete raw records retain the full results, uncertainty and subgroup
information. All declared models and scenarios are included.

## Interpretation boundaries

- Grid restriction and prolongation change the complete prediction path. They
  do not isolate Fourier resolution alone. True-material channels are a
  privileged diagnostic; the observer retains its nominal parameters.
- Local plan diagnostics compare competing actions at one shared state and
  held command. Different training arms subsequently visit different states,
  so aggregate diagnostic differences across arms are descriptive.
- Normalized SLSQP follows historical scaling evidence. Its comparison on the
  original scenarios is a controlled re-comparison, not an untouched tuning
  holdout. One-iteration termination alone is not evidence of failure.
- Active-limit stress uses limits selected from separate development runs.
  Probe noise/dropout clears after 600 s, while material perturbations persist
  and zone observations remain exact. Criterion attainment is distinguished
  from recovery of a case failing that criterion at clearance.
- Stress durations and excess integrals weight recorded 10 s boundaries. The
  preserved `within_2K_of_any_limit_duration_s` field counts time at or above
  either limit minus 2 K, including exceedances. The separate stress-metric
  readback decomposes it into non-violating near-limit time and any-limit-excess
  time. None of these metrics rules out intersample peaks.
- Predicted-plan rejection and PID fallback do not guarantee realized thermal
  constraints. Low heating with poor tracking is not a safety or efficiency
  achievement. The synthetic plant is uncalibrated to equipment.

Exact executed-source snapshots accompany training and control jobs. The data
generator was captured at an independent audit after generation, rather than
being represented as a pre-generation snapshot. The audit regenerates one
matched addition per expanded arm and checks every manifest and paired initial
condition. See the numerical archive for complete provenance and output hashes.
