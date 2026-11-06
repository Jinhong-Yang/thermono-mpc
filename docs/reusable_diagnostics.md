# Reusing prediction, control and diagnostic workflows

The prediction contract is thermal-field-specific and independent of a particular
neural architecture. `Predictor.predict(field_K, zones_K, candidates_W, dt_s)`
returns field `[B,H,Ny,Nx]` and zone `[B,H,Z]` trajectories in kelvin. Plans have
shape `[B,H,Z]` in watts. Inputs and outputs must be finite. Forecast index zero
is the end of the first proposed input interval.

## Reuse boundaries

| Component | Reused directly for another compatible predictor | Caller supplies |
|---|---|---|
| CEM selection | `CEMMPC` | Predictor adapter, MPC and process configuration |
| Closed-loop evaluation | `run_episode(..., "CUSTOM_CEM", predictor=model)` | Scenario and compatible field/zone output |
| Same-plan replay | `same_plan_replay` | Aligned initial estimate, full plan, dt, replay callback |
| Common-state comparison | `common_state_comparison` | Plans generated at the same estimated state and held power; replay and objective callbacks |
| Thermal objective | `plan_objective` | Reference horizon, previous power and explicit weights/scales |
| Thermal replay adapter | `thermal_replay` | A plant and a delay-aligned evaluation-state clone |
| Benchmark orchestration | Historical D1/D9 scripts are study-specific | Scenario IDs, model paths, data locations and protocol |

The reusable diagnostic calculations have no FNO/PINO checkpoint names, scenario
IDs, fixed 10-second interval, fixed 12-step horizon or 0.2-MJ classification.
The thermal objective and plant adapter retain their stated physical conventions;
they are not generic models for unrelated domains.

## State alignment and information availability

For the one-cycle-delay benchmark, apply the held input to the cloned evaluation
state first, then replay the complete candidate plan without reoptimization.
Pass the projected estimated field and zones to the predictor. The difference
from replay includes estimation error. A separate true-start prediction can
remove that initialization discrepancy for diagnosis; true state must not enter
the actual controller.

Every plan in a common-state comparison must be selected or constructed at the
same estimated state, held command, time, reference, horizon and objective
settings. Plans borrowed from unrelated closed-loop trajectories do not satisfy
that comparison. The replay callback must reset to an identical evaluation state
for each plan. `thermal_replay` does this by cloning the supplied state.

Counterfactual replay requires a simulator or another justified evaluation model.
Measured sensor logs alone cannot reveal the outcome of an unexecuted plan.
For a new physical process, the user must supply a process/replay adapter and
validate its accuracy. A protocol-compatible predictor by itself is not sufficient
to transfer the thermal plant, state reconstruction, or physical assumptions.

## End-to-end CPU example

```console
python examples/predictor_workflow.py --output outputs/predictor_workflow
```

The example constructs `LinearThermalPredictor`, a constant-conductance linear
thermal state-space model with exact matrix-exponential time stepping. It first
evaluates a known power plan, then runs 30 partially observed control cycles,
then executes three common-state comparisons against newly selected ROM plans.
The model-specific adapter is the only replacement component; the optimizer,
episode evaluator and diagnostic calculation functions are reused unchanged.
Outputs include the closed-loop CSV, per-call diagnostic CSV, numeric prediction
and replay arrays, and a summary. The example is a reuse demonstration, not a
benchmark or a claim that this simple predictor is optimal.

The earlier `custom_predictor.py` persistence example remains a minimal interface
example. Its predictions ignore power; use the responsive example above for the
complete evaluation and diagnosis workflow.

## Time and horizon capabilities

Released original FNO/PINO weights were trained for 12 steps at 10 seconds.
`TorchPredictor` rejects a different interval, invalid/non-finite dt or a candidate
horizon/zone mismatch before inference. New checkpoint metadata records
`control_dt_s`. For the six original checkpoints, the loader recognizes the exact
archived training-manifest hash and supplies its verified 10-second interval.
Unknown legacy metadata without an interval raises an error rather than guessing.
Calling `TorchPredictor` directly requires choosing `trained_dt_s` (default 10 s).

Physical or other predictors can support different horizons and intervals;
`LinearThermalPredictor`, for example, computes the transition for a positive dt.
The common signature does not promise that every model supports every value.

## Diagnostic interpretation

Same-plan forecast error describes the specified plan and starting state, not
necessarily the subsequent receding-horizon trajectory. Model preference for its
own optimized plan is favoured by construction; realized costs supply a separate
comparison. Calls within an episode overlap and are correlated. Do not count
them as independent training seeds or test trajectories.
