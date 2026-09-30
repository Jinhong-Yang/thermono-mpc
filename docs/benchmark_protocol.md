# Frozen v2 benchmark protocol

The confirmatory source commit is `7beaa86fc6e131a1f7a2cc36208d2811c4f0ca3b`. The frozen configuration is `configs/frozen/protocol_v2.yaml`, SHA-256 `339859e89dcc0be653ec8a23c569137efaada298ac1a0cf556ae32a75c0fa8a8`. The local freeze receipt records the data manifest, split, model, controller, observer, scenario and analysis-plan hashes. Version 2 added a held-out prediction test and input dtype handling before any confirmatory test access; the amendment preserves the v1 receipt.

## Data and model

- 24 training, six validation and six calibration whole trajectories on the 16×16 synthetic plant. No adjacent windows from one trajectory cross these splits.
- Direct-horizon predictor: 12 control intervals; width 24, four Fourier layers, eight modes, 40 epochs, seeds 101/202/303. Data-only and physics-informed variants share architecture and labels; the latter uses finite-volume residual weight 0.01.
- Training began under the v1 source for most checkpoints; the v2 source changed only casting of mixed input dtypes in an audit path. All training tensors were already float32. Checkpoint metadata and weights hashes are retained so the exact training artifacts can be distinguished from the final v2 evaluation source.
- Eight separate prediction-test trajectory seeds 9501–9508 on a 32×32 plant. This is a full-state oracle forecast test, separate from the partial-observation control comparison.

## Closed-loop comparison

- Six nominal scenarios N01–N06, four material-shift scenarios S01–S04 and two sensor-stress scenarios F01–F02 on a 32×32 scoring grid.
- Each run has 90 ten-second control cycles, an ambient-to-340 K reference ramp over 600 seconds, and one-cycle command application delay. The controller sees only the fixed sparse probes and a shared causal state estimate.
- Controllers: zone PID; reduced-order physical predictor with CEM; the same physical predictor with SLSQP as a separate optimizer comparison; data-only FNO with CEM, three model seeds; physics-informed FNO with CEM, three model seeds; physics-informed FNO within the deadline-aware runtime boundary, seed 101.
- Matched CEM budget: 64 candidates, three iterations, eight elites, four control blocks. Command deadline: 0.5 seconds. The primary comparison uses the same power/slew limits and nominal thermal feasibility check; plant violations are scored independently.
- Primary outcomes: whole-episode field RMSE, spatial standard deviation, electric energy, solid and zone thermal-limit excess, fallback and deadline rates, decision latency percentiles and maxima. Online latency includes the shared observer, controller and command selection. Training and data-generation time are separate.

## Analysis

The paired unit is a whole scenario. Three neural-model seeds are averaged within each scenario and their range is kept separately. A paired scenario bootstrap is used for categories with at least four scenarios; two-scenario sensor-stress results retain their observed range without a confidence interval. Overlapping trajectory windows and control cycles are not independent replication units. Every planned run ID and failure is reported; a failed run has no invented numeric score and is counted separately from the successful-run summary. No parameter is retuned after test access.

The frozen run completed all 120 planned closed-loop episodes and eight held-out prediction trajectories. The physics-informed operator did not improve tracking over the physical reduced-order comparator. Raw per-run records, aggregate CSV files and an integrity manifest belong with the versioned research-data deposit; release instructions must distinguish the frozen source commit from the later documentation and packaging commit. Run `scripts/analyze_results.py` on the raw summaries and prediction-window CSV to regenerate scenario and trajectory aggregates. Run `scripts/render_paper_assets.py` on those aggregates to regenerate the paper figures.
