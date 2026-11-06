# Changelog

## 0.1.2 (2026-11-07)

- Extracted same-plan replay and common-state plan comparison into reusable library functions with explicit replay/objective callbacks.
- Added a responsive linear thermal predictor and an executable CPU forecast, closed-loop and diagnostic workflow.
- Added interval/horizon validation and verified legacy metadata handling for archived neural checkpoints.
- Added an explicit SLSQP scaling/start comparator while retaining the original SLSQP implementation.
- Added a nested-grid prediction adapter and an optional sensor disturbance schedule for controlled diagnostic experiments.
- Preserved all original benchmark, checkpoint and D1-D11 evidence; revision studies have separate protocols, sources and outputs.
- Version-specific software archive: https://doi.org/10.5281/zenodo.24019479; computation: https://doi.org/10.5281/zenodo.24019545.

## 0.1.1 (2026-10-01)

- Added documented reviewer diagnostics and packaged reproducibility material.
- Public software archive: https://doi.org/10.5281/zenodo.23073711.

## 0.1.0 (development)

- Added synthetic conservative thermal process and numerical self-checks.
- Added direct-horizon data-only and physics-informed FNO paths, checkpoint hashes and whole-trajectory data generation.
- Added PID, physical ROM-CEM and ROM-SLSQP comparisons, sparse observer and deadline-aware command boundary.
- Added frozen v2 120-run control and eight-trajectory prediction protocol, reproducibility scripts and documentation.
- Added an explicit CPU smoke profile and a hash-checked saved-model example; `REPRODUCE.md` separates quick, saved-model and full-study paths.
- Pinned third-party CI actions to reviewed commits with read-only workflow permissions and distribution checks.
- Added an optional local-process command selector with compact zone-mean transport and separate PID fallback. Its CPU example and process-failure tests are outside the frozen 120-run study. Validation now checks receipt time as well as compute finish time against the deadline.
- Added a measured CPU contention probe, explicit candidate-infeasibility and malformed-command fault cases, and process-restart version rejection.
- Revised publication-size figures with legible labels, explicit applied-power and candidate-prediction paths, whole-trajectory confidence intervals, and a plotted-data provenance file. Frozen measurements are unchanged.
- The frozen synthetic study did not find an FNO/PINO tracking advantage over ROM-CEM. See `docs/model_card.md` and `docs/limitations.md`.

The entries above distinguish the historical releases and benchmark from the version 0.1.2 revision.
