# Changelog

## 0.1.0 (development)

- Added synthetic conservative thermal process and numerical self-checks.
- Added direct-horizon data-only and physics-informed FNO paths, checkpoint hashes and whole-trajectory data generation.
- Added PID, physical ROM-CEM and ROM-SLSQP comparisons, sparse observer and deadline-aware command boundary.
- Added frozen v2 120-run control and eight-trajectory prediction protocol, reproducibility scripts and documentation.
- Added an explicit CPU smoke profile and a hash-checked saved-model example; `REPRODUCE.md` separates quick, saved-model and full-study paths.
- Pinned third-party CI actions to reviewed commits with read-only workflow permissions and distribution checks.
- Added an optional local-process command selector with compact zone-mean transport and separate PID fallback. Its CPU example and process-failure tests are outside the frozen 120-run study. Validation now checks receipt time as well as compute finish time against the deadline.
- Added a measured CPU contention probe, explicit candidate-infeasibility and malformed-command fault cases, and process-restart version rejection.
- The frozen synthetic study did not find an FNO/PINO tracking advantage over ROM-CEM. See `docs/model_card.md` and `docs/limitations.md`.

There is no public tagged release yet. A later release note will identify the exact source commit, artifacts and rights-approved license.
