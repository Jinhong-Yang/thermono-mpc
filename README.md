# ThermoNO-MPC

ThermoNO-MPC is a Python research package for synthetic multi-zone heat-treatment control. It combines a conservative finite-volume thermal plant, candidate-conditioned direct-horizon Fourier neural operator, optional physics residual training, PID and reduced-order MPC comparisons, sparse-probe state estimation, and a software command boundary with freshness, ordering and deadline checks.

The included process parameters are synthetic. No model is calibrated to manufacturing equipment, and this package provides neither equipment interlocks nor hard-real-time guarantees. Do not connect the examples to a live heater.

## Install and try the CPU path

Python 3.11 or newer is required. From a clean checkout:

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -e .
thermono-mpc doctor
thermono-mpc demo --profile cpu-smoke
python examples/cpu_quickstart.py
python examples/reuse_second_case.py
python examples/custom_predictor.py
python examples/process_boundary_demo.py
python scripts/runtime_overload_probe.py
```

The CPU examples run without PyTorch. `reuse_second_case.py` changes geometry, material and the number of heater zones while reusing the same plant and controller APIs; a three-zone FNO checkpoint is not transferable to that four-zone example without retraining. `process_boundary_demo.py` starts a separate local command-selection process and sends only compact zone means and command metadata across its IPC transport. `runtime_overload_probe.py` measures CPU contention and verifies that a late candidate is rejected while a second queued job is refused. These are software demonstrations separate from the frozen performance results.

## Model path

Install the optional PyTorch dependency with `python -m pip install -e ".[torch]"`. The tested CUDA environment is documented in `docs/hardware_profile.md`; this does not imply support for a PLC, Jetson or other edge device. Checkpoints contain numeric arrays in `.npz` plus JSON metadata and SHA-256, and are loaded with pickling disabled.

The model takes an estimated current temperature field, measured zone temperatures, nominal material context and the full proposed future heater-power sequence. It predicts a full field and zone-temperature trajectory. The data-only and physics-informed variants use the same architecture. A low rollout error does not guarantee that an optimizer will choose useful actions.

## Reproduce the frozen benchmark

Follow `REPRODUCE.md` for the CPU smoke, saved-model example and full frozen run; read `docs/benchmark_protocol.md` and the research-data manifest before a full run. The exact test source is the frozen commit recorded in the v2 freeze receipt; a later release commit adds documentation and a CPU CLI profile while retaining the evaluated numerical process, controller and model code. Generate the 36 synthetic whole-trajectory data files from the frozen configuration; verify their manifest hash; train the six declared seed/variant combinations or obtain the versioned checkpoint archive; run held-out prediction and all 120 planned closed-loop episodes. The analysis script aggregates at whole-trajectory or whole-scenario grain and reports failures separately. Do not infer statistical replication from overlapping windows or control cycles.

The benchmark scores a 32×32 synthetic plant with nominal, material-shift and sensor-stress scenarios, sparse observations and a one-cycle command delay. PID, reduced-order CEM, reduced-order SLSQP, data-only FNO–CEM, PINO–CEM and deadline-aware PINO runtime runs receive the same declared conditions. SLSQP is a separate optimizer comparison; the matched predictor comparison uses CEM. Full-state oracle prediction results are labelled separately from partial-observation control results.

The supplementary [post-freeze diagnostics](docs/post_freeze_diagnostics.md)
inspect selected plans, action coverage, grid transfer, residual weights,
optimization budgets, numerical refinement and offline cost. Their outputs
are separate from the frozen benchmark. See the
[predictor extension guide](docs/predictor_extension.md) for a runnable custom
predictor and the [related-software comparison](docs/related_software.md) for
the package's documented scope.

## Package map

- `src/thermono_mpc/process.py`: synthetic thermal plant and conservative finite-volume solver.
- `contracts.py`, `runtime.py`, `process_boundary.py`: state/prediction/command contracts, validation, single-worker supervision, optional local process transport and independent fallback.
- `operator.py`, `physics_loss.py`: direct-horizon FNO, safe checkpoint loader and differentiable residual.
- `controllers.py`, `estimation.py`: PID, physical ROM, CEM/SLSQP and sparse observer.
- `dataset.py`, `simulation.py`: whole-trajectory generation and delayed-command closed-loop evaluation.
- `configs/frozen`, `scripts`, `tests`, `docs`, `examples`, `REPRODUCE.md`: protocol, reproducibility commands, checks and user guidance.

## Scope and limitations

The field and zone dynamics are synthetic design assumptions. The front/back faces are insulated, coefficients are constant within a trajectory, and phase changes and physical sensor calibration are absent. Numerical convergence and energy-balance tests establish implementation consistency only. A software timeout can reject stale or late commands, but it cannot preempt a running GPU kernel or guarantee bounded execution time on general-purpose hardware. Every empirical result is tied to the released protocol, data, model and run hashes; unfavorable comparisons are retained.

## License and citation

Apache-2.0 licensed software release (2026). Copyright 2026 Jinhong Yang.
See `LICENSE`, `NOTICE`, and `CITATION.cff`. Third-party dependencies retain
their respective terms in `THIRD_PARTY_NOTICES.md`.

Version-specific archive identifiers:

- [Software v0.1.0](https://doi.org/10.5281/zenodo.23051699).
- [Synthetic data, frozen checkpoints and post-freeze diagnostics](https://doi.org/10.5281/zenodo.23051736).

The research deposit contains a SHA-256 inventory and `verify_archive.py`.
It preserves frozen evidence and stores exploratory diagnostics separately.


## v0.1.1 reviewer diagnostics

Version 0.1.1 adds [D9/D10](docs/reviewer_diagnostics.md), separate source
snapshots, regression checks and Python 3.11/3.12/3.13 CI jobs. Frozen benchmark
algorithms and outputs remain unchanged. A small field-prediction error does
not imply good plan ordering: the diagnostics compare common-state plans and
inspect candidate feasibility and power response. Original SLSQP replay records
show 2,035 of 2,158 calls ending after one iteration, **not all calls**; 123
iterate further. The D6 five-start diagnostic separately has 5,395 one-iteration
calls. The variable-scaled single-start D10 diagnostic has mean control RMSE
10.64 K over 12 scenarios, compared with the unchanged 41.36 K frozen result.
The single-start intervention and scaling are specified together; this is not
a general ranking of optimizers. Research data are synthetic.
