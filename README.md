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
```

The CPU example uses a small grid and runs without PyTorch. `reuse_second_case.py` changes geometry, material and the number of heater zones while reusing the same plant and controller APIs; a three-zone FNO checkpoint is not transferable to that four-zone example without retraining.

## Model path

Install the optional PyTorch dependency with `python -m pip install -e ".[torch]"`. The tested CUDA environment is documented in `docs/hardware_profile.md`; this does not imply support for a PLC, Jetson or other edge device. Checkpoints contain numeric arrays in `.npz` plus JSON metadata and SHA-256, and are loaded with pickling disabled.

The model takes an estimated current temperature field, measured zone temperatures, nominal material context and the full proposed future heater-power sequence. It predicts a full field and zone-temperature trajectory. The data-only and physics-informed variants use the same architecture. A low rollout error does not guarantee that an optimizer will choose useful actions.

## Reproduce the frozen benchmark

Follow `REPRODUCE.md` for the CPU smoke, saved-model example and full frozen run; read `docs/benchmark_protocol.md` and the research-data manifest before a full run. The exact test source is the frozen commit recorded in the v2 freeze receipt; a later release commit adds documentation and a CPU CLI profile while retaining the evaluated numerical process, controller and model code. Generate the 36 synthetic whole-trajectory data files from the frozen configuration; verify their manifest hash; train the six declared seed/variant combinations or obtain the versioned checkpoint archive; run held-out prediction and all 120 planned closed-loop episodes. The analysis script aggregates at whole-trajectory or whole-scenario grain and reports failures separately. Do not infer statistical replication from overlapping windows or control cycles.

The benchmark scores a 32×32 synthetic plant with nominal, material-shift and sensor-stress scenarios, sparse observations and a one-cycle command delay. PID, reduced-order CEM, reduced-order SLSQP, data-only FNO–CEM, PINO–CEM and deadline-aware PINO runtime runs receive the same declared conditions. SLSQP is a separate optimizer comparison; the matched predictor comparison uses CEM. Full-state oracle prediction results are labelled separately from partial-observation control results.

## Package map

- `src/thermono_mpc/process.py`: synthetic thermal plant and conservative finite-volume solver.
- `contracts.py`, `runtime.py`: state/prediction/command contracts, validation, single-worker supervision and independent fallback.
- `operator.py`, `physics_loss.py`: direct-horizon FNO, safe checkpoint loader and differentiable residual.
- `controllers.py`, `estimation.py`: PID, physical ROM, CEM/SLSQP and sparse observer.
- `dataset.py`, `simulation.py`: whole-trajectory generation and delayed-command closed-loop evaluation.
- `configs/frozen`, `scripts`, `tests`, `docs`, `examples`, `REPRODUCE.md`: protocol, reproducibility commands, checks and user guidance.

## Scope and limitations

The field and zone dynamics are synthetic design assumptions. The front/back faces are insulated, coefficients are constant within a trajectory, and phase changes and physical sensor calibration are absent. Numerical convergence and energy-balance tests establish implementation consistency only. A software timeout can reject stale or late commands, but it cannot preempt a running GPU kernel or guarantee bounded execution time on general-purpose hardware. Every empirical result is tied to the released protocol, data, model and run hashes; unfavorable comparisons are retained.

## License and citation

Pending rights-holder approval. The preferred candidate is the OSI-approved MIT license. Public release metadata, exact version link and dataset citation will be added only after verification.
