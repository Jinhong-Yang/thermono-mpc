# Reproduce the synthetic study

This document has three entry paths. The quick CPU example needs only the source package. The saved-model path needs one JSON/NPZ checkpoint pair. The complete study needs the research-data archive or a fresh synthetic data generation and six model-training runs. None of these paths uses real equipment.

## 1. Small CPU example

With Python 3.11 or newer, run `python -m venv .venv` from a checkout. Activate that environment (PowerShell: `.\.venv\Scripts\Activate.ps1`; POSIX shell: `source .venv/bin/activate`) before the following commands:

```text
python -m pip install -e .
thermono-mpc --help
thermono-mpc doctor
thermono-mpc demo --profile cpu-smoke
python examples/cpu_quickstart.py
python examples/reuse_second_case.py
```

The executable demo is deliberately small. Its output is a smoke result, not one of the paper's 12 scenarios. The four-zone example reuses the plant and physical controllers; three-zone neural weights require retraining before they can be applied to four zones.

## 2. Saved-model closed loop

Install the PyTorch extra with a build appropriate for the host; see `docs/installation_cuda.md` for the tested GPU profile. Obtain a versioned model archive, verify its external SHA-256 against the release receipt, and extract it outside the checkout. The research-data archive has `models/data_only_seed101.json` next to `models/data_only_seed101.npz`, with five more seed/variant pairs. The example checks the checkpoint's internal weight SHA-256 and loads numeric arrays with pickling disabled.

```text
python -m pip install -e ".[torch]"
python examples/model_checkpoint.py --checkpoint PATH_TO_EXTRACTED_ARCHIVE/models/data_only_seed101.json --device cpu
```

The script uses three short partially observed cycles and an eight-candidate CEM budget. It demonstrates loading and command flow; its score is not comparable to the frozen 90-cycle benchmark. The v2 research archive is currently a local draft pending release rights, so there is no valid public download URL yet.

## 3. Full data generation, training and frozen evaluation

The scored source is commit `7beaa86fc6e131a1f7a2cc36208d2811c4f0ca3b`. The exact protocol file `configs/frozen/protocol_v2.yaml` has SHA-256 `339859e89dcc0be653ec8a23c569137efaada298ac1a0cf556ae32a75c0fa8a8`; the 36-trajectory data manifest has SHA-256 `d7dd34e52775c71eca6645f6092b553f5ad9bfaf67a6a75e4d03ff9a90d3d5a1`. The research archive's `provenance/freeze_receipt_v2.json` records both and the test source commit. Work in a separate clean Git checkout at that commit; `predict_eval.py` and `evaluate.py` refuse another commit or a dirty tree. Keep extracted checkpoints and generated output outside the checkout. Activate a Python environment and run:

```text
git switch --detach 7beaa86fc6e131a1f7a2cc36208d2811c4f0ca3b
python -m pip install -e ".[torch]"
```

To regenerate the synthetic training data, run on the frozen checkout:

```text
python scripts/generate_data.py --config configs/frozen/protocol_v2.yaml --output PATH_OUTSIDE_CHECKOUT/training
```

Train both `data_only` and `physics_informed` variants for seeds `101`, `202`, and `303`. Repeat this command for each variant/seed pair, setting `--physics-weight 0.01` for the physics-informed variant and `0` for data-only:

```text
python scripts/train.py --manifest PATH_OUTSIDE_CHECKOUT/training/manifest.json --output PATH_OUTSIDE_CHECKOUT/models --variant physics_informed --seed 101 --epochs 40 --batch-size 8 --learning-rate 0.001 --physics-weight 0.01 --width 24 --layers 4 --modes 8 --patience 10 --device cuda
```

Training on another GPU/software stack need not reproduce identical checkpoint bytes. Retain its metadata, data hash, validation history and device versions; do not substitute newly trained models for the six frozen weights when checking the reported paper numbers.

With the six **frozen** model JSON/NPZ pairs from the versioned research archive, evaluate the distinct full-state oracle forecast and the 120 planned partially observed closed-loop episodes. The `PATH_TO_EXTRACTED_ARCHIVE` directory must contain `models` and `provenance/freeze_receipt_v2.json`:

```text
python scripts/predict_eval.py --protocol configs/frozen/protocol_v2.yaml --freeze-receipt PATH_TO_EXTRACTED_ARCHIVE/provenance/freeze_receipt_v2.json --checkpoints PATH_TO_EXTRACTED_ARCHIVE/models --output PATH_OUTSIDE_CHECKOUT/prediction --device cuda
python scripts/evaluate.py --protocol configs/frozen/protocol_v2.yaml --freeze-receipt PATH_TO_EXTRACTED_ARCHIVE/provenance/freeze_receipt_v2.json --checkpoints PATH_TO_EXTRACTED_ARCHIVE/models --output PATH_OUTSIDE_CHECKOUT/control --device cuda --group all
```

The analysis and figure scripts are in the later development/release candidate source, not the frozen evaluation commit. After the evaluation exits, switch back to the later source while retaining raw results outside the checkout:

```text
git switch main
python -m pip install -e ".[paper]"
python scripts/analyze_results.py --control-dir PATH_OUTSIDE_CHECKOUT/control --prediction-csv PATH_OUTSIDE_CHECKOUT/prediction/prediction_windows.csv --output PATH_OUTSIDE_CHECKOUT/analysis
python scripts/render_paper_assets.py --analysis-dir PATH_OUTSIDE_CHECKOUT/analysis --output PATH_OUTSIDE_CHECKOUT/figures
```

The frozen comparison contains 12 scenarios and 120 controller/seed combinations. `analysis_status.json` must report 120 successes, zero failures, 728 prediction windows and eight held-out prediction trajectories. Scenario or trajectory is the resampling unit; overlapping windows and time cycles are not independent replications. The tested RTX 5080 timing is host-specific. For method definitions, exclusions and the negative FNO/PINO result, see `docs/benchmark_protocol.md` and `docs/limitations.md`.
