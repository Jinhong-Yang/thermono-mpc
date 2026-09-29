"""Evaluate frozen full-state-oracle forecasting on held-out trajectories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import yaml
import numpy as np
import torch

from thermono_mpc.dataset import generate_trajectory, file_sha256
from thermono_mpc.controllers import ROMPredictor
from thermono_mpc.physics_loss import finite_volume_residual
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import make_predictor


def main(protocol_path: Path, receipt_path: Path, checkpoint_dir: Path,
         output: Path, device: str):
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN" or file_sha256(protocol_path) != receipt["protocol_sha256"]:
        raise RuntimeError("not the frozen prediction protocol")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if sha != receipt["source_commit"] or dirty:
        raise RuntimeError("source differs from frozen commit")
    plan = protocol["prediction_test"]
    base = ProcessConfig(**protocol["process"]).with_grid(*plan["grid"])
    h = protocol["data"]["horizon"]
    models = {}
    for variant in ("data_only", "physics_informed"):
        for seed in protocol["training"]["seeds"]:
            meta = checkpoint_dir / f"{variant}_seed{seed}.json"
            info = json.loads(meta.read_text(encoding="utf-8"))
            if file_sha256(checkpoint_dir / info["weights"]) != info["weights_sha256"]:
                raise RuntimeError("checkpoint checksum mismatch")
            models[(variant, seed)] = make_predictor(base, meta, device)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in plan["trajectory_seeds"]:
        print(f"prediction trajectory {seed}", flush=True)
        trajectory = generate_trajectory(base, seed, plan["steps"],
                                         plan["control_dt_s"], plan["max_step_s"])
        raw_path = output / f"trajectory_{seed}.npz"
        np.savez_compressed(raw_path, **trajectory)
        rom = ROMPredictor(base)
        for t in range(plan["steps"] - h + 1):
            field = trajectory["field_K"][t]
            zones = trajectory["zones_K"][t]
            future = trajectory["power_W"][t:t + h]
            target_field = trajectory["field_K"][t + 1:t + h + 1]
            target_zone = trajectory["zones_K"][t + 1:t + h + 1]
            for key, predictor in [("ROM", rom), *models.items()]:
                prediction = predictor.predict(field, zones, future[None], plan["control_dt_s"])
                pf, pz = prediction.temperature_K[0], prediction.zone_temperature_K[0]
                field_rmse = float(np.sqrt(np.mean((pf - target_field)**2)))
                zone_rmse = float(np.sqrt(np.mean((pz - target_zone)**2)))
                physics_residual = None
                if key != "ROM":
                    rf, rz = finite_volume_residual(
                        torch.as_tensor(pf[None], device=device),
                        torch.as_tensor(pz[None], device=device),
                        torch.as_tensor(field[None], device=device),
                        torch.as_tensor(zones[None], device=device),
                        torch.as_tensor(future[None], device=device),
                        plan["control_dt_s"], base)
                    physics_residual = float((rf.square().mean() + rz.square().mean()).cpu())
                variant = key if isinstance(key, str) else key[0]
                model_seed = None if isinstance(key, str) else key[1]
                rows.append({"trajectory_seed": seed, "window_start": t,
                             "variant": variant, "model_seed": model_seed,
                             "field_rmse_K": field_rmse,
                             "zone_rmse_K": zone_rmse,
                             "physics_residual_normalized": physics_residual,
                             "trajectory_sha256": file_sha256(raw_path)})
    path = output / "prediction_windows.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"windows": len(rows), "path": str(path),
                      "sha256": file_sha256(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    main(args.protocol, args.freeze_receipt, args.checkpoints, args.output, args.device)
