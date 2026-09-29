"""Train matched data-only and physics-informed direct-horizon FNO variants."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from thermono_mpc.dataset import file_sha256, load_windows
from thermono_mpc.operator import DirectFNO
from thermono_mpc.physics_loss import finite_volume_residual, supervised_loss
from thermono_mpc.process import ProcessConfig


def tensors(manifest: Path, split: str, device: str):
    return {key: torch.as_tensor(value, device=device) for key, value in load_windows(manifest, split).items() if key != "trajectory_id"}


def save_weights(model: DirectFNO, path: Path) -> None:
    payload = {}
    for key, value in model.state_dict().items():
        data = value.detach().cpu().numpy()
        if np.iscomplexobj(data):
            payload[key + ".real"] = data.real
            payload[key + ".imag"] = data.imag
        else:
            payload[key] = data
    np.savez_compressed(path, **payload)


def load_weights(model: DirectFNO, path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        state = {}
        for key, value in model.state_dict().items():
            if value.is_complex():
                arr = data[key + ".real"] + 1j * data[key + ".imag"]
            else:
                arr = data[key]
            state[key] = torch.as_tensor(arr, dtype=value.dtype)
        model.load_state_dict(state)


def train(manifest: Path, output: Path, variant: str, seed: int, *,
          epochs: int, batch_size: int, learning_rate: float,
          physics_weight: float, width: int, layers: int, modes: int,
          patience: int, device: str) -> dict:
    info = json.loads(manifest.read_text(encoding="utf-8"))
    c = ProcessConfig(**info["config"])
    h = info["horizon"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_data = tensors(manifest, "train", device)
    valid_data = tensors(manifest, "validation", device)
    model = DirectFNO(c.zones, h, width, layers, modes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    log = []
    weights = output / f"{variant}_seed{seed}.npz"
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_data["field"]), device=device)
        total_data = total_physics = 0.
        for start in range(0, len(order), batch_size):
            ix = order[start:start + batch_size]
            f, z = model(train_data["field"][ix], train_data["zone"][ix],
                         train_data["power"][ix], train_data["material"][ix])
            data_loss = supervised_loss(f, z, train_data["target_field"][ix], train_data["target_zone"][ix])
            if physics_weight:
                rf, rz = finite_volume_residual(f, z, train_data["field"][ix],
                                                train_data["zone"][ix], train_data["power"][ix],
                                                info["control_dt_s"], c)
                physics_loss = rf.square().mean() + rz.square().mean()
            else:
                physics_loss = torch.zeros((), device=device)
            loss = data_loss + physics_weight * physics_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_data += float(data_loss.detach()) * len(ix)
            total_physics += float(physics_loss.detach()) * len(ix)
        model.eval()
        with torch.inference_mode():
            vf, vz = model(valid_data["field"], valid_data["zone"],
                           valid_data["power"], valid_data["material"])
            validation = float(supervised_loss(vf, vz, valid_data["target_field"], valid_data["target_zone"]))
        log.append({"epoch": epoch, "train_data_loss": total_data / len(order),
                    "train_physics_loss": total_physics / len(order), "validation_loss": validation})
        if validation < best - 1e-8:
            best = validation
            save_weights(model, weights)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    log_path = output / f"{variant}_seed{seed}_history.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log[0]))
        writer.writeheader(); writer.writerows(log)
    result = {"variant": variant, "seed": seed, "best_validation_loss": best,
              "epochs_completed": len(log), "weights": weights.name,
              "weights_sha256": file_sha256(weights), "manifest_sha256": file_sha256(manifest),
              "architecture": {"zones": c.zones, "horizon": h, "width": width,
                               "layers": layers, "modes": modes},
              "physics_weight": physics_weight, "device": device,
              "torch_version": torch.__version__}
    (output / f"{variant}_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["data_only", "physics_informed"], required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.variant == "data_only":
        args.physics_weight = 0.
    result = train(args.manifest, args.output, args.variant, args.seed,
                   epochs=args.epochs, batch_size=args.batch_size,
                   learning_rate=args.learning_rate, physics_weight=args.physics_weight,
                   width=args.width, layers=args.layers, modes=args.modes,
                   patience=args.patience, device=args.device)
    print(json.dumps(result, indent=2))
