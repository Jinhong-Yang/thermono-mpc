"""Synthetic trajectory generator with whole-trajectory split manifests."""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from .process import ProcessConfig, ThermalPlant


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def random_powers(rng: np.random.Generator, steps: int, c: ProcessConfig) -> np.ndarray:
    powers = np.zeros((steps, c.zones))
    previous = np.zeros(c.zones)
    for t in range(steps):
        if t % 4 == 0:
            target = rng.uniform(0, c.pmax)
        previous = np.clip(target, np.maximum(0, previous - c.slew),
                           np.minimum(c.pmax, previous + c.slew))
        powers[t] = previous
    return powers


def generate_trajectory(config: ProcessConfig, seed: int, steps: int,
                        control_dt_s: float, max_step_s: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Parameters change only between trajectories; the controller can receive
    # nominal values while scoring retains actual values in private metadata.
    c = replace(config, rho=config.rho * rng.uniform(.95, 1.05),
                cp=config.cp * rng.uniform(.95, 1.05),
                k=config.k * rng.uniform(.9, 1.1))
    plant = ThermalPlant(c)
    state = plant.reset(config.initial + rng.uniform(-5., 5.))
    controls = random_powers(rng, steps, c)
    fields = np.empty((steps + 1, c.ny, c.nx))
    zones = np.empty((steps + 1, c.zones))
    fields[0], zones[0] = state.field, state.zones
    for t, power in enumerate(controls):
        state = plant.advance(state, power, control_dt_s, max_step_s)
        fields[t + 1], zones[t + 1] = state.field, state.zones
    return {"field_K": fields, "zones_K": zones, "power_W": controls,
            "material": np.array([c.rho / 7800., c.cp / 500., c.k / 40.]),
            "seed": np.array(seed), "control_dt_s": np.array(control_dt_s)}


def generate_dataset(root: Path, config: ProcessConfig, *, counts: dict[str, int],
                     steps: int, horizon: int, control_dt_s: float,
                     max_step_s: float, seed: int = 20260930) -> Path:
    if steps < horizon or any(n < 1 for n in counts.values()):
        raise ValueError("each split needs at least one full-horizon trajectory")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "thermono-data-1", "config": asdict(config),
                "steps": steps, "horizon": horizon, "control_dt_s": control_dt_s,
                "max_step_s": max_step_s, "split_unit": "whole_trajectory",
                "trajectories": []}
    index = 0
    for split, count in counts.items():
        for _ in range(count):
            trajectory_seed = seed + index
            payload = generate_trajectory(config, trajectory_seed, steps, control_dt_s, max_step_s)
            name = f"trajectory_{index:04d}.npz"
            path = root / name
            np.savez_compressed(path, **payload)
            manifest["trajectories"].append({"id": index, "split": split,
                                              "seed": trajectory_seed, "file": name,
                                              "sha256": file_sha256(path)})
            index += 1
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def load_windows(manifest_path: Path, split: str) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    h = manifest["horizon"]
    output: dict[str, list[np.ndarray]] = {key: [] for key in ("field", "zone", "power", "material", "target_field", "target_zone", "trajectory_id")}
    for item in manifest["trajectories"]:
        if item["split"] != split:
            continue
        path = manifest_path.parent / item["file"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"trajectory checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            field, zones, power, material = (data[x] for x in ("field_K", "zones_K", "power_W", "material"))
            for t in range(len(power) - h + 1):
                output["field"].append(field[t].astype("float32"))
                output["zone"].append(zones[t].astype("float32"))
                output["power"].append(power[t:t + h].astype("float32"))
                output["material"].append(material.astype("float32"))
                output["target_field"].append(field[t + 1:t + h + 1].astype("float32"))
                output["target_zone"].append(zones[t + 1:t + h + 1].astype("float32"))
                output["trajectory_id"].append(np.array(item["id"]))
    if not output["field"]:
        raise ValueError(f"empty split: {split}")
    return {key: np.stack(values) for key, values in output.items()}
