"""Generate synthetic trajectories from a validated experiment schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from thermono_mpc.dataset import generate_dataset
from thermono_mpc.process import ProcessConfig


def read_experiment(path: Path):
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj.get("schema_version") != "experiment-1":
        raise ValueError("unknown experiment schema")
    c = ProcessConfig(**obj["process"])
    data = obj["data"]
    if data["control_dt_s"] <= 0 or data["max_step_s"] <= 0 or data["horizon"] < 1:
        raise ValueError("invalid experiment time settings")
    return c, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c, d = read_experiment(args.config)
    counts = {"train": d["train_trajectories"],
              "validation": d["validation_trajectories"]}
    for name in ("development", "calibration"):
        key = name + "_trajectories"
        if key in d:
            counts[name] = d[key]
    path = generate_dataset(args.output, c, counts=counts,
                            steps=d["steps"], horizon=d["horizon"],
                            control_dt_s=d["control_dt_s"],
                            max_step_s=d["max_step_s"], seed=d["seed"])
    print(json.dumps({"manifest": str(path), "trajectories": sum(counts.values())}))
