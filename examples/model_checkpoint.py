"""Small synthetic closed-loop demonstration using a saved numeric-array model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermono_mpc.controllers import MPCConfig
from thermono_mpc.dataset import file_sha256
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import Scenario, run_episode
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="JSON metadata next to its numeric .npz weights")
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/frozen/protocol_v2.yaml"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    metadata = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    weight_name = metadata["weights"]
    if Path(weight_name).name != weight_name or not weight_name.endswith(".npz"):
        raise ValueError("checkpoint weights must be a local .npz filename")
    weights = args.checkpoint.parent / weight_name
    if file_sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    variant = metadata["variant"]
    controller = {"data_only": "B2_DATA_NO_MPC",
                  "physics_informed": "B3_PINO_MPC"}[variant]
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    process = ProcessConfig(**protocol["process"])
    horizon = int(metadata["architecture"]["horizon"])
    settings = MPCConfig(horizon=horizon, blocks=4, candidates=8,
                         iterations=1, elite=2)
    summary, _ = run_episode(
        process, Scenario("saved-model-demo", 7), controller,
        steps=3, dt_s=10.0, max_step_s=2.0, horizon=horizon,
        target_K=300.0, ramp_s=30.0, observation="partial",
        checkpoint=args.checkpoint, device=args.device,
        mpc_settings=settings,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
