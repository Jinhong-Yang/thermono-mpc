"""Execute every frozen scenario/controller pair without post-test tuning."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import yaml

from thermono_mpc.controllers import MPCConfig
from thermono_mpc.dataset import file_sha256
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import Scenario, run_episode


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config_path: Path, freeze_receipt: Path, checkpoints: Path, output: Path, device: str,
             group: str = "all") -> None:
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN":
        raise RuntimeError("confirmatory evaluation requires FROZEN protocol")
    freeze = json.loads(freeze_receipt.read_text(encoding="utf-8"))
    if file_sha256(config_path) != freeze["protocol_sha256"]:
        raise RuntimeError("protocol checksum differs from freeze receipt")
    if git_sha() != freeze["source_commit"] or git_dirty():
        raise RuntimeError("source commit differs from freeze or working tree is dirty")
    settings = MPCConfig(**protocol["evaluation"]["mpc"])
    train_seeds = protocol["training"]["seeds"]
    jobs = []
    for item in protocol["evaluation"]["scenarios"]:
        scenario = Scenario(**item)
        for name in ("B0_PID", "B1_ROM_CEM", "B1b_ROM_SLSQP"):
            if group in ("all", "baseline"):
                jobs.append((scenario, name, None, None))
        for seed in train_seeds:
            for name, variant in (("B2_DATA_NO_MPC", "data_only"),
                                  ("B3_PINO_MPC", "physics_informed")):
                if group in ("all", "neural"):
                    jobs.append((scenario, name, seed,
                                 checkpoints / f"{variant}_seed{seed}.json"))
        if group in ("all", "runtime"):
            seed = protocol["evaluation"]["runtime_seed"]
            jobs.append((scenario, "B4_PINO_RUNTIME", seed,
                         checkpoints / f"physics_informed_seed{seed}.json"))
    # Validate all model artifacts before opening any test scenario.
    for _, _, _, checkpoint in jobs:
        if checkpoint is not None:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            meta = json.loads(checkpoint.read_text(encoding="utf-8"))
            weights = checkpoint.parent / meta["weights"]
            if file_sha256(weights) != meta["weights_sha256"]:
                raise RuntimeError(f"checkpoint checksum mismatch: {weights}")
    output.mkdir(parents=True, exist_ok=True)
    eval_cfg = protocol["evaluation"]
    base = ProcessConfig(**protocol["process"]).with_grid(*eval_cfg["grid"])
    summaries = []
    failures = []
    for index, (scenario, name, seed, checkpoint) in enumerate(jobs, 1):
        run_id = f"{scenario.scenario_id}_{name}" + (f"_seed{seed}" if seed is not None else "")
        summary_path = output / f"{run_id}_summary.json"
        raw_path = output / f"{run_id}_raw.csv"
        if summary_path.exists() and raw_path.exists():
            record = json.loads(summary_path.read_text(encoding="utf-8"))
            if record.get("status") == "SUCCEEDED":
                summaries.append(record)
                continue
        print(f"[{index}/{len(jobs)}] {run_id}", flush=True)
        tic = time.perf_counter()
        try:
            summary, rows = run_episode(
                base, scenario, name, steps=eval_cfg["episode_steps"],
                dt_s=eval_cfg["control_dt_s"], max_step_s=eval_cfg["max_step_s"],
                horizon=eval_cfg["mpc"]["horizon"],
                target_K=eval_cfg["reference_target_K"],
                ramp_s=eval_cfg["reference_ramp_s"],
                observation=eval_cfg["observation"], checkpoint=checkpoint,
                device=device, mpc_settings=settings,
                deadline_s=eval_cfg["runtime_deadline_s"],
                pid_gains=tuple(eval_cfg["pid_gains"]))
            write_csv(raw_path, rows)
            record = {"status": "SUCCEEDED", "run_id": run_id, "seed": seed,
                      "source_commit": git_sha(), "protocol_sha256": freeze["protocol_sha256"],
                      "wall_s": time.perf_counter() - tic,
                      "raw_sha256": file_sha256(raw_path), **summary}
            summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            summaries.append(record)
        except Exception as exc:
            record = {"status": "FAILED", "run_id": run_id,
                      "error_type": type(exc).__name__, "error": str(exc),
                      "traceback": traceback.format_exc()}
            (output / f"{run_id}_failure.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            failures.append(record)
            print(f"FAILED {run_id}: {exc}", flush=True)
    if summaries:
        write_csv(output / f"summary_{group}.csv", summaries)
    (output / f"failures_{group}.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps({"group": group, "succeeded": len(summaries),
                      "failed": len(failures), "planned": len(jobs)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--group", choices=["all", "baseline", "neural", "runtime"], default="all")
    args = parser.parse_args()
    evaluate(args.protocol, args.freeze_receipt, args.checkpoints, args.output, args.device, args.group)
