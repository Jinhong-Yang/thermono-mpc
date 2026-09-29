"""Record a read-only freeze receipt outside the public repository."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import yaml

from thermono_mpc.dataset import file_sha256


def digest_object(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze(protocol_path: Path, manifest_path: Path, output: Path) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN":
        raise ValueError("protocol status must be FROZEN")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["horizon"] != protocol["data"]["horizon"] or manifest["steps"] != protocol["data"]["steps"]:
        raise ValueError("data manifest does not match protocol")
    splits = {}
    for item in manifest["trajectories"]:
        if item["split"] == "test":
            raise ValueError("test trajectories cannot be accessed before freeze")
        path = manifest_path.parent / item["file"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"data file hash mismatch: {path}")
        splits.setdefault(item["split"], []).append(item["id"])
    expected = {key: protocol["data"][key + "_trajectories"]
                for key in ("train", "validation", "calibration")}
    if {key: len(splits.get(key, [])) for key in expected} != expected:
        raise ValueError("split counts differ from protocol")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("commit all source and protocol changes before freeze")
    scenarios = protocol["evaluation"]["scenarios"]
    if len({x["scenario_id"] for x in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario ID")
    receipt = {
        "schema": "thermono-freeze-1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "data_manifest_path": str(manifest_path),
        "data_manifest_sha256": file_sha256(manifest_path),
        "split_ids": splits,
        "split_sha256": digest_object(splits),
        "model_config_sha256": digest_object(protocol["training"]),
        "controller_config_sha256": digest_object(protocol["evaluation"]["mpc"]),
        "observer_config_sha256": digest_object({"mode": protocol["evaluation"]["observation"],
                                                    "algorithm": "SparseObserver default fixed probes"}),
        "scenario_sha256": digest_object(scenarios),
        "analysis_plan_sha256": digest_object(protocol["analysis"]),
        "runtime_budget_sha256": digest_object({"deadline_s": protocol["evaluation"]["runtime_deadline_s"],
                                                  "control_dt_s": protocol["evaluation"]["control_dt_s"]}),
        "test_access_status": "NOT_ACCESSED_AT_FREEZE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("existing freeze receipt is immutable")
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.protocol, args.manifest, args.output), indent=2))
