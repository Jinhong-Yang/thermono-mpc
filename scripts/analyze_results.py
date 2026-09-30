"""Frozen analysis: scenario-level paired summaries with explicit failures."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def interval(values, seed=12345):
    a = np.asarray(values, dtype=float)
    if len(a) < 4:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(a, size=(10000, len(a)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def main(control_dir: Path, prediction_csv: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in control_dir.glob("*_summary.json")]
    failures = [json.loads(p.read_text(encoding="utf-8"))
                for p in control_dir.glob("*_failure.json")]
    metrics = ["rmse_K", "mean_spatial_std_K", "energy_J", "solid_excess_Ks",
               "zone_excess_Ks", "fallback_rate", "deadline_miss_rate",
               "latency_p50_s", "latency_p95_s", "latency_p99_s", "latency_max_s"]
    groups = defaultdict(list)
    for record in records:
        for metric in metrics:
            groups[(record["scenario_id"], record["category"],
                    record["controller"], metric)].append(float(record[metric]))
    rows = []
    for (scenario, category, controller, metric), values in sorted(groups.items()):
        rows.append({"scenario_id": scenario, "category": category,
                     "controller": controller, "metric": metric,
                     "seed_mean": float(np.mean(values)),
                     "seed_min": float(np.min(values)),
                     "seed_max": float(np.max(values)),
                     "model_seed_count": len(values)})
    write_csv(output / "scenario_metrics.csv", rows)
    aggregate = []
    for category in ("all", "nominal", "parameter_shift", "sensor_stress"):
        for controller in sorted(set(r["controller"] for r in rows)):
            for metric in metrics:
                selection = [r["seed_mean"] for r in rows if r["controller"] == controller
                             and r["metric"] == metric and
                             (category == "all" or r["category"] == category)]
                if selection:
                    low, high = interval(selection)
                    aggregate.append({"category": category, "controller": controller,
                                      "metric": metric, "n_scenarios": len(selection),
                                      "mean": float(np.mean(selection)),
                                      "ci95_low": low, "ci95_high": high,
                                      "observed_min": float(np.min(selection)),
                                      "observed_max": float(np.max(selection))})
    write_csv(output / "control_aggregate.csv", aggregate)
    paired = []
    comparisons = [("B3_PINO_MPC", "B2_DATA_NO_MPC"),
                   ("B3_PINO_MPC", "B1_ROM_CEM"),
                   ("B3_PINO_MPC", "B0_PID"),
                   ("B1_ROM_CEM", "B1b_ROM_SLSQP")]
    lookup = {(r["scenario_id"], r["controller"], r["metric"]): r
              for r in rows}
    for category in ("all", "nominal", "parameter_shift", "sensor_stress"):
        scenarios = sorted({r["scenario_id"] for r in rows
                            if category == "all" or r["category"] == category})
        for left, right in comparisons:
            for metric in metrics:
                differences = []
                for scenario in scenarios:
                    a = lookup.get((scenario, left, metric))
                    b = lookup.get((scenario, right, metric))
                    if a and b:
                        differences.append(a["seed_mean"] - b["seed_mean"])
                if differences:
                    low, high = interval(differences)
                    paired.append({"category": category, "left": left, "right": right,
                                   "metric": metric, "n_paired_scenarios": len(differences),
                                   "mean_left_minus_right": float(np.mean(differences)),
                                   "ci95_low": low, "ci95_high": high,
                                   "observed_min": float(np.min(differences)),
                                   "observed_max": float(np.max(differences))})
    write_csv(output / "control_paired.csv", paired)
    pred_rows = read_csv(prediction_csv)
    windows = defaultdict(list)
    for row in pred_rows:
        key = (int(row["trajectory_seed"]), row["variant"], row["model_seed"])
        windows[key].append(row)
    trajectory_prediction = []
    for (seed, variant, model_seed), items in sorted(windows.items()):
        trajectory_prediction.append({"trajectory_seed": seed, "variant": variant,
            "model_seed": model_seed,
            "field_rmse_K": float(np.mean([float(x["field_rmse_K"]) for x in items])),
            "zone_rmse_K": float(np.mean([float(x["zone_rmse_K"]) for x in items])),
            "physics_residual_normalized": (float(np.mean([float(x["physics_residual_normalized"])
                                              for x in items])) if items[0]["physics_residual_normalized"] else None),
            "window_count": len(items)})
    write_csv(output / "prediction_trajectory_metrics.csv", trajectory_prediction)
    pred_scenario = defaultdict(list)
    for row in trajectory_prediction:
        for metric in ("field_rmse_K", "zone_rmse_K", "physics_residual_normalized"):
            if row[metric] is not None:
                pred_scenario[(row["trajectory_seed"], row["variant"], metric)].append(row[metric])
    pred_aggregate = []
    for variant in ("ROM", "data_only", "physics_informed"):
        for metric in ("field_rmse_K", "zone_rmse_K", "physics_residual_normalized"):
            values = [float(np.mean(v)) for (seed, name, key), v in pred_scenario.items()
                      if name == variant and key == metric]
            if values:
                low, high = interval(values)
                pred_aggregate.append({"variant": variant, "metric": metric,
                                       "n_trajectories": len(values), "mean": float(np.mean(values)),
                                       "ci95_low": low, "ci95_high": high,
                                       "observed_min": float(np.min(values)),
                                       "observed_max": float(np.max(values))})
    write_csv(output / "prediction_aggregate.csv", pred_aggregate)
    expected_control_runs = 12 * (3 + 2 * 3 + 1)
    report = {"control_success_count": len(records), "control_failure_count": len(failures),
              "control_expected_runs": expected_control_runs,
              "prediction_window_rows": len(pred_rows),
              "prediction_trajectory_seed_count": len(set(x["trajectory_seed"] for x in trajectory_prediction)),
              "status": ("COMPLETE" if len(records) == expected_control_runs and not failures else "INCOMPLETE"),
              "notes": "Bootstrap resamples whole scenarios or trajectories, not time samples or overlapping windows. Three training seeds are averaged within scenario and their range is retained."}
    (output / "analysis_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.control_dir, args.prediction_csv, args.output)
