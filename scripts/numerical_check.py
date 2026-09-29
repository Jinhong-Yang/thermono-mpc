"""Analytic, conservation, and refinement checks for the synthetic plant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from thermono_mpc.process import ProcessConfig, ThermalPlant


def main(output: Path):
    results = {"description": "Numerical self-check, not physical validation",
               "grid_refinement": [], "time_refinement": []}
    for n in (16, 32, 64):
        c = ProcessConfig(nx=n, ny=n)
        plant = ThermalPlant(c)
        initial = plant.reset()
        tic = time.perf_counter()
        final = plant.advance(initial, np.array([800., 1200., 1000.]), 300., 2.)
        results["grid_refinement"].append({"grid": n, "max_step_s": 2.,
            "mean_field_K": float(final.field.mean()),
            "zone_K": final.zones.tolist(), "wall_s": time.perf_counter() - tic})
    for dt in (2., 1., .5):
        c = ProcessConfig(nx=16, ny=16)
        plant = ThermalPlant(c)
        tic = time.perf_counter()
        final = plant.advance(plant.reset(), np.array([800., 1200., 1000.]), 300., dt)
        results["time_refinement"].append({"grid": 16, "max_step_s": dt,
            "mean_field_K": float(final.field.mean()),
            "zone_K": final.zones.tolist(), "wall_s": time.perf_counter() - tic})
    results["grid_mean_delta_16_32_K"] = abs(results["grid_refinement"][0]["mean_field_K"] - results["grid_refinement"][1]["mean_field_K"])
    results["grid_mean_delta_32_64_K"] = abs(results["grid_refinement"][1]["mean_field_K"] - results["grid_refinement"][2]["mean_field_K"])
    results["time_mean_delta_2_1_K"] = abs(results["time_refinement"][0]["mean_field_K"] - results["time_refinement"][1]["mean_field_K"])
    results["time_mean_delta_1_half_K"] = abs(results["time_refinement"][1]["mean_field_K"] - results["time_refinement"][2]["mean_field_K"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
