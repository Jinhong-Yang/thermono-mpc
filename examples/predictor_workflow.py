"""CPU model -> forecast -> MPC -> generic replay/comparison tutorial.

Run after installing the package:
    python examples/predictor_workflow.py --output outputs/predictor_workflow
Only the LinearThermalPredictor adapter is model-specific. The simulator,
optimizer, episode evaluator, objective and diagnostic functions are reused.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from thermono_mpc.controllers import CEMMPC, MPCConfig, ROMPredictor, limit_power
from thermono_mpc.diagnostics import common_state_comparison, plan_objective, same_plan_replay, thermal_replay
from thermono_mpc.linear_predictor import LinearThermalPredictor
from thermono_mpc.process import ProcessConfig, ThermalPlant
from thermono_mpc.simulation import Scenario, run_episode


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def run_example(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = ProcessConfig(nx=8, ny=8)
    settings = MPCConfig(horizon=4, blocks=2, candidates=16, iterations=2, elite=4, energy_weight=0.0001)
    predictor = LinearThermalPredictor(c)
    plant = ThermalPlant(c)
    initial = plant.reset()
    powers, held = [], np.zeros(c.zones)
    for _ in range(settings.horizon):
        held = limit_power(np.full(c.zones, 1200.), held, c)
        powers.append(held.copy())
    plan = np.asarray(powers)[None]
    forecast = same_plan_replay(predictor, initial.field, initial.zones, plan, 10., thermal_replay(plant, initial))
    np.savez_compressed(output/"forecast.npz", plan_W=plan,
                        predicted_K=forecast["predicted"].temperature_K,
                        realized_K=forecast["realized"].temperature_K)
    diagnostic_rows = []
    def inspect(event):
        if event["cycle"] % 10:
            return
        if event["result"] is None or event["result"].status != "OK":
            diagnostic_rows.append({"cycle": event["cycle"], "status": "NO_PLAN",
                                    "own_predicted_cost": None, "rom_predicted_cost": None,
                                    "own_realized_cost": None, "rom_realized_cost": None,
                                    "own_endpoint_bias_K": None})
            return
        field, zones = event["projected_field_K"], event["projected_zone_K"]
        previous, ref = event["held_power_W"], event["reference_K"]
        alternate = CEMMPC(c, ROMPredictor(c), settings, seed=3000+event["cycle"])
        rom_plan = alternate.optimize(field, zones, ref, previous, 10.).sequence_W
        plans = np.stack([event["result"].sequence_W, rom_plan])
        replay = thermal_replay(ThermalPlant(event["actual_config"]), event["state_after_held"])
        result = common_state_comparison(predictor, field, zones, plans, 10., replay,
                   lambda fields, actions: plan_objective(fields, actions, ref, previous, settings, 10.))
        diagnostic_rows.append({"cycle": event["cycle"], "status": "OK",
             "own_predicted_cost": float(result["predicted_cost"][0]),
             "rom_predicted_cost": float(result["predicted_cost"][1]),
             "own_realized_cost": float(result["realized_cost"][0]),
             "rom_realized_cost": float(result["realized_cost"][1]),
             "own_endpoint_bias_K": float(result["endpoint_bias_K"][0])})
        np.savez_compressed(output/f"diagnostic_cycle_{event['cycle']:03}.npz", plans_W=plans,
                            predicted_K=result["predicted"].temperature_K,
                            realized_K=result["realized"].temperature_K,
                            projected_field_K=field, projected_zones_K=zones,
                            held_power_W=previous, reference_K=ref,
                            aligned_true_field_K=event["state_after_held"].field)
    summary, rows = run_episode(c, Scenario("linear_tutorial", 731), "CUSTOM_CEM",
                 predictor=predictor, steps=30, dt_s=10., max_step_s=2., horizon=settings.horizon,
                 target_K=320., ramp_s=180., mpc_settings=settings, diagnostic_hook=inspect,
                 pid_gains=(80., .02))
    write_csv(output/"closed_loop.csv", rows)
    write_csv(output/"diagnostics.csv", diagnostic_rows)
    result = {"scope": "CPU extension demonstration; no model-performance superiority claim",
              "forecast_rmse_K": float(forecast["field_rmse_K"][0]), "closed_loop": summary,
              "diagnostic_calls": len(diagnostic_rows), "successful_diagnostic_calls": sum(x["status"] == "OK" for x in diagnostic_rows),
              "modified_component": "LinearThermalPredictor",
              "reused_components": ["CEMMPC", "run_episode", "plan_objective", "same_plan_replay", "common_state_comparison", "thermal_replay"]}
    (output/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictor_workflow"))
    print(json.dumps(run_example(parser.parse_args().output), indent=2))
