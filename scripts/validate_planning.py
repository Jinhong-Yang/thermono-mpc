"""Validate the bundled planning defaults as design input, not a runnable job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from thermono_mpc.process import ProcessConfig


def validate(path: Path) -> dict:
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    if p.get("schema_version") != "planning-1.0" or p.get("status") != "PLANNING_ONLY_NOT_EXECUTABLE_UNTIL_SCHEMA_IMPLEMENTED":
        raise ValueError("not the expected planning-default input")
    process = p["process"]
    g, m, e, z, constraints = (process[key] for key in
                               ("geometry", "material", "thermal_exchange", "heating_zones", "constraints"))
    c = ProcessConfig(nx=process["numerics"]["generator_grid"][1],
                      ny=process["numerics"]["generator_grid"][0],
                      lx=g["length_x_m"], ly=g["length_y_m"],
                      thickness=g["effective_thickness_m"],
                      rho=m["rho_kg_per_m3"], cp=m["cp_J_per_kgK"],
                      k=m["k_W_per_mK"],
                      h=tuple(e["h_W_per_m2K"]),
                      emissivity=tuple(e["emissivity"]),
                      zone_capacity=tuple(z["capacity_J_per_K"]),
                      zone_efficiency=tuple(z["power_efficiency"]),
                      zone_loss=tuple(z["loss_W_per_K"]),
                      zone_coupling=tuple(tuple(row) for row in z["coupling_W_per_K"]),
                      pmax=tuple(z["power_max_W"]),
                      slew=tuple(z["power_slew_max_W_per_update"]),
                      ambient=process["ambient_temperature_K"],
                      initial=process["initial_temperature_K"],
                      solid_limit=constraints["solid_temperature_upper_K"],
                      zone_limit=constraints["zone_temperature_upper_K"])
    if g["zones"] != c.zones or process["units"]["temperature_internal"] != "K":
        raise ValueError("zone/unit contract mismatch")
    if p["operator"]["forecast_horizon_steps"] != p["mpc"]["horizon_steps"]:
        raise ValueError("model/controller horizon mismatch")
    return {"schema": p["schema_version"], "status": "VALID_DESIGN_INPUT_NOT_EXECUTABLE",
            "generator_grid": [c.ny, c.nx], "zones": c.zones,
            "reference_reachability": "FAILED_PILOT_AT_600S; see pilot_reachability.json",
            "freeze_status": p["freeze"]["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.path), indent=2))
