"""A second synthetic geometry/material/zone configuration.

The process and generic controller core run without edits. A trained FNO
checkpoint from the three-zone case is not reused for this four-zone case.
"""
import json
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import Scenario, run_episode


if __name__ == "__main__":
    process = ProcessConfig(
        nx=10, ny=6, lx=0.08, ly=0.05, thickness=0.015,
        rho=7700., cp=520., k=35.,
        h=(22., 24., 26., 28.), emissivity=(.65, .65, .7, .7),
        zone_capacity=(1800., 1800., 1800., 1800.),
        zone_efficiency=(.8, .8, .8, .8),
        zone_loss=(2., 2., 2., 2.),
        zone_coupling=((0., 0., 0., 0.),) * 4,
        pmax=(1500., 1500., 1500., 1500.),
        slew=(300., 300., 300., 300.))
    for controller in ("B0_PID", "B1_ROM_CEM"):
        summary, _ = run_episode(process, Scenario("reuse-four-zone", 99),
                                 controller, steps=6, dt_s=2., max_step_s=1.,
                                 horizon=2, target_K=300., ramp_s=12.)
        print(json.dumps(summary, indent=2))
