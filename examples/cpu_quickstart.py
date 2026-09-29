"""Runs without PyTorch; outputs measured synthetic demonstration metrics."""
import json
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import Scenario, run_episode


if __name__ == "__main__":
    process = ProcessConfig(nx=8, ny=8)
    scenario = Scenario("example", 42)
    for name in ("B0_PID", "B1_ROM_CEM"):
        summary, _ = run_episode(process, scenario, name, steps=6,
                                 dt_s=2., max_step_s=1., horizon=2,
                                 target_K=300., ramp_s=12.)
        print(json.dumps(summary, indent=2))
