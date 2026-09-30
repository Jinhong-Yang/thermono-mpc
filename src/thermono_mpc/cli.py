"""Small public command-line demonstrations."""
from __future__ import annotations

import argparse
import json

from .process import ProcessConfig
from .simulation import Scenario, run_episode


def main() -> None:
    parser = argparse.ArgumentParser(prog="thermono-mpc")
    parser.add_argument("command", choices=["doctor", "demo"])
    parser.add_argument("--profile", choices=["cpu-smoke"],
                        help="Demo profile; currently only cpu-smoke is available")
    args = parser.parse_args()
    if args.command != "demo" and args.profile is not None:
        parser.error("--profile is only valid with demo")
    if args.command == "doctor":
        import numpy, scipy
        print(json.dumps({"numpy": numpy.__version__, "scipy": scipy.__version__,
                          "process": "synthetic", "live_equipment": "unsupported"}, indent=2))
    else:
        c = ProcessConfig(nx=8, ny=8)
        summary, _ = run_episode(c, Scenario("cpu-demo", 42), "B0_PID",
                                 steps=6, dt_s=2., max_step_s=1.,
                                 horizon=2, target_K=300., ramp_s=12.)
        print(json.dumps(summary, indent=2))
