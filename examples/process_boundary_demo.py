"""Show a compact command crossing a real local process boundary on CPU."""
import json
import time

import numpy as np

from thermono_mpc.contracts import ControlCommand, ControlObservation, StateEstimate
from thermono_mpc.process import ProcessConfig
from thermono_mpc.process_boundary import ProcessCommandBoundary
from thermono_mpc.runtime import SignedCommand


def main() -> None:
    config = ProcessConfig(nx=8, ny=8)
    with ProcessCommandBoundary(config) as boundary:
        sample_ns = time.perf_counter_ns()
        estimate = StateEstimate(np.full((8, 8), 300.), np.full(3, 300.),
                                 sample_ns, 1, "host-monotonic-ns", "synthetic-v1")
        compact = ControlObservation.from_estimate(estimate, config.zones)
        compute_start_ns = time.perf_counter_ns()
        command = ControlCommand(1, 1, compact.state_id, compact.sample_time_ns,
                                 compact.clock_domain_id, 1, sample_ns + 10_000_000_000,
                                 np.full(config.zones, 200.), "example", "synthetic-v1", 1)
        applied = boundary.select(compact, np.zeros(config.zones), 320.,
                                  current_cycle=0, expected_generation=1,
                                  compute_start_ns=compute_start_ns,
                                  finish_ns=time.perf_counter_ns(),
                                  signed_command=SignedCommand.create(command), dt_s=1.)
        print(json.dumps({"mode": applied.mode, "reason": applied.reason,
                          "power_W": applied.actual_power_W.tolist(),
                          "event": boundary.events[-1]}, indent=2))


if __name__ == "__main__":
    main()
