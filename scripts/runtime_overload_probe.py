"""Measure a CPU-contended software deadline; no arbitrary sleep injection."""
import json
import time

import numpy as np

from thermono_mpc.contracts import StateEstimate
from thermono_mpc.process import ProcessConfig
from thermono_mpc.runtime import RuntimePolicy, RuntimeSupervisor


def _cpu_work() -> np.ndarray:
    checksum = sum(i * i for i in range(5_000_000))
    if checksum <= 0:
        raise AssertionError("CPU workload did not run")
    return np.full(3, 200.)


def main() -> None:
    config = ProcessConfig(nx=8, ny=8)
    deadline_ns = 10_000_000
    policy = RuntimePolicy(deadline_ns=deadline_ns)
    supervisor = RuntimeSupervisor(config, policy)
    try:
        sample_ns = time.perf_counter_ns()
        state = StateEstimate(np.full((8, 8), 300.), np.full(3, 300.),
                              sample_ns, 1, policy.clock_domain_id, policy.process_version)
        generation = supervisor.submit(_cpu_work)
        second_generation = supervisor.submit(_cpu_work)
        checksum = 0
        while time.perf_counter_ns() - sample_ns <= 2 * deadline_ns:
            checksum += sum(i * i for i in range(100_000))
        decision_ns = time.perf_counter_ns()
        applied = supervisor.select(state, np.zeros(3), 320., current_cycle=0,
                                    request_generation=generation,
                                    start_ns=sample_ns, now_ns=decision_ns)
        supervisor.inflight.result(timeout=10.)
        receipt = {
            "workload_kind": "MEASURED_OVERLOAD",
            "clock": "host monotonic perf_counter_ns",
            "deadline_ns": deadline_ns,
            "decision_elapsed_ns": decision_ns - sample_ns,
            "second_job_rejected_by_bounded_queue": second_generation is None,
            "main_cpu_checksum_positive": checksum > 0,
            "applied_mode": applied.mode,
            "reason": applied.reason,
            "candidate_applied": applied.mode == "CANDIDATE",
            "scope": "CPU contention on this host only; no GPU preemption or worst-case timing claim",
        }
        if (receipt["decision_elapsed_ns"] <= deadline_ns
                or not receipt["second_job_rejected_by_bounded_queue"]
                or receipt["candidate_applied"]):
            raise AssertionError(receipt)
        print(json.dumps(receipt, indent=2))
    finally:
        supervisor.close()


if __name__ == "__main__":
    main()
