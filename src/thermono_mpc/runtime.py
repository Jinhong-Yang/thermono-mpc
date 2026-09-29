"""Software-only command boundary; no hard real-time or hardware safety claim."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Callable
import numpy as np

from .contracts import AppliedCommand, ControlCommand, StateEstimate
from .controllers import PIDController, limit_power
from .process import ProcessConfig


def command_digest(cmd: ControlCommand) -> str:
    payload = {key: getattr(cmd, key) for key in cmd.__dataclass_fields__ if key != "setpoints_W"}
    payload["setpoints_W"] = np.asarray(cmd.setpoints_W).tolist()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SignedCommand:
    command: ControlCommand
    integrity_check: str

    @classmethod
    def create(cls, command: ControlCommand) -> SignedCommand:
        return cls(command, command_digest(command))


@dataclass(frozen=True)
class RuntimePolicy:
    protocol_version: int = 1
    clock_domain_id: str = "host-monotonic-ns"
    max_state_age_ns: int = 10_000_000_000
    deadline_ns: int = 500_000_000
    process_version: str = "synthetic-v1"


class CommandValidator:
    def __init__(self, config: ProcessConfig, policy: RuntimePolicy = RuntimePolicy()):
        self.config = config
        self.policy = policy
        self.last_sequence = -1
        self.last_cycle = -1

    def validate(self, signed: SignedCommand, state: StateEstimate,
                 previous_W: np.ndarray, *, now_ns: int, current_cycle: int,
                 expected_generation: int, finish_ns: int) -> str:
        p, c, cmd = self.policy, self.config, signed.command
        try:
            state.validate(c.ny, c.nx, c.zones)
            if not state.valid:
                return "INVALID_STATE"
            setpoints = np.asarray(cmd.setpoints_W, dtype=float)
            if setpoints.shape != (c.zones,) or not np.all(np.isfinite(setpoints)):
                return "INVALID_SHAPE_OR_NONFINITE"
            if signed.integrity_check != command_digest(cmd):
                return "INTEGRITY_MISMATCH"
        except (ValueError, TypeError, OverflowError):
            return "INVALID_SHAPE_OR_NONFINITE"
        if cmd.protocol_version != p.protocol_version:
            return "PROTOCOL_MISMATCH"
        if cmd.clock_domain_id != p.clock_domain_id or state.clock_domain_id != p.clock_domain_id:
            return "CLOCK_DOMAIN_MISMATCH"
        if cmd.units != "W":
            return "UNITS_MISMATCH"
        if cmd.parameter_version != p.process_version or state.parameter_version != p.process_version:
            return "PROCESS_VERSION_MISMATCH"
        if cmd.generation_id != expected_generation:
            return "STALE_GENERATION"
        if cmd.source_state_id != state.state_id or cmd.source_sample_time_ns != state.sample_time_ns:
            return "SOURCE_MISMATCH"
        if now_ns < state.sample_time_ns or now_ns - state.sample_time_ns > p.max_state_age_ns:
            return "STALE_SENSOR"
        if cmd.sequence_id <= self.last_sequence or cmd.apply_cycle <= self.last_cycle:
            return "OUT_OF_ORDER_OR_DUPLICATE"
        if cmd.apply_cycle != current_cycle + 1:
            return "APPLY_CYCLE_MISMATCH"
        if now_ns > cmd.expires_at_ns:
            return "EXPIRED_COMMAND"
        if finish_ns > state.sample_time_ns + p.deadline_ns:
            return "DEADLINE_OVERRUN"
        previous = np.asarray(previous_W, dtype=float)
        if setpoints.shape != (c.zones,) or previous.shape != (c.zones,) or not np.all(np.isfinite(setpoints)):
            return "INVALID_SHAPE_OR_NONFINITE"
        if np.any(setpoints < 0) or np.any(setpoints > c.pmax):
            return "POWER_BOUNDS"
        if np.any(np.abs(setpoints - previous) > np.asarray(c.slew) + 1e-9):
            return "SLEW_BOUNDS"
        self.last_sequence = cmd.sequence_id
        self.last_cycle = cmd.apply_cycle
        return "ACCEPTED"


class RuntimeSupervisor:
    """Single-worker computation; a late result is invalidated by generation.

    A timed-out GPU call may continue running. A full worker falls back to an
    independent PID controller rather than accumulating an unbounded queue.
    """

    def __init__(self, config: ProcessConfig, policy: RuntimePolicy = RuntimePolicy(),
                 fallback: PIDController | None = None):
        self.config = config
        self.policy = policy
        self.validator = CommandValidator(config, policy)
        self.fallback = fallback or PIDController(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thermono-predictor")
        self.inflight: Future | None = None
        self.generation = 0
        self.sequence = 0
        self.events: list[dict] = []

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, compute: Callable[[], np.ndarray]) -> int | None:
        if self.inflight is not None and not self.inflight.done():
            self.events.append({"event": "QUEUE_FULL", "generation": self.generation})
            return None
        self.generation += 1
        self.inflight = self.executor.submit(compute)
        return self.generation

    def select(self, state: StateEstimate, previous_W: np.ndarray,
               reference_K: float, *, current_cycle: int,
               request_generation: int | None, start_ns: int | None = None,
               now_ns: int | None = None, finish_ns: int | None = None) -> AppliedCommand:
        now = time.perf_counter_ns() if now_ns is None else now_ns
        start = now if start_ns is None else start_ns
        reason = "NO_RESULT"
        signed = None
        if request_generation is None:
            reason = "QUEUE_FULL"
        elif request_generation != self.generation:
            reason = "STALE_GENERATION"
        elif self.inflight is not None and self.inflight.done():
            try:
                proposed = np.asarray(self.inflight.result(), dtype=float)
                completed = now if finish_ns is None else finish_ns
                self.sequence += 1
                command = ControlCommand(
                    self.policy.protocol_version, self.sequence, state.state_id,
                    state.sample_time_ns, self.policy.clock_domain_id,
                    current_cycle + 1, start + self.policy.max_state_age_ns,
                    proposed, "candidate", self.policy.process_version,
                    request_generation)
                signed = SignedCommand.create(command)
                reason = self.validator.validate(signed, state, previous_W,
                                                 now_ns=now, current_cycle=current_cycle,
                                                 expected_generation=self.generation,
                                                 finish_ns=completed)
            except Exception as exc:
                reason = "WORKER_FAILED:" + type(exc).__name__
        elif now > start + self.policy.deadline_ns:
            reason = "DEADLINE_OVERRUN"
            self.generation += 1
        if reason == "ACCEPTED" and signed is not None:
            applied = AppliedCommand(signed.command.sequence_id, current_cycle + 1,
                                     signed.command.setpoints_W.copy(), "CANDIDATE", reason, now)
        else:
            try:
                safe = self.fallback.propose(state.field_K, reference_K, previous_W, 1.)
                safe = limit_power(safe, previous_W, self.config)
            except Exception:
                safe = limit_power(np.zeros(self.config.zones), previous_W, self.config)
                reason += "+FALLBACK_ERROR"
            applied = AppliedCommand(self.sequence, current_cycle + 1, safe,
                                     "FALLBACK", reason, now)
        self.events.append({"event": "APPLIED", "cycle": applied.apply_cycle,
                            "mode": applied.mode, "reason": applied.reason,
                            "generation": request_generation})
        return applied
