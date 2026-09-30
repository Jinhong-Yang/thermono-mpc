"""Optional process-separated, compact software command boundary.

This local IPC example is not a PLC, hardware isolation, or a real-time guarantee.
Only zone means and command metadata cross the control-process transport.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
import time

import numpy as np

from .contracts import AppliedCommand, ControlObservation
from .controllers import limit_power
from .process import ProcessConfig
from .runtime import CommandValidator, RuntimePolicy, SignedCommand


@dataclass(frozen=True)
class BoundaryRequest:
    observation: ControlObservation
    previous_W: np.ndarray
    reference_K: float
    current_cycle: int
    expected_generation: int
    compute_start_ns: int
    finish_ns: int
    signed_command: SignedCommand | None
    dt_s: float
    enqueue_ns: int


def _fallback(observation: ControlObservation, previous_W: np.ndarray,
              reference_K: float, dt_s: float, config: ProcessConfig,
              integral: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous = np.asarray(previous_W, dtype=float)
    if previous.shape != (config.zones,) or not np.all(np.isfinite(previous)):
        raise ValueError("invalid previous actuator state")
    means = np.asarray(observation.zone_mean_K, dtype=float)
    if (not observation.valid or means.shape != (config.zones,)
            or not np.all(np.isfinite(means)) or not np.isfinite(reference_K)
            or not np.isfinite(dt_s) or dt_s <= 0):
        return limit_power(np.zeros(config.zones), previous, config), integral
    error = reference_K - means
    next_integral = np.clip(integral + dt_s * error, -20000., 20000.)
    requested = 15. * error + .01 * next_integral
    return limit_power(requested, previous, config), next_integral


def _selector_process(conn: Connection, config: ProcessConfig, policy: RuntimePolicy) -> None:
    validator = CommandValidator(config, policy)
    integral = np.zeros(config.zones)
    try:
        conn.send({"status": "READY"})
        while True:
            request = conn.recv()
            if request is None:
                break
            now_ns = time.perf_counter_ns()
            reason = "NO_RESULT"
            signed = request.signed_command
            if not (request.observation.sample_time_ns <= request.compute_start_ns
                    <= request.finish_ns <= now_ns):
                reason = "INVALID_TIMESTAMPS"
            elif signed is not None:
                reason = validator.validate(
                    signed, request.observation, request.previous_W,
                    now_ns=now_ns, current_cycle=request.current_cycle,
                    expected_generation=request.expected_generation,
                    finish_ns=request.finish_ns,
                )
            if reason == "ACCEPTED":
                applied = AppliedCommand(signed.command.sequence_id, request.current_cycle + 1,
                                         np.asarray(signed.command.setpoints_W).copy(),
                                         "CANDIDATE", reason, now_ns)
            else:
                try:
                    power, integral = _fallback(request.observation, request.previous_W,
                                                request.reference_K, request.dt_s,
                                                config, integral)
                except Exception:
                    previous = np.asarray(request.previous_W, dtype=float)
                    if previous.shape != (config.zones,) or not np.all(np.isfinite(previous)):
                        previous = np.zeros(config.zones)
                    power = limit_power(np.zeros(config.zones), previous, config)
                    reason += "+FALLBACK_ERROR"
                applied = AppliedCommand(validator.last_sequence, request.current_cycle + 1,
                                         power, "FALLBACK", reason, now_ns)
            conn.send((applied, {"sensor_ns": request.observation.sample_time_ns,
                                 "compute_start_ns": request.compute_start_ns,
                                 "compute_finish_ns": request.finish_ns,
                                 "enqueue_ns": request.enqueue_ns,
                                 "validated_ns": now_ns,
                                 "selected_ns": applied.applied_at_ns,
                                 "mode": applied.mode, "reason": applied.reason}))
    except EOFError:
        pass
    finally:
        conn.close()


class ProcessCommandBoundary:
    """One-request-at-a-time IPC selector with independent compact PID fallback."""

    def __init__(self, config: ProcessConfig, policy: RuntimePolicy = RuntimePolicy()):
        self.config = config
        self.policy = policy
        context = get_context("spawn")
        self._parent, child = context.Pipe(duplex=True)
        self._process = context.Process(target=_selector_process, args=(child, config, policy),
                                        daemon=True, name="thermono-command-selector")
        self._process.start()
        child.close()
        self.events: list[dict] = []
        self._closed = False
        if not self._parent.poll(10.) or self._parent.recv() != {"status": "READY"}:
            self.close()
            raise RuntimeError("command selector did not start")

    @property
    def alive(self) -> bool:
        return not self._closed and self._process.is_alive()

    def select(self, observation: ControlObservation, previous_W: np.ndarray,
               reference_K: float, *, current_cycle: int, expected_generation: int,
               compute_start_ns: int, finish_ns: int, signed_command: SignedCommand | None,
               dt_s: float, timeout_s: float | None = None) -> AppliedCommand:
        if self._closed:
            raise RuntimeError("command boundary is closed")
        start_ns = time.perf_counter_ns()
        request = BoundaryRequest(observation, np.asarray(previous_W, dtype=float).copy(),
                                  reference_K, current_cycle, expected_generation,
                                  compute_start_ns, finish_ns, signed_command, dt_s, start_ns)
        reason = "CONTROL_PROCESS_UNAVAILABLE"
        if self._process.is_alive():
            try:
                self._parent.send(request)
                wait_s = self.policy.deadline_ns / 1e9 if timeout_s is None else timeout_s
                if self._parent.poll(max(0., wait_s)):
                    applied, event = self._parent.recv()
                    self.events.append(event)
                    return applied
                reason = "CONTROL_PROCESS_TIMEOUT"
            except (BrokenPipeError, EOFError, OSError):
                reason = "CONTROL_PROCESS_UNAVAILABLE"
        self.close()
        try:
            power, _ = _fallback(observation, request.previous_W, reference_K,
                                 dt_s, self.config, np.zeros(self.config.zones))
        except Exception:
            power = np.zeros(self.config.zones)
            reason += "+FALLBACK_ERROR"
        applied = AppliedCommand(-1, current_cycle + 1, power, "FALLBACK",
                                 reason, time.perf_counter_ns())
        self.events.append({"sensor_ns": observation.sample_time_ns,
                            "compute_start_ns": compute_start_ns,
                            "compute_finish_ns": finish_ns,
                            "enqueue_ns": start_ns,
                            "validated_ns": None,
                            "selected_ns": applied.applied_at_ns,
                            "mode": applied.mode, "reason": applied.reason})
        return applied

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._parent.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.)
        self._parent.close()

    def __enter__(self) -> ProcessCommandBoundary:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
