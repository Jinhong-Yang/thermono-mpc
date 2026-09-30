from dataclasses import replace
import threading
import numpy as np
import pytest

from thermono_mpc.process import ProcessConfig
from thermono_mpc.controllers import CEMMPC, SLSQPMPC, MPCConfig, ROMPredictor, PIDController, limit_power
from thermono_mpc.contracts import ControlCommand, StateEstimate
from thermono_mpc.runtime import CommandValidator, RuntimePolicy, SignedCommand, RuntimeSupervisor


def setup():
    c = ProcessConfig(nx=8, ny=8)
    state = StateEstimate(np.full((8, 8), 350.), np.full(3, 400.), 1_000_000_000,
                          4, "host-monotonic-ns", "synthetic-v1")
    return c, state


def test_rom_and_cem_react_to_candidate_power():
    c, state = setup()
    rom = ROMPredictor(c)
    u = np.stack([np.zeros((4, 3)), np.full((4, 3), 1000.)])
    prediction = rom.predict(state.field_K, state.zones_K, u, 10.)
    prediction.validate(2, 4, 8, 8, 3)
    assert prediction.zone_temperature_K[1, -1].mean() > prediction.zone_temperature_K[0, -1].mean()
    mpc = CEMMPC(c, rom, MPCConfig(horizon=4, blocks=2, candidates=8, iterations=2, elite=2), seed=1)
    result = mpc.optimize(state.field_K, state.zones_K, np.full(4, 380.), np.zeros(3), 10.)
    assert result.status == "OK" and np.all(result.power_W <= c.slew)
    assert result.feasible_count > 0
    qp = SLSQPMPC(c, MPCConfig(horizon=4, blocks=2, candidates=8, iterations=2, elite=2))
    qresult = qp.optimize(state.field_K, state.zones_K, np.full(4, 380.), np.zeros(3), 10.)
    assert qresult.status == "OK" and np.all(qresult.power_W <= np.asarray(c.slew) + 1e-5)


def test_command_validation_faults():
    c, state = setup()
    policy = RuntimePolicy()
    cmd = ControlCommand(1, 1, state.state_id, state.sample_time_ns,
                         state.clock_domain_id, 1, 1_500_000_000,
                         np.full(3, 200.), "rom-1", "synthetic-v1", 1)
    def check(command=cmd, **kwargs):
        v = CommandValidator(c, policy)
        return v.validate(SignedCommand.create(command), state, np.zeros(3),
                          now_ns=1_100_000_000, finish_ns=1_100_000_000,
                          current_cycle=0, expected_generation=1, **kwargs)
    assert check() == "ACCEPTED"
    assert check(replace(cmd, units="kW")) == "UNITS_MISMATCH"
    assert check(replace(cmd, clock_domain_id="utc")) == "CLOCK_DOMAIN_MISMATCH"
    invalid = replace(cmd, setpoints_W=np.array([np.nan, 0., 0.]))
    assert CommandValidator(c, policy).validate(SignedCommand(invalid, "invalid"), state, np.zeros(3),
                now_ns=1_100_000_000, finish_ns=1_100_000_000,
                current_cycle=0, expected_generation=1) == "INVALID_SHAPE_OR_NONFINITE"
    assert check(replace(cmd, setpoints_W=np.full(3, 500.))) == "SLEW_BOUNDS"
    assert check(replace(cmd, generation_id=0)) == "STALE_GENERATION"
    assert check(replace(cmd, apply_cycle=2)) == "APPLY_CYCLE_MISMATCH"
    v = CommandValidator(c)
    signed = SignedCommand.create(cmd)
    assert v.validate(signed, state, np.zeros(3), now_ns=1_100_000_000, finish_ns=1_100_000_000,
                      current_cycle=0, expected_generation=1) == "ACCEPTED"
    assert v.validate(signed, state, np.zeros(3), now_ns=1_100_000_000, finish_ns=1_100_000_000,
                      current_cycle=0, expected_generation=1) == "OUT_OF_ORDER_OR_DUPLICATE"
    v2 = CommandValidator(c)
    assert v2.validate(signed, state, np.zeros(3), now_ns=1_100_000_000, finish_ns=1_600_000_000,
                       current_cycle=0, expected_generation=1) == "DEADLINE_OVERRUN"


def test_supervisor_fallback_on_worker_error():
    c, state = setup()
    sup = RuntimeSupervisor(c, fallback=PIDController(c))
    try:
        def fail():
            raise RuntimeError("injected worker failure")
        gen = sup.submit(fail)
        import time
        for _ in range(100):
            if sup.inflight.done():
                break
            time.sleep(.001)
        applied = sup.select(state, np.zeros(3), 370., current_cycle=0,
                             request_generation=gen, start_ns=1_000_000_000,
                             now_ns=1_100_000_000)
        assert applied.mode == "FALLBACK" and "WORKER_FAILED" in applied.reason
        assert np.all(applied.actual_power_W <= c.slew)
    finally:
        sup.close()


@pytest.mark.parametrize("mutation,reason", [
    ({"protocol_version": 2}, "PROTOCOL_MISMATCH"),
    ({"parameter_version": "after-restart"}, "PROCESS_VERSION_MISMATCH"),
    ({"source_state_id": 9}, "SOURCE_MISMATCH"),
    ({"expires_at_ns": 1_050_000_000}, "EXPIRED_COMMAND"),
    ({"setpoints_W": np.full(3, 2500.)}, "POWER_BOUNDS"),
])
def test_more_command_faults(mutation, reason):
    c, state = setup()
    cmd = ControlCommand(1, 1, state.state_id, state.sample_time_ns,
                         state.clock_domain_id, 1, 1_500_000_000,
                         np.full(3, 200.), "rom-1", "synthetic-v1", 1)
    validator = CommandValidator(c)
    signed = SignedCommand.create(replace(cmd, **mutation))
    assert validator.validate(signed, state, np.zeros(3), now_ns=1_100_000_000,
                              finish_ns=1_100_000_000, current_cycle=0,
                              expected_generation=1) == reason


def test_late_worker_result_and_bounded_queue():
    c, state = setup()
    blocker = threading.Event()
    sup = RuntimeSupervisor(c, RuntimePolicy(deadline_ns=1_000_000))
    try:
        old_generation = sup.submit(lambda: (blocker.wait(timeout=1.), np.full(3, 200.))[1])
        assert sup.submit(lambda: np.zeros(3)) is None
        applied = sup.select(state, np.zeros(3), 370., current_cycle=0,
                             request_generation=old_generation,
                             start_ns=state.sample_time_ns,
                             now_ns=state.sample_time_ns + 2_000_000)
        assert applied.mode == "FALLBACK" and applied.reason == "DEADLINE_OVERRUN"
        blocker.set()
        sup.inflight.result(timeout=1)
        late = sup.select(state, np.zeros(3), 370., current_cycle=0,
                          request_generation=old_generation,
                          start_ns=state.sample_time_ns,
                          now_ns=state.sample_time_ns + 3_000_000)
        assert late.reason == "STALE_GENERATION"
    finally:
        blocker.set()
        sup.close()


def test_stale_sensor_and_integrity_mismatch_reject_commands():
    c, state = setup()
    cmd = ControlCommand(1, 1, state.state_id, state.sample_time_ns,
                         state.clock_domain_id, 1, 20_000_000_000,
                         np.full(3, 200.), "rom-1", "synthetic-v1", 1)
    policy = RuntimePolicy(max_state_age_ns=2_000_000_000,
                           deadline_ns=20_000_000_000)
    validator = CommandValidator(c, policy)
    assert validator.validate(SignedCommand.create(cmd), state, np.zeros(3),
                              now_ns=4_000_000_001, finish_ns=1_100_000_000,
                              current_cycle=0, expected_generation=1) == "STALE_SENSOR"
    tampered = SignedCommand(replace(cmd, setpoints_W=np.full(3, 300.)),
                             SignedCommand.create(cmd).integrity_check)
    assert validator.validate(tampered, state, np.zeros(3),
                              now_ns=1_100_000_000, finish_ns=1_100_000_000,
                              current_cycle=0, expected_generation=1) == "INTEGRITY_MISMATCH"


def test_invalid_sensor_estimate_and_shape_reject_commands():
    c, state = setup()
    cmd = ControlCommand(1, 1, state.state_id, state.sample_time_ns,
                         state.clock_domain_id, 1, 1_500_000_000,
                         np.full(3, 200.), "rom-1", "synthetic-v1", 1)
    validator = CommandValidator(c)
    invalid = replace(state, valid=False)
    assert validator.validate(SignedCommand.create(cmd), invalid, np.zeros(3),
                              now_ns=1_100_000_000, finish_ns=1_100_000_000,
                              current_cycle=0, expected_generation=1) == "INVALID_STATE"
    malformed = replace(cmd, setpoints_W=np.zeros(2))
    assert validator.validate(SignedCommand.create(malformed), state, np.zeros(3),
                              now_ns=1_100_000_000, finish_ns=1_100_000_000,
                              current_cycle=0, expected_generation=1) == "INVALID_SHAPE_OR_NONFINITE"


def test_missing_job_uses_bounded_fallback():
    c, state = setup()
    sup = RuntimeSupervisor(c)
    try:
        applied = sup.select(state, np.zeros(3), 370., current_cycle=0,
                             request_generation=None, start_ns=state.sample_time_ns,
                             now_ns=state.sample_time_ns + 1_000_000)
        assert applied.mode == "FALLBACK" and applied.reason == "QUEUE_FULL"
        assert np.all(applied.actual_power_W >= 0)
        assert np.all(applied.actual_power_W <= np.asarray(c.slew))
    finally:
        sup.close()


def test_corrupted_warm_start_cannot_break_cem_actuator_bounds():
    c, state = setup()
    settings = MPCConfig(horizon=4, blocks=2, candidates=8, iterations=2, elite=2)
    mpc = CEMMPC(c, ROMPredictor(c), settings, seed=3)
    mpc.warm = np.full((4, c.zones), np.nan)
    result = mpc.optimize(state.field_K, state.zones_K,
                          np.full(4, 380.), np.zeros(c.zones), 10.)
    assert result.status == "OK"
    assert np.all(result.sequence_W >= 0)
    assert np.all(result.sequence_W <= np.asarray(c.pmax))
    assert np.all(np.abs(np.diff(np.vstack([np.zeros(c.zones), result.sequence_W]),
                                 axis=0)) <= np.asarray(c.slew) + 1e-9)
