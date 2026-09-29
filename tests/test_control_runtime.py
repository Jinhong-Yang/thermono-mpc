from dataclasses import replace
import numpy as np

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
