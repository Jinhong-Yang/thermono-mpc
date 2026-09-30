import time

import numpy as np

from thermono_mpc.contracts import ControlCommand, ControlObservation, StateEstimate
from thermono_mpc.process import ProcessConfig
from thermono_mpc.process_boundary import ProcessCommandBoundary
from thermono_mpc.runtime import RuntimePolicy, SignedCommand


def test_process_boundary_accepts_compact_command_then_falls_back():
    config = ProcessConfig(nx=8, ny=8)
    policy = RuntimePolicy(deadline_ns=2_000_000_000)
    with ProcessCommandBoundary(config, policy) as boundary:
        assert boundary.alive
        sample_ns = time.perf_counter_ns()
        state = StateEstimate(np.full((8, 8), 300.), np.full(3, 300.), sample_ns,
                              1, policy.clock_domain_id, policy.process_version)
        observation = ControlObservation.from_estimate(state, config.zones)
        assert observation.zone_mean_K.shape == (config.zones,)
        assert not hasattr(observation, "field_K")
        command = ControlCommand(1, 1, 1, sample_ns, policy.clock_domain_id,
                                 1, sample_ns + 3_000_000_000,
                                 np.full(3, 200.), "test", policy.process_version, 1)
        applied = boundary.select(observation, np.zeros(3), 320.,
                                  current_cycle=0, expected_generation=1,
                                  compute_start_ns=sample_ns,
                                  finish_ns=time.perf_counter_ns(),
                                  signed_command=SignedCommand.create(command),
                                  dt_s=1., timeout_s=2.)
        assert applied.mode == "CANDIDATE" and applied.reason == "ACCEPTED"
        assert np.allclose(applied.actual_power_W, 200.)
        assert boundary.events[-1]["validated_ns"] is not None
        next_state = StateEstimate(state.field_K, state.zones_K, time.perf_counter_ns(),
                                   2, policy.clock_domain_id, policy.process_version)
        fallback = boundary.select(ControlObservation.from_estimate(next_state, config.zones),
                                   applied.actual_power_W, 320., current_cycle=1,
                                   expected_generation=2, compute_start_ns=next_state.sample_time_ns,
                                   finish_ns=time.perf_counter_ns(),
                                   signed_command=None, dt_s=1., timeout_s=2.)
        assert fallback.mode == "FALLBACK" and fallback.reason == "NO_RESULT"
        assert np.all(fallback.actual_power_W <= np.asarray(config.pmax))
        assert np.all(np.abs(fallback.actual_power_W - applied.actual_power_W) <= np.asarray(config.slew))
        third_sample = time.perf_counter_ns()
        invalid_command = ControlCommand(1, 2, 3, third_sample, policy.clock_domain_id,
                                         3, third_sample + 3_000_000_000,
                                         np.full(3, 200.), "test", policy.process_version, 3,
                                         units="kW")
        third_observation = ControlObservation(np.full(3, 300.), third_sample, 3,
                                               policy.clock_domain_id, policy.process_version)
        rejected = boundary.select(third_observation, fallback.actual_power_W, 320.,
                                   current_cycle=2, expected_generation=3,
                                   compute_start_ns=third_sample,
                                   finish_ns=time.perf_counter_ns(),
                                   signed_command=SignedCommand.create(invalid_command),
                                   dt_s=1., timeout_s=2.)
        assert rejected.mode == "FALLBACK" and rejected.reason == "UNITS_MISMATCH"


def test_process_death_uses_emergency_fallback():
    config = ProcessConfig(nx=8, ny=8)
    with ProcessCommandBoundary(config) as boundary:
        boundary._process.terminate()
        boundary._process.join(timeout=2.)
        sample_ns = time.perf_counter_ns()
        observation = ControlObservation(np.full(config.zones, 300.), sample_ns, 1,
                                         "host-monotonic-ns", "synthetic-v1")
        applied = boundary.select(observation, np.zeros(config.zones), 320.,
                                  current_cycle=0, expected_generation=1,
                                  compute_start_ns=sample_ns,
                                  finish_ns=time.perf_counter_ns(),
                                  signed_command=None, dt_s=1.)
        assert applied.mode == "FALLBACK" and applied.reason == "CONTROL_PROCESS_UNAVAILABLE"
        assert not boundary.alive
    restarted_policy = RuntimePolicy(process_version="synthetic-v2", deadline_ns=2_000_000_000)
    with ProcessCommandBoundary(config, restarted_policy) as restarted:
        sample_ns = time.perf_counter_ns()
        observation = ControlObservation(np.full(config.zones, 300.), sample_ns, 2,
                                         restarted_policy.clock_domain_id,
                                         restarted_policy.process_version)
        stale = ControlCommand(1, 1, 2, sample_ns, restarted_policy.clock_domain_id,
                               1, sample_ns + 3_000_000_000,
                               np.full(3, 200.), "test", "synthetic-v1", 1)
        rejected = restarted.select(observation, np.zeros(3), 320., current_cycle=0,
                                    expected_generation=1, compute_start_ns=sample_ns,
                                    finish_ns=time.perf_counter_ns(),
                                    signed_command=SignedCommand.create(stale), dt_s=1., timeout_s=2.)
        assert rejected.mode == "FALLBACK" and rejected.reason == "PROCESS_VERSION_MISMATCH"
