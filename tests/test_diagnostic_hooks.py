"""Instrumentation must preserve decisions and the one-cycle delay."""
import numpy as np
from thermono_mpc.controllers import MPCConfig
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import Scenario, run_episode


def test_diagnostic_hooks_preserve_episode_and_time_alignment():
    config = ProcessConfig(nx=4, ny=4)
    settings = MPCConfig(horizon=2, blocks=2, candidates=8, elite=2, iterations=2)
    kwargs = dict(steps=3, dt_s=10., max_step_s=2., horizon=2, mpc_settings=settings)
    _, base = run_episode(config, Scenario('test', 12), 'B1_ROM_CEM', **kwargs)
    events, candidates = [], []
    def configure(mpc):
        mpc.diagnostic_hook = candidates.append
        return mpc
    _, traced = run_episode(config, Scenario('test', 12), 'B1_ROM_CEM',
                            controller_transform=configure, diagnostic_hook=events.append, **kwargs)
    assert len(events) == 3 and len(candidates) == 6
    for before, after, event in zip(base, traced, events):
        for key in before:
            if key != 'decision_latency_s':
                assert before[key] == after[key]
        assert event['state_after_held'].time_s == event['state_before'].time_s + 10.
        assert np.array_equal(event['result'].sequence_W[0], after['selected_next_power_W'])
    assert traced[0]['applied_power_W'] == [0., 0., 0.]
    assert traced[1]['applied_power_W'] == traced[0]['selected_next_power_W']
