from dataclasses import replace
import pytest
from thermono_mpc.process import ProcessConfig
from thermono_mpc.simulation import Scenario,run_episode


def test_complete_probe_loss_then_recovery():
    c=ProcessConfig(nx=4,ny=4)
    _,rows=run_episode(c,Scenario('schedule',42),'B0_PID',steps=3,dt_s=10.,max_step_s=2.,
                       sensor_schedule=lambda cycle,scenario:(0.,1. if cycle==1 else 0.))
    assert [r['reason'] for r in rows]==['OK','NO_VALID_PROBES','OK']
    assert [r['mode'] for r in rows]==['CANDIDATE','FALLBACK','CANDIDATE']


@pytest.mark.parametrize('values',[(-1.,0.),(0.,1.1),(float('nan'),0.)])
def test_invalid_sensor_schedule(values):
    with pytest.raises(ValueError,match='sensor_schedule'):
        run_episode(ProcessConfig(nx=4,ny=4),Scenario('bad',42),'B0_PID',steps=1,dt_s=10.,max_step_s=2.,
                    sensor_schedule=lambda cycle,scenario:values)
