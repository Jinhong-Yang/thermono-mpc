import importlib.util
from pathlib import Path
import numpy as np
from thermono_mpc.process import ProcessConfig
from thermono_mpc.controllers import MPCConfig


def test_five_start_slsqp_keeps_constraints_and_reuses_warm_plan(monkeypatch):
    scripts=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('diagnostics_additional',scripts/'diagnostics_additional.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=ProcessConfig(nx=4,ny=4)
    controller=module.FiveStartSLSQP(config,MPCConfig(horizon=2,blocks=2),42)
    previous=np.array([700.,900.,1100.]);field=np.full((4,4),310.);zones=np.full(3,330.)
    for _ in range(2):
        result=controller.optimize(field,zones,np.array([315.,320.]),previous,10.)
        assert result.status=='OK'
        assert len(controller.calls[-1]['starts'])==5
        assert np.all(result.sequence_W>=-1e-5) and np.all(result.sequence_W<=np.array(config.pmax)+1e-5)
        delta=np.diff(np.vstack([previous,result.sequence_W]),axis=0)
        assert np.all(np.abs(delta)<=np.asarray(config.slew)+1e-5)
        np.testing.assert_array_equal(controller.warm,result.sequence_W)
        previous=result.power_W
