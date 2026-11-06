import numpy as np
import pytest
from thermono_mpc.contracts import Prediction
from thermono_mpc.controllers import MPCConfig, SLSQPMPC, CEMMPC, ROMPredictor
from thermono_mpc.diagnostics import plan_objective
from thermono_mpc.grid_adapter import NestedGridPredictor
from thermono_mpc.optimization import ConfiguredSLSQPMPC
from thermono_mpc.process import ProcessConfig


def test_thermal_objective_agrees_with_selected_cem_cost():
    c=ProcessConfig(nx=8,ny=8)
    s=MPCConfig(horizon=4,blocks=2,candidates=16,elite=4,iterations=2,energy_weight=.0001)
    f,z=np.full((8,8),300.),np.full(3,310.)
    ref=np.full(4,320.);previous=np.full(3,100.)
    predictor=ROMPredictor(c)
    result=CEMMPC(c,predictor,s,seed=4).optimize(f,z,ref,previous,10.)
    prediction=predictor.predict(f,z,result.sequence_W[None],10.)
    actual=plan_objective(prediction.temperature_K,result.sequence_W[None],ref,previous,s,10.)[0]
    assert actual == pytest.approx(result.objective,abs=1e-14)


def test_unscaled_two_start_comparator_matches_legacy_commands():
    c=ProcessConfig(nx=8,ny=8)
    s=MPCConfig(horizon=4,blocks=2,energy_weight=.0001)
    legacy=SLSQPMPC(c,s)
    comparator=ConfiguredSLSQPMPC(c,s,normalized=False,starts=2)
    for previous,field_temp in [(np.zeros(3),293.15),(np.full(3,400.),300.)]:
        f,z=np.full((8,8),field_temp),np.full(3,330.)
        ref=np.full(4,340.)
        expected=legacy.optimize(f,z,ref,previous,10.)
        result=comparator.optimize(f,z,ref,previous,10.)
        np.testing.assert_allclose(result.sequence_W,expected.sequence_W,atol=1e-6,rtol=0)
        assert result.objective == pytest.approx(expected.objective,abs=1e-12)


def test_nested_grid_retains_batch_horizon_zone_axes_and_mean():
    class Identity:
        def predict(self,f,z,u,dt):
            assert f.shape==(2,2)
            return Prediction(np.broadcast_to(f,(len(u),u.shape[1],2,2)).copy(),np.broadcast_to(z,u.shape).copy(),"identity")
    field=np.arange(16.).reshape(4,4)+300.
    model=NestedGridPredictor(Identity(),(2,2))
    result=model.predict(field,np.array([300.]),np.zeros((3,5,1)),10.)
    assert result.temperature_K.shape==(3,5,4,4)
    np.testing.assert_allclose(result.temperature_K.mean(axis=(2,3)),field.mean())
    np.testing.assert_allclose(result.temperature_K[0,0,:2,:2],field[:2,:2].mean())
    with pytest.raises(ValueError,match="integer refinement"):
        model.predict(np.ones((5,4))*300.,np.array([300.]),np.zeros((3,5,1)),10.)
