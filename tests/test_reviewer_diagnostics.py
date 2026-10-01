"""Regression checks for the new diagnostic interventions, not frozen scores."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from thermono_mpc.controllers import SLSQPMPC, MPCConfig, ROMPredictor
from thermono_mpc.process import ProcessConfig

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from reviewer_diagnostics import ScaledSingleStart, costs

def test_variable_scaling_makes_progress_at_small_gradient():
    c=ProcessConfig(nx=8,ny=8)
    s=MPCConfig(energy_weight=.0001)
    controller=ScaledSingleStart(SLSQPMPC(c,s))
    ref=np.array([293.15+(340-293.15)*(j+2)*10/600 for j in range(12)])
    r=controller.optimize(np.full((8,8),293.15),np.full(3,293.15),ref,np.zeros(3),10.)
    assert r.status=='OK'
    call=controller.calls[0]
    assert call['nit']>1 and call['final_objective']<call['initial_objective']
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(3),r.sequence_W]),axis=0)))<=400+1e-5
    assert np.min(r.sequence_W)>=-1e-8 and np.max(r.sequence_W)<=2000+1e-8

def test_common_plan_objective_does_not_depend_on_batch_companions():
    c=ProcessConfig(nx=4,ny=4);s=MPCConfig(energy_weight=.0001)
    plans=np.stack([np.zeros((12,3)),np.full((12,3),1000.)])
    p=ROMPredictor(c).predict(np.full((4,4),293.15),np.full(3,293.15),plans,10.)
    ref=np.full(12,320.)
    batch=costs(p.temperature_K,plans,ref,np.zeros(3),s)
    separate=np.array([costs(p.temperature_K[i:i+1],plans[i:i+1],ref,np.zeros(3),s)[0] for i in range(2)])
    np.testing.assert_array_equal(batch,separate)
