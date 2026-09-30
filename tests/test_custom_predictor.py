"""A user-defined predictor must satisfy the public contract and plug into MPC."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('custom_predictor',Path(__file__).parents[1]/'examples/custom_predictor.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)


def test_custom_predictor_contract_and_mpc_integration():
    field=np.arange(32,dtype=float).reshape(4,8)+293.
    zone=np.array([300.,310.,320.]);powers=np.zeros((5,4,3))
    predictor=example.PersistencePredictor()
    result=predictor.predict(field,zone,powers,10.)
    result.validate(5,4,4,8,3)
    np.testing.assert_array_equal(result.temperature_K[4,3],field)
    np.testing.assert_array_equal(result.zone_temperature_K[2,1],zone)
    result.temperature_K[0,0,0,0]=-1
    assert field[0,0]==293.  # Returned arrays must not alias caller-owned inputs.
    with pytest.raises(ValueError):predictor.predict(field,zone,powers,-1.)
    assert example.run_example()['plan_shape']==[4,3]
