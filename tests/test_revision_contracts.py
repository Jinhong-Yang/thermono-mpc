import json
import numpy as np
import pytest
import torch
from thermono_mpc.contracts import Prediction
from thermono_mpc.diagnostics import common_state_comparison, thermal_replay
from thermono_mpc.linear_predictor import LinearThermalPredictor
from thermono_mpc.operator import DirectFNO, TorchPredictor
from thermono_mpc.process import ProcessConfig, ThermalPlant
from thermono_mpc.simulation import make_predictor


@pytest.mark.parametrize("interval", [5., 20., 0., -1., np.nan, np.inf])
def test_fixed_interval_checkpoint_rejects_unsupported_intervals(interval):
    c = ProcessConfig(nx=8, ny=8)
    predictor = TorchPredictor(DirectFNO(3, 4, width=4, layers=1, modes=2), c, trained_dt_s=10.)
    with pytest.raises(ValueError, match="supports dt_s=10"):
        predictor.predict(np.full((8, 8), 300.), np.full(3, 300.), np.zeros((1, 4, 3)), interval)


def test_interval_validation_does_not_change_valid_inference():
    torch.manual_seed(17)
    c = ProcessConfig(nx=8, ny=8)
    model = DirectFNO(3, 4, width=4, layers=1, modes=2).eval()
    f, z, u = np.full((8, 8), 300.), np.full(3, 305.), np.full((2, 4, 3), 200.)
    with torch.inference_mode():
        expected = model(torch.tensor(np.broadcast_to(f, (2, 8, 8)).copy(), dtype=torch.float32),
                         torch.tensor(np.broadcast_to(z, (2, 3)).copy(), dtype=torch.float32), torch.tensor(u, dtype=torch.float32))
    predictor = TorchPredictor(model, c)
    result = predictor.predict(f, z, u, 10.)
    np.testing.assert_array_equal(result.temperature_K, expected[0].numpy())
    np.testing.assert_array_equal(result.zone_temperature_K, expected[1].numpy())
    with pytest.raises(ValueError, match="requires"):
        predictor.predict(f, z, np.zeros((1, 5, 3)), 10.)


def test_unknown_legacy_time_metadata_is_not_guessed(tmp_path):
    meta = tmp_path/"model.json"
    meta.write_text(json.dumps({"manifest_sha256": "unknown"}))
    with pytest.raises(ValueError, match="lacks control_dt_s"):
        make_predictor(ProcessConfig(), meta, "cpu")


def test_linear_predictor_equilibrium_positive_response_and_composition():
    c = ProcessConfig(nx=8, ny=8)
    model = LinearThermalPredictor(c)
    f, z = np.full((8, 8), c.ambient), np.full(3, c.ambient)
    zero = model.predict(f, z, np.zeros((1, 4, 3)), 10.)
    np.testing.assert_allclose(zero.temperature_K, c.ambient, atol=1e-12)
    heat = model.predict(f, z, np.full((1, 4, 3), 500.), 10.)
    assert heat.temperature_K[0, -1].mean() > heat.temperature_K[0, 0].mean() > c.ambient
    first = model.predict(f, z, np.full((1, 2, 3), 500.), 10.)
    second = model.predict(first.temperature_K[0, -1], first.zone_temperature_K[0, -1], np.full((1, 2, 3), 500.), 10.)
    np.testing.assert_allclose(second.temperature_K, heat.temperature_K[:, 2:], atol=1e-10)


def test_generic_diagnostics_accept_foreign_model_and_shared_replay_state():
    class Alternate:
        def predict(self, f, z, u, dt):
            out = np.broadcast_to(f, (len(u), u.shape[1], *f.shape)).copy() + u.sum(axis=2)[:, :, None, None]*dt
            return Prediction(out, np.broadcast_to(z, u.shape).copy(), "foreign")
    model = Alternate()
    f, z = np.full((2, 2), 300.), np.full(1, 300.)
    u = np.array([[[0.], [0.]], [[1.], [1.]]])
    def replay(actions, dt):
        pred = model.predict(f, z, actions, dt)
        return Prediction(pred.temperature_K-2., pred.zone_temperature_K, "oracle")
    result = common_state_comparison(model, f, z, u, 1., replay,
                                    lambda fields, actions: ((fields-302.)**2).mean(axis=(1, 2, 3)))
    np.testing.assert_allclose(result["field_rmse_K"], [2., 2.])
    np.testing.assert_allclose(result["predicted_cost"], [4., 1.])
    np.testing.assert_allclose(result["realized_cost"], [16., 9.])


def test_thermal_replay_copies_initial_state_for_each_plan():
    c = ProcessConfig(nx=4, ny=4)
    plant = ThermalPlant(c)
    initial = plant.reset()
    plan = np.full((2, 2, 3), 400.)
    replay = thermal_replay(plant, initial)
    result = replay(plan, 10.)
    np.testing.assert_array_equal(result.temperature_K[0], result.temperature_K[1])
    np.testing.assert_array_equal(initial.field, np.full((4, 4), c.initial))
