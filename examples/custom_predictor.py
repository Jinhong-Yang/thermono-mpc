"""Add a predictor without changing the thermal plant, MPC, or command layer."""
import json
import numpy as np
from thermono_mpc.contracts import Prediction
from thermono_mpc.controllers import CEMMPC, MPCConfig
from thermono_mpc.process import ProcessConfig


class PersistencePredictor:
    """CPU interface example; persistence ignores power and is not a control model."""
    def predict(self, field_K: np.ndarray, zones_K: np.ndarray,
                candidates_W: np.ndarray, dt_s: float) -> Prediction:
        if dt_s <= 0 or candidates_W.ndim != 3 or candidates_W.shape[2] != len(zones_K):
            raise ValueError('Expected positive time step and [batch,horizon,zones] powers')
        batch, horizon, _ = candidates_W.shape
        return Prediction(np.broadcast_to(field_K, (batch, horizon, *field_K.shape)).copy(),
                          np.broadcast_to(zones_K, (batch, horizon, len(zones_K))).copy(),
                          'persistence-example-1')


def run_example():
    config = ProcessConfig(nx=8, ny=8)
    settings = MPCConfig(horizon=4, blocks=2, candidates=16, iterations=2, elite=4)
    mpc = CEMMPC(config, PersistencePredictor(), settings, seed=12)
    result = mpc.optimize(np.full((8, 8), config.initial), np.full(3, config.initial),
                          np.full(4, 310.), np.zeros(3), 10.)
    assert result.status == 'OK'
    return {'status': result.status, 'plan_shape': list(result.sequence_W.shape),
            'first_power_W': result.power_W.tolist(),
            'scope': 'Interface reuse demonstration; not a learned or physically responsive predictor'}


if __name__ == '__main__': print(json.dumps(run_example(), indent=2))
