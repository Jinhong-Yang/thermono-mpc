import numpy as np
import torch

from thermono_mpc.operator import DirectFNO, TorchPredictor
from thermono_mpc.process import ProcessConfig


def test_fno_shape_control_sensitivity_and_backward():
    torch.manual_seed(7)
    model = DirectFNO(3, 4, width=8, layers=2, modes=4, pad=2)
    field = torch.full((2, 8, 8), 350.)
    zone = torch.full((2, 3), 400.)
    control = torch.stack((torch.zeros(4, 3), torch.full((4, 3), 1000.)))
    tf, tz = model(field, zone, control)
    assert tf.shape == (2, 4, 8, 8) and tz.shape == (2, 4, 3)
    assert not torch.allclose(tf[0], tf[1])
    loss = tf.square().mean() + tz.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_predictor_public_axes():
    c = ProcessConfig(nx=8, ny=8)
    model = DirectFNO(3, 4, width=8, layers=1, modes=4)
    pred = TorchPredictor(model, c).predict(np.full((8, 8), 350.),
                                           np.full(3, 400.), np.zeros((2, 4, 3)), 10.)
    pred.validate(2, 4, 8, 8, 3)
