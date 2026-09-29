import numpy as np
import pytest

from thermono_mpc.estimation import SparseObserver
from thermono_mpc.process import ProcessConfig


def test_observer_uses_only_probe_values_and_causal_order():
    c = ProcessConfig(nx=8, ny=8)
    observer = SparseObserver(c)
    field = np.full((8, 8), 350.)
    zones = np.full(3, 400.)
    packet = observer.observe(field, zones, sample_time_ns=100, state_id=1,
                              rng=np.random.default_rng(1))
    estimate = observer.update(packet, np.zeros(3), 10.)
    changed = field.copy()
    changed[0, 0] = 700.  # not a probe
    observer2 = SparseObserver(c)
    packet2 = observer2.observe(changed, zones, sample_time_ns=100, state_id=1,
                                rng=np.random.default_rng(1))
    estimate2 = observer2.update(packet2, np.zeros(3), 10.)
    np.testing.assert_allclose(estimate.field_K, estimate2.field_K)
    with pytest.raises(ValueError):
        observer.update(packet, np.zeros(3), 10.)


def test_dropout_does_not_fill_from_hidden_truth():
    c = ProcessConfig(nx=8, ny=8)
    observer = SparseObserver(c)
    packet = observer.observe(np.full((8, 8), 600.), np.full(3, 400.),
                              sample_time_ns=100, state_id=1,
                              dropout_probability=1., rng=np.random.default_rng(1))
    estimate = observer.update(packet, np.zeros(3), 10.)
    assert not estimate.valid
    assert float(estimate.field_K.mean()) == pytest.approx(c.initial)
