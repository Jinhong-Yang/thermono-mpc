"""Independent CPU linear state-space predictor for the extension tutorial."""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from .contracts import Prediction


class LinearThermalPredictor:
    """A constant-conductance linearization, integrated by a matrix exponential.

State is solid-zone mean temperature plus heating-zone temperature. The output
field is piecewise constant in the x-zones. This is an alternative example, not
a model fitted to the benchmark test set or a claim of industrial accuracy.
"""
    def __init__(self, config, linearization_K=350.0):
        self.config = config
        c = config
        xzones = np.minimum(((np.arange(c.nx) + .5) * c.zones / c.nx).astype(int), c.zones - 1)
        self.mask = np.broadcast_to(xzones, (c.ny, c.nx))
        cap = np.bincount(self.mask.ravel(), minlength=c.zones) * c.cell_heat_capacity
        area = (2 * c.ly * c.thickness + 2 * c.lx * c.thickness) / c.zones
        conductance = (np.asarray(c.h) + 4 * np.asarray(c.emissivity) * 5.670374419e-8 * linearization_K**3) * area
        a = np.zeros((2*c.zones, 2*c.zones)); b = np.zeros((2*c.zones, c.zones))
        for i in range(c.zones):
            j = c.zones + i
            a[i, i], a[i, j] = -conductance[i]/cap[i], conductance[i]/cap[i]
            a[j, i] = conductance[i]/c.zone_capacity[i]
            a[j, j] = -(conductance[i]+c.zone_loss[i])/c.zone_capacity[i]
            b[j, i] = c.zone_efficiency[i]/c.zone_capacity[i]
        self.a, self.b = a, b
        self._transition = {}

    def predict(self, field_K, zones_K, candidates_W, dt_s):
        c = self.config
        u = np.asarray(candidates_W, dtype=float)
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")
        if np.shape(field_K) != (c.ny, c.nx) or np.shape(zones_K) != (c.zones,):
            raise ValueError("initial field/zone shape mismatch")
        if u.ndim != 3 or min(u.shape[:2]) == 0 or u.shape[2] != c.zones or not np.isfinite(u).all():
            raise ValueError("finite nonempty candidate powers [B,H,Z] required")
        if not np.isfinite(field_K).all() or not np.isfinite(zones_K).all():
            raise ValueError("initial temperatures must be finite")
        if dt_s not in self._transition:
            generator = np.zeros((3*c.zones, 3*c.zones))
            generator[:2*c.zones, :2*c.zones] = self.a
            generator[:2*c.zones, 2*c.zones:] = self.b
            transition = expm(generator*dt_s)
            self._transition[dt_s] = transition[:2*c.zones, :2*c.zones], transition[:2*c.zones, 2*c.zones:]
        ad, bd = self._transition[dt_s]
        initial = np.r_[[np.mean(np.asarray(field_K)[self.mask == i]) for i in range(c.zones)], zones_K] - c.ambient
        state = np.broadcast_to(initial, (len(u), len(initial))).copy()
        fields = np.empty((len(u), u.shape[1], c.ny, c.nx)); zones = np.empty(u.shape)
        for t in range(u.shape[1]):
            state = state @ ad.T + u[:, t] @ bd.T
            fields[:, t] = state[:, :c.zones][:, self.mask] + c.ambient
            zones[:, t] = state[:, c.zones:] + c.ambient
        return Prediction(fields, zones, "linear-thermal-exponential-1")
