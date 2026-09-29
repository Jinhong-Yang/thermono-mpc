"""Causal sparse-probe reconstruction shared by all controllers."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import StateEstimate
from .controllers import ROMPredictor
from .process import ProcessConfig


@dataclass(frozen=True)
class SensorPacket:
    probe_K: np.ndarray
    probe_valid: np.ndarray
    zone_K: np.ndarray
    sample_time_ns: int
    state_id: int
    clock_domain_id: str = "host-monotonic-ns"
    parameter_version: str = "synthetic-v1"


class SparseObserver:
    """ROM prediction plus spatially interpolated probe innovation.

    This is not an EKF. It never reads the plant field outside fixed probes.
    """

    def __init__(self, config: ProcessConfig, probes: tuple[tuple[int, int], ...] | None = None):
        self.config = config
        if probes is None:
            probes = tuple((config.ny // 2, min(config.nx - 1,
                                               int((i + .5) * config.nx / config.zones)))
                           for i in range(config.zones))
        if not probes or any(y < 0 or y >= config.ny or x < 0 or x >= config.nx for y, x in probes):
            raise ValueError("invalid probe positions")
        self.probes = probes
        self.rom = ROMPredictor(config)
        self.estimate = np.full((config.ny, config.nx), config.initial)
        self.zones = np.full(config.zones, config.initial)
        self.previous_state_id = -1

    def observe(self, field_K: np.ndarray, zones_K: np.ndarray,
                *, sample_time_ns: int, state_id: int,
                noise_std_K: float = 0., dropout_probability: float = 0.,
                rng: np.random.Generator | None = None) -> SensorPacket:
        """Simulation sensor adapter. Controller receives only returned packet."""
        c = self.config
        if field_K.shape != (c.ny, c.nx) or zones_K.shape != (c.zones,):
            raise ValueError("plant sensor shape mismatch")
        rng = rng or np.random.default_rng()
        probe = np.array([field_K[y, x] for y, x in self.probes])
        probe = probe + rng.normal(0, noise_std_K, len(probe))
        valid = rng.random(len(probe)) >= dropout_probability
        return SensorPacket(probe, valid, zones_K.copy(), sample_time_ns, state_id)

    def update(self, packet: SensorPacket, held_power_W: np.ndarray,
               dt_s: float) -> StateEstimate:
        c = self.config
        if packet.state_id <= self.previous_state_id:
            raise ValueError("out-of-order sensor packet")
        if packet.probe_K.shape != (len(self.probes),) or packet.probe_valid.shape != (len(self.probes),):
            raise ValueError("probe shape mismatch")
        if packet.zone_K.shape != (c.zones,) or not np.all(np.isfinite(packet.zone_K)):
            raise ValueError("zone sensor invalid")
        valid = np.asarray(packet.probe_valid, dtype=bool)
        if valid.any() and not np.all(np.isfinite(packet.probe_K[valid])):
            raise ValueError("valid probe has nonfinite temperature")
        if self.previous_state_id >= 0:
            prediction = self.rom.predict(self.estimate, self.zones,
                                          np.asarray(held_power_W)[None, None, :], dt_s)
            self.estimate = prediction.temperature_K[0, 0]
        # Piecewise-linear interpolation of innovation by physical x coordinate.
        if valid.any():
            px = np.array([x for _, x in self.probes])[valid]
            innovations = np.array([packet.probe_K[i] - self.estimate[y, x]
                                    for i, (y, x) in enumerate(self.probes) if valid[i]])
            order = np.argsort(px)
            correction = np.interp(np.arange(c.nx), px[order], innovations[order])
            self.estimate = self.estimate + correction[None, :]
        self.zones = packet.zone_K.copy()
        self.previous_state_id = packet.state_id
        return StateEstimate(self.estimate.copy(), self.zones.copy(),
                             packet.sample_time_ns, packet.state_id,
                             packet.clock_domain_id, packet.parameter_version,
                             valid=bool(valid.any()))
