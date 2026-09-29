"""Public shape, unit, and version contracts."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have finite shape {shape}, received {a.shape}")
    return a


@dataclass(frozen=True)
class StateEstimate:
    field_K: np.ndarray
    zones_K: np.ndarray
    sample_time_ns: int
    state_id: int
    clock_domain_id: str
    parameter_version: str
    valid: bool = True

    def validate(self, ny: int, nx: int, zones: int) -> None:
        finite_array(self.field_K, (ny, nx), "field_K")
        finite_array(self.zones_K, (zones,), "zones_K")
        if self.sample_time_ns < 0 or self.state_id < 0 or not self.clock_domain_id or not self.parameter_version:
            raise ValueError("invalid state metadata")


@dataclass(frozen=True)
class ControlPlan:
    power_W: np.ndarray  # [H, n_zones]
    dt_s: float
    process_version: str

    def validate(self, horizon: int, zones: int) -> None:
        finite_array(self.power_W, (horizon, zones), "power_W")
        if self.dt_s <= 0 or not np.isfinite(self.dt_s) or not self.process_version:
            raise ValueError("invalid plan metadata")


@dataclass(frozen=True)
class Prediction:
    temperature_K: np.ndarray  # [B, H, Ny, Nx]
    zone_temperature_K: np.ndarray  # [B, H, n_zones]
    model_version: str

    def validate(self, batch: int, horizon: int, ny: int, nx: int, zones: int) -> None:
        finite_array(self.temperature_K, (batch, horizon, ny, nx), "temperature_K")
        finite_array(self.zone_temperature_K, (batch, horizon, zones), "zone_temperature_K")
        if np.min(self.temperature_K) <= 0 or np.min(self.zone_temperature_K) <= 0:
            raise ValueError("predicted kelvin temperature must be positive")


@dataclass(frozen=True)
class ControlCommand:
    protocol_version: int
    sequence_id: int
    source_state_id: int
    source_sample_time_ns: int
    clock_domain_id: str
    apply_cycle: int
    expires_at_ns: int
    setpoints_W: np.ndarray
    model_version: str
    parameter_version: str
    generation_id: int
    units: str = "W"


@dataclass(frozen=True)
class AppliedCommand:
    sequence_id: int
    apply_cycle: int
    actual_power_W: np.ndarray
    mode: str
    reason: str
    applied_at_ns: int
