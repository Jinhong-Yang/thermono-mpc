"""PID, reduced-order thermal predictor, and bounded-budget CEM-MPC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy.optimize import minimize

from .contracts import Prediction
from .process import ProcessConfig


class Predictor(Protocol):
    def predict(self, field_K: np.ndarray, zones_K: np.ndarray,
                candidates_W: np.ndarray, dt_s: float) -> Prediction: ...


def limit_power(requested: np.ndarray, previous: np.ndarray, config: ProcessConfig) -> np.ndarray:
    """Sequential actuator clipping, not an Euclidean projection."""
    requested = np.asarray(requested, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if requested.shape != (config.zones,) or previous.shape != (config.zones,) or not np.all(np.isfinite(requested)):
        raise ValueError("invalid requested or previous power")
    return np.maximum(0, np.minimum(np.asarray(config.pmax),
                     np.maximum(previous - config.slew,
                                np.minimum(previous + config.slew, requested))))


@dataclass
class PIDController:
    config: ProcessConfig
    kp: float = 15.0
    ki: float = 0.01
    integral_limit: float = 20000.0

    def __post_init__(self) -> None:
        self.integral = np.zeros(self.config.zones)

    def propose(self, field_K: np.ndarray, reference_K: float,
                previous_W: np.ndarray, dt_s: float) -> np.ndarray:
        c = self.config
        if field_K.shape != (c.ny, c.nx) or not np.isfinite(reference_K):
            raise ValueError("invalid PID input")
        # Every zone sees only its own spatial section of the shared estimate.
        xzones = np.minimum((np.arange(c.nx) + .5) * c.zones / c.nx, c.zones - 1).astype(int)
        means = np.array([field_K[:, xzones == i].mean() for i in range(c.zones)])
        error = reference_K - means
        self.integral = np.clip(self.integral + dt_s * error, -self.integral_limit, self.integral_limit)
        requested = self.kp * error + self.ki * self.integral
        return limit_power(requested, previous_W, c)


class ROMPredictor:
    """Independent three-lump solid and zone energy model.

    Spatial field is reconstructed by broadcasting each lump. The model is
    deliberately low order; it does not call the evaluation plant solver.
    """

    def __init__(self, config: ProcessConfig):
        self.config = config
        xzones = np.minimum((np.arange(config.nx) + .5) * config.zones / config.nx, config.zones - 1).astype(int)
        self.mask = np.broadcast_to(xzones[None, :], (config.ny, config.nx))
        self.count = np.bincount(self.mask.ravel(), minlength=config.zones)
        self.cap = config.cell_heat_capacity * self.count
        self.area = (2 * config.ly * config.thickness / config.zones +
                     2 * config.lx * config.thickness / config.zones)

    def predict(self, field_K: np.ndarray, zones_K: np.ndarray,
                candidates_W: np.ndarray, dt_s: float) -> Prediction:
        c = self.config
        u = np.asarray(candidates_W, dtype=float)
        if u.ndim != 3 or u.shape[2] != c.zones or not np.all(np.isfinite(u)):
            raise ValueError("candidate shape [B,H,Z] required")
        b, h, _ = u.shape
        solid = np.broadcast_to(np.array([field_K[self.mask == i].mean() for i in range(c.zones)]), (b, c.zones)).copy()
        zone = np.broadcast_to(zones_K, (b, c.zones)).copy()
        fields = np.empty((b, h, c.ny, c.nx))
        zones = np.empty((b, h, c.zones))
        cap = np.asarray(self.cap)
        zcap = np.asarray(c.zone_capacity)
        loss = np.asarray(c.zone_loss)
        for t in range(h):
            # Semi-implicit exchange across the air/solid interface.
            film = np.asarray(c.h) + np.asarray(c.emissivity) * 5.670374419e-8 * (zone + solid) * (zone**2 + solid**2)
            q = film * self.area * (zone - solid)
            solid = solid + dt_s * q / cap
            zone = zone + dt_s * (np.asarray(c.zone_efficiency) * u[:, t] - q - loss * (zone - c.ambient)) / zcap
            fields[:, t] = solid[:, self.mask]
            zones[:, t] = zone
        return Prediction(fields, zones, "rom-1")


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 12
    blocks: int = 4
    candidates: int = 64
    iterations: int = 3
    elite: int = 8
    tracking_weight: float = 1.0
    variance_weight: float = 0.2
    slew_weight: float = 0.01
    energy_weight: float = 0.01
    temperature_scale_K: float = 100.0
    power_scale_W: float = 2000.0
    energy_scale_J: float = 240000.0


@dataclass(frozen=True)
class MPCResult:
    power_W: np.ndarray
    sequence_W: np.ndarray
    objective: float
    feasible_count: int
    status: str


class CEMMPC:
    def __init__(self, config: ProcessConfig, predictor: Predictor,
                 settings: MPCConfig = MPCConfig(), seed: int = 101,
                 diagnostic_hook=None):
        self.process = config
        self.predictor = predictor
        self.settings = settings
        self.rng = np.random.default_rng(seed)
        self.warm: np.ndarray | None = None
        self.diagnostic_hook = diagnostic_hook
        if settings.horizon < 1 or settings.blocks < 1 or settings.candidates < settings.elite or settings.iterations < 1:
            raise ValueError("invalid CEM settings")

    def _expand(self, blocks: np.ndarray, previous_W: np.ndarray) -> np.ndarray:
        s, c = self.settings, self.process
        indices = np.minimum(np.arange(s.horizon) * s.blocks // s.horizon, s.blocks - 1)
        raw = blocks[:, indices]
        repaired = np.empty_like(raw)
        prev = np.broadcast_to(previous_W, (len(raw), c.zones)).copy()
        for t in range(s.horizon):
            repaired[:, t] = np.clip(raw[:, t], np.maximum(0, prev - c.slew), np.minimum(c.pmax, prev + c.slew))
            prev = repaired[:, t]
        return repaired

    def optimize(self, field_K: np.ndarray, zones_K: np.ndarray,
                 reference_K: np.ndarray, previous_W: np.ndarray, dt_s: float) -> MPCResult:
        s, c = self.settings, self.process
        ref = np.asarray(reference_K, dtype=float)
        if ref.shape != (s.horizon,) or not np.all(np.isfinite(ref)):
            raise ValueError("reference shape mismatch")
        mean = np.full((s.blocks, c.zones), np.asarray(c.pmax) / 2)
        if self.warm is not None and self.warm.shape == (s.horizon, c.zones) and np.all(np.isfinite(self.warm)):
            mean = self.warm[np.minimum(np.arange(s.blocks) * s.horizon // s.blocks + 1, s.horizon - 1)]
        std = np.broadcast_to(np.asarray(c.pmax) / 3, mean.shape).copy()
        best_cost = np.inf
        best_seq = None
        feasible_count = 0
        for iteration in range(s.iterations):
            draws = self.rng.normal(mean, std, (s.candidates, s.blocks, c.zones))
            draws[0] = np.broadcast_to(previous_W, mean.shape)
            seq = self._expand(draws, previous_W)
            prediction = self.predictor.predict(field_K, zones_K, seq, dt_s)
            prediction.validate(s.candidates, s.horizon, c.ny, c.nx, c.zones)
            field = prediction.temperature_K
            zone = prediction.zone_temperature_K
            feasible = (field.max(axis=(1, 2, 3)) <= c.solid_limit) & (zone.max(axis=(1, 2)) <= c.zone_limit)
            feasible_count += int(feasible.sum())
            track = ((field - ref[None, :, None, None]) / s.temperature_scale_K) ** 2
            variance = field.var(axis=(-2, -1)) / s.temperature_scale_K**2
            delta = np.diff(np.concatenate([np.broadcast_to(previous_W, (s.candidates, 1, c.zones)), seq], axis=1), axis=1)
            cost = (s.tracking_weight * track.mean(axis=(1, 2, 3)) +
                    s.variance_weight * variance.mean(axis=1) +
                    s.slew_weight * ((delta / s.power_scale_W) ** 2).mean(axis=(1, 2)) +
                    s.energy_weight * (seq.sum(axis=(1, 2)) * dt_s / s.energy_scale_J))
            cost[~feasible] = np.inf
            order = np.argsort(cost)
            elite = order[np.isfinite(cost[order])][:s.elite]
            if self.diagnostic_hook is not None:
                self.diagnostic_hook({"iteration": iteration, "sequences_W": seq.copy(),
                                      "cost": cost.copy(), "elite_indices": elite.copy(),
                                      "iteration_best_field_K": field[elite[0]].copy() if len(elite) else None,
                                      "iteration_best_zone_K": zone[elite[0]].copy() if len(elite) else None})
            if len(elite):
                if cost[elite[0]] < best_cost:
                    best_cost = float(cost[elite[0]])
                    best_seq = seq[elite[0]].copy()
                mean = draws[elite].mean(axis=0)
                std = np.maximum(draws[elite].std(axis=0), np.asarray(c.pmax) / 100)
        if best_seq is None:
            return MPCResult(np.full(c.zones, np.nan), np.empty((0, c.zones)), np.inf, 0, "INFEASIBLE_CANDIDATES")
        self.warm = best_seq
        return MPCResult(best_seq[0], best_seq, best_cost, feasible_count, "OK")


class SLSQPMPC:
    """Deterministic ROM baseline with bounded inputs and explicit slew constraints.

    This uses SciPy SLSQP on a small nonlinear ROM, not acados or a QP. It
    offers an optimizer-diverse practical comparison to the matched CEM.
    """

    def __init__(self, config: ProcessConfig, settings: MPCConfig = MPCConfig(),
                 max_iterations: int = 80):
        self.process = config
        self.settings = settings
        self.predictor = ROMPredictor(config)
        self.max_iterations = max_iterations
        self.warm: np.ndarray | None = None

    def optimize(self, field_K: np.ndarray, zones_K: np.ndarray,
                 reference_K: np.ndarray, previous_W: np.ndarray, dt_s: float) -> MPCResult:
        s, c = self.settings, self.process
        ref = np.asarray(reference_K, dtype=float)
        if ref.shape != (s.horizon,) or not np.all(np.isfinite(ref)):
            raise ValueError("reference shape mismatch")
        indices = np.minimum(np.arange(s.horizon) * s.blocks // s.horizon, s.blocks - 1)
        def sequence(flat):
            return flat.reshape(s.blocks, c.zones)[indices]
        def predict(flat):
            seq = sequence(flat)
            prediction = self.predictor.predict(field_K, zones_K, seq[None], dt_s)
            return seq, prediction.temperature_K[0], prediction.zone_temperature_K[0]
        def objective(flat):
            seq, field, _ = predict(flat)
            delta = np.diff(np.vstack([previous_W, seq]), axis=0)
            return float(s.tracking_weight * (((field - ref[:, None, None]) / s.temperature_scale_K)**2).mean() +
                         s.variance_weight * (field.var(axis=(-2, -1)) / s.temperature_scale_K**2).mean() +
                         s.slew_weight * ((delta / s.power_scale_W)**2).mean() +
                         s.energy_weight * (seq.sum() * dt_s / s.energy_scale_J))
        def slew_margin(flat):
            blocks = flat.reshape(s.blocks, c.zones)
            delta = np.diff(np.vstack([previous_W, blocks]), axis=0)
            return np.r_[(np.asarray(c.slew) - delta).ravel(),
                         (np.asarray(c.slew) + delta).ravel()]
        def temperature_margin(flat):
            _, field, zone = predict(flat)
            return np.r_[c.solid_limit - field.max(axis=(1, 2)),
                         c.zone_limit - zone.max(axis=1)]
        if self.warm is not None and self.warm.shape == (s.horizon, c.zones):
            guess = self.warm[np.minimum(np.arange(s.blocks) * s.horizon // s.blocks + 1, s.horizon - 1)].copy()
        else:
            guess = np.broadcast_to(np.minimum(previous_W + c.slew, c.pmax), (s.blocks, c.zones)).copy()
        for j in range(s.blocks):
            guess[j] = limit_power(guess[j], previous_W if j == 0 else guess[j-1], c)
        high = np.empty_like(guess)
        prev = previous_W.copy()
        for j in range(s.blocks):
            high[j] = limit_power(np.asarray(c.pmax), prev, c)
            prev = high[j]
        options = {"maxiter": self.max_iterations, "ftol": 1e-9,
                   "eps": 1., "disp": False}
        candidates = [guess.ravel(), high.ravel(), np.zeros(s.blocks * c.zones)]
        for start in (guess.ravel(), high.ravel()):
            result = minimize(objective, start, method="SLSQP",
                              bounds=[(0, c.pmax[i]) for _ in range(s.blocks) for i in range(c.zones)],
                              constraints=[{"type": "ineq", "fun": slew_margin},
                                           {"type": "ineq", "fun": temperature_margin}],
                              options=options)
            candidates.append(result.x)
        feasible = []
        for candidate in candidates:
            if (np.all(slew_margin(candidate) >= -1e-5) and
                    np.all(temperature_margin(candidate) >= -1e-5)):
                feasible.append((objective(candidate), sequence(candidate).copy()))
        if not feasible:
            return MPCResult(np.full(c.zones, np.nan), np.empty((0, c.zones)),
                             np.inf, 0, "INFEASIBLE_CANDIDATES")
        best_cost, best_seq = min(feasible, key=lambda item: item[0])
        self.warm = best_seq
        return MPCResult(best_seq[0], best_seq, best_cost, len(feasible), "OK")
