"""Conservative cell-centred finite-volume thermal plant.

All temperatures are kelvin, time seconds, length metres, and power watts.
The 2D cross-section has insulated front/back faces and a finite thickness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class ProcessConfig:
    nx: int = 16
    ny: int = 16
    lx: float = 0.12
    ly: float = 0.04
    thickness: float = 0.02
    rho: float = 7800.0
    cp: float = 500.0
    k: float = 40.0
    h: tuple[float, ...] = (25.0, 25.0, 25.0)
    emissivity: tuple[float, ...] = (0.7, 0.7, 0.7)
    zone_capacity: tuple[float, ...] = (2000.0, 2000.0, 2000.0)
    zone_efficiency: tuple[float, ...] = (0.85, 0.85, 0.85)
    zone_loss: tuple[float, ...] = (2.0, 2.0, 2.0)
    zone_coupling: tuple[tuple[float, ...], ...] = ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.))
    pmax: tuple[float, ...] = (2000.0, 2000.0, 2000.0)
    slew: tuple[float, ...] = (400.0, 400.0, 400.0)
    ambient: float = 293.15
    initial: float = 293.15
    solid_limit: float = 723.15
    zone_limit: float = 773.15

    def __post_init__(self) -> None:
        n = len(self.h)
        if self.nx < 2 or self.ny < 2 or n < 1:
            raise ValueError("grid and zone counts must be positive")
        for name in ("emissivity", "zone_capacity", "zone_efficiency", "zone_loss", "pmax", "slew"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have {n} values")
        c = np.asarray(self.zone_coupling, dtype=float)
        if c.shape != (n, n) or not np.allclose(c, c.T) or np.any(c < 0):
            raise ValueError("zone coupling must be a symmetric nonnegative matrix")
        scalars = (self.lx, self.ly, self.thickness, self.rho, self.cp, self.k, self.ambient, self.initial)
        if not np.all(np.isfinite(scalars)) or min(scalars) <= 0:
            raise ValueError("physical scales must be finite and positive")
        if any(x < 0 for x in (*self.h, *self.emissivity, *self.zone_loss, *self.pmax, *self.slew)):
            raise ValueError("negative coefficients or actuator limits")
        if any(x <= 0 for x in self.zone_capacity):
            raise ValueError("zone capacity must be positive")
        if any(not 0 <= x <= 1 for x in (*self.emissivity, *self.zone_efficiency)):
            raise ValueError("emissivity and efficiency must lie in [0,1]")

    @property
    def zones(self) -> int:
        return len(self.h)

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def cell_heat_capacity(self) -> float:
        return self.rho * self.cp * self.dx * self.dy * self.thickness

    def with_grid(self, nx: int, ny: int) -> ProcessConfig:
        return replace(self, nx=nx, ny=ny)


@dataclass
class ThermalState:
    field: np.ndarray  # [ny, nx], kelvin
    zones: np.ndarray  # [n_zones], kelvin
    time_s: float = 0.0

    def copy(self) -> ThermalState:
        return ThermalState(self.field.copy(), self.zones.copy(), self.time_s)


class ThermalPlant:
    """Independent numerical plant, never delegated to a learned predictor."""

    def __init__(self, config: ProcessConfig):
        self.config = config
        self._faces = self._boundary_faces()
        if len(self._faces) != 2 * (config.nx + config.ny):
            raise AssertionError("each in-plane boundary face must be assigned once")

    def reset(self, temperature: float | None = None) -> ThermalState:
        t = self.config.initial if temperature is None else float(temperature)
        if not np.isfinite(t) or t <= 0:
            raise ValueError("invalid initial temperature")
        return ThermalState(np.full((self.config.ny, self.config.nx), t),
                            np.full(self.config.zones, t))

    def _boundary_faces(self) -> list[tuple[int, int, float, float]]:
        c = self.config
        faces = []
        for iy in range(c.ny):
            for ix in range(c.nx):
                index = iy * c.nx + ix
                x = (ix + 0.5) * c.dx
                zone = min(int(x / (c.lx / c.zones)), c.zones - 1)
                if iy == 0:
                    faces.append((index, zone, c.dx * c.thickness, c.dy / 2))
                if iy == c.ny - 1:
                    faces.append((index, zone, c.dx * c.thickness, c.dy / 2))
                if ix == 0:
                    faces.append((index, 0, c.dy * c.thickness, c.dx / 2))
                if ix == c.nx - 1:
                    faces.append((index, c.zones - 1, c.dy * c.thickness, c.dx / 2))
        return faces

    @property
    def boundary_zone_counts(self) -> np.ndarray:
        return np.bincount([f[1] for f in self._faces], minlength=self.config.zones)

    def energy(self, state: ThermalState) -> float:
        c = self.config
        return float(c.cell_heat_capacity * state.field.sum() +
                     np.dot(c.zone_capacity, state.zones))

    def _validate(self, state: ThermalState, power: np.ndarray, dt: float) -> None:
        c = self.config
        if state.field.shape != (c.ny, c.nx) or state.zones.shape != (c.zones,):
            raise ValueError("state shape mismatch")
        if power.shape != (c.zones,) or not np.all(np.isfinite(power)):
            raise ValueError("power shape/finite mismatch")
        if np.any(power < -1e-9) or np.any(power > c.pmax):
            raise ValueError("power outside actuator bounds")
        if not np.all(np.isfinite(state.field)) or not np.all(np.isfinite(state.zones)) or np.min(state.field) <= 0 or np.min(state.zones) <= 0:
            raise ValueError("invalid temperature")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid time step")

    def step(self, state: ThermalState, power: np.ndarray, dt: float = 0.5,
             *, tol: float = 1e-8, max_iter: int = 20) -> ThermalState:
        """Implicit Euler with radiation conductance updated to convergence.

        The same face flux couples solid and zone, so their internal exchange
        cancels exactly in the discrete total-energy balance.
        """
        c = self.config
        power = np.asarray(power, dtype=float)
        self._validate(state, power, dt)
        nc = c.nx * c.ny
        z = c.zones
        old = np.r_[state.field.ravel(), state.zones]
        guess = old.copy()
        cap = c.cell_heat_capacity
        gx = c.k * c.thickness * c.dy / c.dx
        gy = c.k * c.thickness * c.dx / c.dy
        for iteration in range(max_iter):
            a = lil_matrix((nc + z, nc + z), dtype=float)
            rhs = np.zeros(nc + z)
            a.setdiag(np.r_[np.full(nc, cap / dt), np.asarray(c.zone_capacity) / dt])
            rhs[:nc] = cap / dt * old[:nc]
            rhs[nc:] = np.asarray(c.zone_capacity) / dt * old[nc:] + np.asarray(c.zone_efficiency) * power + np.asarray(c.zone_loss) * c.ambient
            for iy in range(c.ny):
                for ix in range(c.nx):
                    i = iy * c.nx + ix
                    if ix + 1 < c.nx:
                        j = i + 1
                        a[i, i] += gx; a[j, j] += gx
                        a[i, j] -= gx; a[j, i] -= gx
                    if iy + 1 < c.ny:
                        j = i + c.nx
                        a[i, i] += gy; a[j, j] += gy
                        a[i, j] -= gy; a[j, i] -= gy
            for i, zi, area, half_dist in self._faces:
                theta = guess[nc + zi]
                ts = guess[i]
                # Factorisation of (theta**4 - ts**4) avoids cancellation.
                rad = c.emissivity[zi] * STEFAN_BOLTZMANN * (theta + ts) * (theta * theta + ts * ts)
                film = c.h[zi] + rad
                conductance = area / (half_dist / c.k + 1 / film) if film > 0 else 0.0
                j = nc + zi
                a[i, i] += conductance; a[j, j] += conductance
                a[i, j] -= conductance; a[j, i] -= conductance
            for i in range(z):
                a[nc + i, nc + i] += c.zone_loss[i]
                for j in range(i + 1, z):
                    coupling = c.zone_coupling[i][j]
                    a[nc + i, nc + i] += coupling
                    a[nc + j, nc + j] += coupling
                    a[nc + i, nc + j] -= coupling
                    a[nc + j, nc + i] -= coupling
            sol = spsolve(a.tocsr(), rhs)
            if not np.all(np.isfinite(sol)) or np.min(sol) <= 0:
                raise RuntimeError("thermal solve produced invalid temperature")
            if np.max(np.abs(sol - guess)) < tol:
                return ThermalState(sol[:nc].reshape(c.ny, c.nx), sol[nc:], state.time_s + dt)
            guess = sol
        raise RuntimeError(f"radiation nonlinear solve did not converge in {max_iter} iterations")

    def advance(self, state: ThermalState, power: np.ndarray, duration_s: float,
                max_step_s: float = 0.5) -> ThermalState:
        if duration_s <= 0 or max_step_s <= 0:
            raise ValueError("durations must be positive")
        count = int(np.ceil(duration_s / max_step_s))
        dt = duration_s / count
        for _ in range(count):
            state = self.step(state, power, dt)
        return state
