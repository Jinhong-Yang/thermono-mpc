"""Differentiable finite-volume residual on predicted future states."""
from __future__ import annotations

import torch

from .process import ProcessConfig, ThermalPlant, STEFAN_BOLTZMANN


def finite_volume_residual(field_K: torch.Tensor, zone_K: torch.Tensor,
                           initial_field_K: torch.Tensor, initial_zone_K: torch.Tensor,
                           power_W: torch.Tensor, dt_s: float,
                           config: ProcessConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Implicit-Euler residual; normalized to a 100 K scale.

    This uses the same PDE family as the generator and is a training prior,
    not proof of prediction or closed-loop accuracy.
    """
    c = config
    b, h, ny, nx = field_K.shape
    if (ny, nx) != (c.ny, c.nx) or zone_K.shape != (b, h, c.zones):
        raise ValueError("physics residual shape mismatch")
    prev_field = torch.cat([initial_field_K[:, None], field_K[:, :-1]], dim=1)
    prev_zone = torch.cat([initial_zone_K[:, None], zone_K[:, :-1]], dim=1)
    rhs = torch.zeros_like(field_K)
    # Every interior face is visited once and inserted with opposite signs.
    gx = c.k * c.thickness * c.dy / c.dx
    gy = c.k * c.thickness * c.dx / c.dy
    qx = gx * (field_K[..., 1:] - field_K[..., :-1])
    rhs[..., :-1] += qx
    rhs[..., 1:] -= qx
    qy = gy * (field_K[..., 1:, :] - field_K[..., :-1, :])
    rhs[..., :-1, :] += qy
    rhs[..., 1:, :] -= qy
    faces = ThermalPlant(c)._faces
    device, dtype = field_K.device, field_K.dtype
    cell_index = torch.tensor([f[0] for f in faces], device=device, dtype=torch.long)
    zone_index = torch.tensor([f[1] for f in faces], device=device, dtype=torch.long)
    area = torch.tensor([f[2] for f in faces], device=device, dtype=dtype)
    half_dist = torch.tensor([f[3] for f in faces], device=device, dtype=dtype)
    ts = field_K.reshape(b, h, -1)[:, :, cell_index]
    theta = zone_K[:, :, zone_index]
    hcoef = torch.tensor(c.h, device=device, dtype=dtype)[zone_index]
    emissivity = torch.tensor(c.emissivity, device=device, dtype=dtype)[zone_index]
    film = hcoef + emissivity * STEFAN_BOLTZMANN * (ts + theta) * (ts**2 + theta**2)
    conductance = area * film / (1 + half_dist * film / c.k)
    qin = conductance * (theta - ts)
    rhs_flat = rhs.reshape(b, h, -1)
    rhs_flat.scatter_add_(2, cell_index[None, None, :].expand(b, h, -1), qin)
    zone_rhs = (torch.tensor(c.zone_efficiency, device=device, dtype=dtype) * power_W -
                torch.tensor(c.zone_loss, device=device, dtype=dtype) * (zone_K - c.ambient))
    zone_rhs.scatter_add_(2, zone_index[None, None, :].expand(b, h, -1), -qin)
    coupling = torch.tensor(c.zone_coupling, device=device, dtype=dtype)
    zone_rhs += (coupling[None, None] * (zone_K[:, :, None, :] - zone_K[:, :, :, None])).sum(dim=-1)
    solid_residual = (field_K - prev_field - dt_s * rhs / c.cell_heat_capacity) / 100.
    zone_residual = (zone_K - prev_zone - dt_s * zone_rhs /
                     torch.tensor(c.zone_capacity, device=device, dtype=dtype)) / 100.
    return solid_residual, zone_residual


def supervised_loss(pred_field: torch.Tensor, pred_zone: torch.Tensor,
                    target_field: torch.Tensor, target_zone: torch.Tensor) -> torch.Tensor:
    return (((pred_field - target_field) / 100.).square().mean() +
            ((pred_zone - target_zone) / 100.).square().mean())
