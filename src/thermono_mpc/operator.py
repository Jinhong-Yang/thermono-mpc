"""A small direct-horizon Fourier neural operator for candidate controls."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import Prediction
from .process import ProcessConfig


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int):
        super().__init__()
        self.modes_y, self.modes_x = modes_y, modes_x
        scale = 1 / width
        self.positive = nn.Parameter(scale * torch.randn(width, width, modes_y, modes_x, dtype=torch.cfloat))
        self.negative = nn.Parameter(scale * torch.randn(width, width, modes_y, modes_x, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, width, ny, nx = x.shape
        spectrum = torch.fft.rfft2(x)
        out = torch.zeros(b, width, ny, nx // 2 + 1, dtype=spectrum.dtype, device=x.device)
        my = min(self.modes_y, ny // 2)
        mx = min(self.modes_x, nx // 2 + 1)
        out[:, :, :my, :mx] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, :my, :mx], self.positive[:, :, :my, :mx])
        out[:, :, -my:, :mx] = torch.einsum("bixy,ioxy->boxy", spectrum[:, :, -my:, :mx], self.negative[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=(ny, nx))


class DirectFNO(nn.Module):
    """Inputs include the complete H-by-zone future power proposal."""

    def __init__(self, zones: int, horizon: int, width: int = 24,
                 layers: int = 4, modes: int = 8, pad: int = 4):
        super().__init__()
        if zones < 1 or horizon < 1 or width < 1 or layers < 1 or modes < 1:
            raise ValueError("invalid FNO dimensions")
        self.zones, self.horizon, self.pad = zones, horizon, pad
        channels = 1 + zones + zones + 2 + horizon * zones + 3
        self.lift = nn.Conv2d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.field_head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, horizon, 1))
        self.zone_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, horizon * zones))

    def forward(self, field_K: torch.Tensor, zones_K: torch.Tensor,
                power_W: torch.Tensor, material: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if field_K.ndim != 3 or zones_K.ndim != 2 or power_W.ndim != 3:
            raise ValueError("expected [B,Ny,Nx], [B,Z], [B,H,Z]")
        b, ny, nx = field_K.shape
        if zones_K.shape != (b, self.zones) or power_W.shape != (b, self.horizon, self.zones):
            raise ValueError("batch/horizon/zone shape mismatch")
        if material is None:
            material = torch.ones(b, 3, device=field_K.device, dtype=field_K.dtype)
        if material.shape != (b, 3):
            raise ValueError("material shape [B,3] required")
        x = (torch.arange(nx, device=field_K.device, dtype=field_K.dtype) + .5) / nx
        y = (torch.arange(ny, device=field_K.device, dtype=field_K.dtype) + .5) / ny
        xx = x[None, None, None, :].expand(b, 1, ny, nx)
        yy = y[None, None, :, None].expand(b, 1, ny, nx)
        mask_idx = torch.clamp((x * self.zones).long(), max=self.zones - 1)
        mask = F.one_hot(mask_idx, self.zones).T[None, :, None, :].expand(b, -1, ny, -1)
        zone_field = zones_K[:, :, None, None].expand(-1, -1, ny, nx)
        control_field = (power_W / 2000.).reshape(b, self.horizon * self.zones, 1, 1).expand(-1, -1, ny, nx)
        material_field = material[:, :, None, None].expand(-1, -1, ny, nx)
        features = torch.cat(((field_K[:, None] - 300.) / 200., mask.to(field_K.dtype),
                              (zone_field - 300.) / 200., xx, yy,
                              control_field, material_field), dim=1)
        hidden = self.lift(features)
        if self.pad:
            hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        for spectral, local in zip(self.spectral, self.local):
            hidden = F.gelu(spectral(hidden) + local(hidden))
        if self.pad:
            hidden = hidden[:, :, self.pad:-self.pad, self.pad:-self.pad]
        field = field_K[:, None] + 200 * self.field_head(hidden)
        zone = zones_K[:, None] + 200 * self.zone_head(hidden.mean(dim=(-2, -1))).reshape(b, self.horizon, self.zones)
        return field, zone


class TorchPredictor:
    def __init__(self, model: DirectFNO, config: ProcessConfig, device: str = "cpu",
                 version: str = "untrained", *, trained_dt_s: float = 10.0,
                 material_context: np.ndarray | None = None):
        if not np.isfinite(trained_dt_s) or trained_dt_s <= 0:
            raise ValueError("trained_dt_s must be a finite positive interval")
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.version = version
        self.trained_dt_s = float(trained_dt_s)
        self.material_context = None if material_context is None else np.asarray(material_context, dtype=np.float32).copy()
        if self.material_context is not None and (self.material_context.shape != (3,) or
                not np.all(np.isfinite(self.material_context)) or np.any(self.material_context <= 0)):
            raise ValueError("material_context requires three positive density/capacity/conductivity ratios")

    @torch.inference_mode()
    def predict(self, field_K: np.ndarray, zones_K: np.ndarray,
                candidates_W: np.ndarray, dt_s: float) -> Prediction:
        c = self.config
        if not np.isfinite(dt_s) or not np.isclose(dt_s, self.trained_dt_s, rtol=0, atol=1e-9):
            raise ValueError(f"This checkpoint supports dt_s={self.trained_dt_s:g} only; received {dt_s!r}. Adapt and retrain for another interval.")
        u = np.asarray(candidates_W, dtype=np.float32)
        if u.ndim != 3 or u.shape[1:] != (self.model.horizon, c.zones):
            raise ValueError(f"This checkpoint requires [B,{self.model.horizon},{c.zones}] candidate powers; received {u.shape}")
        if len(u) == 0 or not np.all(np.isfinite(u)):
            raise ValueError("candidate powers require a nonempty batch of finite values")
        if np.shape(field_K) != (c.ny, c.nx) or np.shape(zones_K) != (c.zones,):
            raise ValueError("initial field/zone shape differs from the configured prediction grid")
        if not np.all(np.isfinite(field_K)) or not np.all(np.isfinite(zones_K)):
            raise ValueError("initial temperatures must be finite")
        b = len(u)
        tfield = torch.as_tensor(np.broadcast_to(field_K, (b, c.ny, c.nx)).copy(), dtype=torch.float32, device=self.device)
        tzone = torch.as_tensor(np.broadcast_to(zones_K, (b, c.zones)).copy(), dtype=torch.float32, device=self.device)
        tcontrol = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        if self.material_context is None:
            out_field, out_zone = self.model(tfield, tzone, tcontrol)
        else:
            material = torch.as_tensor(np.broadcast_to(self.material_context, (b, 3)).copy(), dtype=torch.float32, device=self.device)
            out_field, out_zone = self.model(tfield, tzone, tcontrol, material)
        return Prediction(out_field.cpu().numpy(), out_zone.cpu().numpy(), self.version)


def load_npz_weights(model: DirectFNO, path: str) -> None:
    """Load numeric arrays only; never execute arbitrary checkpoint pickle."""
    with np.load(path, allow_pickle=False) as data:
        state = {}
        for key, value in model.state_dict().items():
            if value.is_complex():
                arr = data[key + ".real"] + 1j * data[key + ".imag"]
            else:
                arr = data[key]
            if arr.shape != tuple(value.shape):
                raise ValueError(f"checkpoint tensor shape mismatch: {key}")
            state[key] = torch.as_tensor(arr, dtype=value.dtype)
        model.load_state_dict(state)
