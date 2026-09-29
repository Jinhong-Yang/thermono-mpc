import json
import numpy as np
import torch

from thermono_mpc.dataset import generate_dataset, load_windows
from thermono_mpc.physics_loss import finite_volume_residual
from thermono_mpc.process import ProcessConfig


def test_dataset_split_and_hashes(tmp_path):
    c = ProcessConfig(nx=6, ny=6)
    manifest = generate_dataset(tmp_path, c, counts={"train": 2, "validation": 1},
                                steps=5, horizon=3, control_dt_s=2., max_step_s=1.)
    info = json.loads(manifest.read_text())
    assert len(info["trajectories"]) == 3
    train = load_windows(manifest, "train")
    valid = load_windows(manifest, "validation")
    assert set(train["trajectory_id"]).isdisjoint(valid["trajectory_id"])
    assert train["target_field"].shape == (6, 3, 6, 6)


def test_physics_residual_of_generated_label_is_small(tmp_path):
    c = ProcessConfig(nx=6, ny=6)
    manifest = generate_dataset(tmp_path, c, counts={"train": 1},
                                steps=3, horizon=2, control_dt_s=1., max_step_s=1.)
    data = load_windows(manifest, "train")
    f, z = finite_volume_residual(
        torch.tensor(data["target_field"]), torch.tensor(data["target_zone"]),
        torch.tensor(data["field"]), torch.tensor(data["zone"]),
        torch.tensor(data["power"]), 1., c)
    assert torch.isfinite(f).all() and torch.isfinite(z).all()
    # Labels use slightly varied material; nominal residual is not exactly zero.
    assert float(f.square().mean() + z.square().mean()) < 1e-4
