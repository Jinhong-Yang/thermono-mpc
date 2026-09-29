import numpy as np
import pytest

from thermono_mpc.process import ProcessConfig, ThermalPlant, ThermalState


def test_insulated_uniform_field_and_energy():
    c = ProcessConfig(nx=8, ny=6, h=(0., 0., 0.), emissivity=(0., 0., 0.),
                      zone_loss=(0., 0., 0.))
    plant = ThermalPlant(c)
    initial = plant.reset(400.)
    final = plant.step(initial, np.zeros(3), dt=5.)
    np.testing.assert_allclose(final.field, initial.field, atol=1e-10)
    np.testing.assert_allclose(final.zones, initial.zones, atol=1e-10)
    assert abs(plant.energy(final) - plant.energy(initial)) < 1e-6


def test_hot_zone_heats_cold_solid_and_energy_balance():
    c = ProcessConfig(nx=8, ny=8, zone_loss=(0., 0., 0.))
    plant = ThermalPlant(c)
    state = ThermalState(np.full((8, 8), 293.15), np.full(3, 500.))
    final = plant.step(state, np.zeros(3), dt=2.)
    assert final.field.mean() > state.field.mean()
    assert final.zones.mean() < state.zones.mean()
    assert abs(plant.energy(final) - plant.energy(state)) / plant.energy(state) < 1e-10


def test_electric_input_and_ambient_loss_balance():
    c = ProcessConfig(nx=8, ny=8)
    plant = ThermalPlant(c)
    state = plant.reset(350.)
    power = np.array([100., 200., 300.])
    dt = 3.
    final = plant.step(state, power, dt)
    expected = dt * (np.dot(c.zone_efficiency, power) -
                     np.dot(c.zone_loss, final.zones - c.ambient))
    assert abs((plant.energy(final) - plant.energy(state)) - expected) < 1e-5


def test_face_partition_and_power_validation():
    c = ProcessConfig(nx=10, ny=6)
    plant = ThermalPlant(c)
    assert plant.boundary_zone_counts.sum() == 2 * (c.nx + c.ny)
    assert np.all(plant.boundary_zone_counts > 0)
    with pytest.raises(ValueError):
        plant.step(plant.reset(), np.array([np.nan, 0., 0.]))


def test_neumann_cosine_solution_refines():
    """Independent analytic check for insulated 2D conduction."""
    errors = []
    for n in (8, 16, 32):
        c = ProcessConfig(nx=n, ny=n, h=(0., 0., 0.), emissivity=(0., 0., 0.),
                          zone_loss=(0., 0., 0.))
        x = (np.arange(n) + .5) * c.dx
        initial = 400 + 10 * np.cos(np.pi * x / c.lx)[None, :] * np.ones((n, 1))
        state = ThermalState(initial, np.full(3, 400.))
        t = 2.
        result = ThermalPlant(c).advance(state, np.zeros(3), t, max_step_s=.25)
        alpha = c.k / (c.rho * c.cp)
        exact = 400 + 10 * np.exp(-alpha * (np.pi / c.lx)**2 * t) * np.cos(np.pi * x / c.lx)[None, :]
        errors.append(np.sqrt(np.mean((result.field - exact)**2)))
    assert errors[2] < errors[1] < errors[0], errors


def test_implicit_step_crosschecks_separate_explicit_stencil():
    """For a small dt, independent explicit Neumann stencil should agree."""
    c = ProcessConfig(nx=8, ny=8, h=(0., 0., 0.), emissivity=(0., 0., 0.),
                      zone_loss=(0., 0., 0.))
    rng = np.random.default_rng(12)
    field = 350 + rng.normal(0, 2, (8, 8))
    state = ThermalState(field, np.full(3, 350.))
    dt = 1e-3
    implicit = ThermalPlant(c).step(state, np.zeros(3), dt).field
    # Separate straightforward finite-difference implementation with
    # reflecting Neumann ghost values, deliberately not using plant matrix.
    alpha = c.k / (c.rho * c.cp)
    padded = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    laplace = ((padded[1:-1, 2:] - 2 * field + padded[1:-1, :-2]) / c.dx**2 +
               (padded[2:, 1:-1] - 2 * field + padded[:-2, 1:-1]) / c.dy**2)
    explicit = field + dt * alpha * laplace
    np.testing.assert_allclose(implicit, explicit, atol=2e-4)


def test_zone_lumped_limit_without_exchange():
    c = ProcessConfig(nx=6, ny=6, h=(0., 0., 0.), emissivity=(0., 0., 0.),
                      zone_loss=(0., 0., 0.))
    state = ThermalPlant(c).reset()
    power = np.array([100., 200., 300.])
    dt = 5.
    final = ThermalPlant(c).step(state, power, dt)
    expected_zone = state.zones + dt * np.array(c.zone_efficiency) * power / np.array(c.zone_capacity)
    np.testing.assert_allclose(final.zones, expected_zone, atol=1e-10)
