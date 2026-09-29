"""Closed-loop synthetic evaluation with a one-cycle application delay."""
from __future__ import annotations

from concurrent.futures import TimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
import time
import numpy as np

from .contracts import StateEstimate
from .controllers import CEMMPC, SLSQPMPC, MPCConfig, PIDController, ROMPredictor, limit_power
from .estimation import SparseObserver
from .process import ProcessConfig, ThermalPlant
from .runtime import RuntimePolicy, RuntimeSupervisor


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    category: str = "nominal"
    rho_scale: float = 1.
    cp_scale: float = 1.
    k_scale: float = 1.
    sensor_noise_K: float = 0.
    sensor_dropout: float = 0.


def reference_K(time_s: float, *, initial: float = 293.15,
                target: float = 340., ramp_s: float = 600.) -> float:
    return initial + (target - initial) * min(max(time_s / ramp_s, 0.), 1.)


def make_predictor(config: ProcessConfig, model_metadata: Path, device: str):
    import json
    from .operator import DirectFNO, TorchPredictor, load_npz_weights
    metadata = json.loads(model_metadata.read_text(encoding="utf-8"))
    arch = metadata["architecture"]
    if arch["zones"] != config.zones:
        raise ValueError("checkpoint zone count differs from process")
    model = DirectFNO(arch["zones"], arch["horizon"], arch["width"],
                      arch["layers"], arch["modes"])
    load_npz_weights(model, str(model_metadata.parent / metadata["weights"]))
    return TorchPredictor(model, config, device, metadata["weights_sha256"][:12])


def run_episode(config: ProcessConfig, scenario: Scenario, controller_name: str,
                *, steps: int, dt_s: float, max_step_s: float,
                horizon: int = 4, target_K: float = 340., ramp_s: float = 600.,
                observation: str = "partial", checkpoint: Path | None = None,
                device: str = "cpu", mpc_settings: MPCConfig | None = None,
                deadline_s: float = .5,
                pid_gains: tuple[float, float] = (15., .01)) -> tuple[dict, list[dict]]:
    if observation not in ("partial", "full"):
        raise ValueError("invalid observation mode")
    actual = replace(config, rho=config.rho * scenario.rho_scale,
                     cp=config.cp * scenario.cp_scale, k=config.k * scenario.k_scale)
    plant = ThermalPlant(actual)
    state = plant.reset()
    observer = SparseObserver(config)
    rom = ROMPredictor(config)
    pid = PIDController(config, kp=pid_gains[0], ki=pid_gains[1])
    settings = mpc_settings or MPCConfig(horizon=horizon, blocks=min(2, horizon),
                                         candidates=32, iterations=2, elite=4)
    if settings.horizon != horizon:
        raise ValueError("horizon mismatch")
    if controller_name == "B0_PID":
        mpc = None
    elif controller_name == "B1_ROM_CEM":
        mpc = CEMMPC(config, rom, settings, scenario.seed)
    elif controller_name == "B1b_ROM_SLSQP":
        mpc = SLSQPMPC(config, settings)
    elif controller_name in ("B2_DATA_NO_MPC", "B3_PINO_MPC", "B4_PINO_RUNTIME"):
        if checkpoint is None:
            raise ValueError("checkpoint required for neural MPC")
        mpc = CEMMPC(config, make_predictor(config, checkpoint, device), settings, scenario.seed)
    else:
        raise ValueError("unknown controller")
    supervisor = RuntimeSupervisor(config, RuntimePolicy(deadline_ns=int(deadline_s * 1e9)),
                                   PIDController(config, kp=pid_gains[0], ki=pid_gains[1])) if controller_name == "B4_PINO_RUNTIME" else None
    rng = np.random.default_rng(scenario.seed)
    held_W = np.zeros(config.zones)
    rows = []
    try:
        for cycle in range(steps):
            start_ns = time.perf_counter_ns()
            packet = observer.observe(state.field, state.zones,
                                      sample_time_ns=start_ns, state_id=cycle,
                                      noise_std_K=scenario.sensor_noise_K,
                                      dropout_probability=scenario.sensor_dropout,
                                      rng=rng)
            estimate = observer.update(packet, held_W, dt_s)
            if observation == "full":
                estimate = StateEstimate(state.field.copy(), state.zones.copy(),
                                         start_ns, cycle, "host-monotonic-ns",
                                         "synthetic-v1", True)
            # The command computed now is applied at the next boundary. Every
            # controller sees the same ROM projection through the held input.
            projected = rom.predict(estimate.field_K, estimate.zones_K,
                                    held_W[None, None, :], dt_s)
            # Preserve spatial deviations in the shared state estimate; add
            # only the ROM's predicted change over the held-input interval.
            zonal_mean = np.array([estimate.field_K[rom.mask == i].mean()
                                   for i in range(config.zones)])
            projected_field = estimate.field_K + projected.temperature_K[0, 0] - zonal_mean[rom.mask]
            projected_zone = projected.zone_temperature_K[0, 0]
            ref = np.array([reference_K((cycle + j + 2) * dt_s,
                                        initial=config.initial, target=target_K,
                                        ramp_s=ramp_s) for j in range(horizon)])
            reason = "OK"
            mode = "CANDIDATE"
            if not estimate.valid:
                proposed = pid.propose(projected_field, ref[0], held_W, dt_s)
                reason = "NO_VALID_PROBES"
                mode = "FALLBACK"
            elif mpc is None:
                proposed = pid.propose(projected_field, ref[0], held_W, dt_s)
            elif supervisor is None:
                result = mpc.optimize(projected_field, projected_zone, ref, held_W, dt_s)
                if result.status == "OK":
                    proposed = result.power_W
                else:
                    proposed = pid.propose(projected_field, ref[0], held_W, dt_s)
                    reason = result.status
                    mode = "FALLBACK"
            else:
                generation = supervisor.submit(lambda: mpc.optimize(projected_field, projected_zone,
                                                                     ref, held_W.copy(), dt_s))
                if generation is not None and supervisor.inflight is not None:
                    try:
                        supervisor.inflight.result(timeout=deadline_s)
                    except TimeoutError:
                        pass
                    except Exception:
                        pass
                # Convert an MPCResult into setpoints in the supervisor's
                # worker, rather than forwarding an unvalidated object.
                if supervisor.inflight is not None and supervisor.inflight.done():
                    try:
                        result = supervisor.inflight.result()
                        if result.status == "OK":
                            from concurrent.futures import Future
                            f = Future(); f.set_result(result.power_W)
                            supervisor.inflight = f
                    except Exception:
                        pass
                applied = supervisor.select(estimate, held_W, ref[0],
                                            current_cycle=cycle,
                                            request_generation=generation,
                                            start_ns=start_ns)
                proposed = applied.actual_power_W
                mode, reason = applied.mode, applied.reason
            proposed = limit_power(proposed, held_W, config)
            decision_latency_s = (time.perf_counter_ns() - start_ns) / 1e9
            # Apply the PREVIOUS cycle's selected command for the current
            # interval. This avoids a zero-delay virtual-time shortcut.
            state = plant.advance(state, held_W, dt_s, max_step_s)
            held_applied = held_W.copy()
            held_W = proposed.copy()
            actual_ref = reference_K((cycle + 1) * dt_s, initial=config.initial,
                                     target=target_K, ramp_s=ramp_s)
            rows.append({"scenario_id": scenario.scenario_id,
                         "controller": controller_name, "cycle": cycle,
                         "time_s": (cycle + 1) * dt_s,
                         "reference_K": actual_ref,
                         "mean_field_K": float(state.field.mean()),
                         "field_rmse_K": float(np.sqrt(np.mean((state.field - actual_ref)**2))),
                         "spatial_std_K": float(state.field.std()),
                         "max_field_K": float(state.field.max()),
                         "max_zone_K": float(state.zones.max()),
                         "solid_excess_K": float(max(0., state.field.max() - config.solid_limit)),
                         "zone_excess_K": float(max(0., state.zones.max() - config.zone_limit)),
                         "energy_J": float(held_applied.sum() * dt_s),
                         "decision_latency_s": decision_latency_s,
                         "mode": mode, "reason": reason,
                         "applied_power_W": held_applied.tolist(),
                         "selected_next_power_W": held_W.tolist()})
    finally:
        if supervisor is not None:
            supervisor.close()
    latency = np.array([r["decision_latency_s"] for r in rows])
    summary = {"scenario_id": scenario.scenario_id, "category": scenario.category,
               "controller": controller_name, "observation": observation,
               "rmse_K": float(np.sqrt(np.mean([r["field_rmse_K"]**2 for r in rows]))),
               "mean_spatial_std_K": float(np.mean([r["spatial_std_K"] for r in rows])),
               "energy_J": float(sum(r["energy_J"] for r in rows)),
               "solid_excess_Ks": float(sum(r["solid_excess_K"] * dt_s for r in rows)),
               "zone_excess_Ks": float(sum(r["zone_excess_K"] * dt_s for r in rows)),
               "fallback_rate": sum(r["mode"] == "FALLBACK" for r in rows) / len(rows),
               "deadline_miss_rate": sum("DEADLINE" in r["reason"] for r in rows) / len(rows),
               "latency_p50_s": float(np.percentile(latency, 50)),
               "latency_p95_s": float(np.percentile(latency, 95)),
               "latency_p99_s": float(np.percentile(latency, 99)),
               "latency_max_s": float(latency.max()),
               "latency_cold_s": float(latency[0])}
    return summary, rows
