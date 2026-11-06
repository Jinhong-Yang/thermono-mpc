"""Predictor-independent plan diagnostics with explicit replay and cost callbacks.

The caller supplies delay-aligned states, plans and a replay environment. A replay
call must start from the same cloned evaluation state for every batch member;
unexecuted counterfactual outcomes cannot be reconstructed from sensor logs alone.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from .contracts import Prediction


def plan_objective(fields, plans_W, reference_K, previous_W, settings, dt_s):
    """The thermal benchmark objective, without feasibility masking.

This objective is explicitly thermal-specific; replay and comparison functions
also accept another objective callback for a different application.
"""
    fields = np.asarray(fields, dtype=float)
    plans = np.asarray(plans_W, dtype=float)
    reference = np.asarray(reference_K, dtype=float)
    previous = np.asarray(previous_W, dtype=float)
    if fields.ndim != 4 or plans.ndim != 3 or fields.shape[:2] != plans.shape[:2]:
        raise ValueError("field and plan batch/horizon axes must agree")
    if reference.shape != (plans.shape[1],) or previous.shape != (plans.shape[2],):
        raise ValueError("reference horizon or held-power zone shape mismatch")
    if not np.isfinite(dt_s) or dt_s <= 0 or any(not np.isfinite(a).all() for a in [fields, plans, reference, previous]):
        raise ValueError("finite arrays and positive dt_s required")
    delta = np.diff(np.concatenate([np.broadcast_to(previous, (len(plans), 1, len(previous))), plans], axis=1), axis=1)
    s = settings
    return (s.tracking_weight * (((fields - reference[None, :, None, None]) / s.temperature_scale_K)**2).mean(axis=(1, 2, 3))
            + s.variance_weight * (fields.var(axis=(-2, -1)) / s.temperature_scale_K**2).mean(axis=1)
            + s.slew_weight * ((delta / s.power_scale_W)**2).mean(axis=(1, 2))
            + s.energy_weight * plans.sum(axis=(1, 2)) * dt_s / s.energy_scale_J)


def _forecast(predictor, field_K, zones_K, plans_W, dt_s):
    field, zones, plans = [np.asarray(a, dtype=float) for a in (field_K, zones_K, plans_W)]
    if field.ndim != 2 or zones.ndim != 1 or plans.ndim != 3 or plans.shape[2] != len(zones):
        raise ValueError("expected field [Ny,Nx], zones [Z] and plans [B,H,Z]")
    if min(plans.shape[:2]) < 1 or dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("nonempty plans and positive finite dt_s required")
    if any(not np.isfinite(a).all() for a in (field, zones, plans)):
        raise ValueError("diagnostic inputs must be finite")
    # Input copies isolate diagnostic consumers from accidental predictor mutation.
    prediction = predictor.predict(field.copy(), zones.copy(), plans.copy(), dt_s)
    prediction.validate(len(plans), plans.shape[1], *field.shape, len(zones))
    return prediction, field, zones, plans


def same_plan_replay(predictor, field_K, zones_K, plans_W, dt_s,
                     replay: Callable[[np.ndarray, float], Prediction]):
    prediction, field, zones, plans = _forecast(predictor, field_K, zones_K, plans_W, dt_s)
    realized = replay(plans.copy(), dt_s)
    realized.validate(len(plans), plans.shape[1], *field.shape, len(zones))
    error = prediction.temperature_K.astype(float) - realized.temperature_K
    zone_error = prediction.zone_temperature_K.astype(float) - realized.zone_temperature_K
    return {"model_version": prediction.model_version,
            "predicted": prediction, "realized": realized,
            "field_rmse_K": np.sqrt((error**2).mean(axis=(1, 2, 3))),
            "endpoint_bias_K": error[:, -1].mean(axis=(1, 2)),
            "zone_rmse_K": np.sqrt((zone_error**2).mean(axis=(1, 2)))}


def common_state_comparison(predictor, field_K, zones_K, plans_W, dt_s,
                            replay, objective):
    """Score supplied plans from one predictor state and one replay state.

Plans must be generated at this same estimated state and held command. This
function does not select the plans or imply independent diagnostic samples.
"""
    result = same_plan_replay(predictor, field_K, zones_K, plans_W, dt_s, replay)
    plans = np.asarray(plans_W, dtype=float)
    pcost = np.asarray(objective(result["predicted"].temperature_K, plans), dtype=float)
    rcost = np.asarray(objective(result["realized"].temperature_K, plans), dtype=float)
    if pcost.shape != (len(plans),) or rcost.shape != (len(plans),) or not np.isfinite([pcost, rcost]).all():
        raise ValueError("objective callback must return one finite score per plan")
    result.update(predicted_cost=pcost, realized_cost=rcost)
    return result


def thermal_replay(plant, aligned_state, *, max_step_s=2.0):
    """Adapt a ThermalPlant clone to the generic replay contract.

The state must already include the held-input delay. There is no reoptimization.
"""
    start = aligned_state.copy()
    def replay(plans, dt_s):
        fields, zones = [], []
        for sequence in plans:
            state = start.copy()
            fs, zs = [], []
            for power in sequence:
                state = plant.advance(state, power, dt_s, max_step_s)
                fs.append(state.field.copy()); zs.append(state.zones.copy())
            fields.append(fs); zones.append(zs)
        return Prediction(np.asarray(fields), np.asarray(zones), "thermal-plant-replay")
    return replay
