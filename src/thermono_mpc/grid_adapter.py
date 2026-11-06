"""Explicit, mean-preserving nested-grid prediction path for diagnostics."""
import numpy as np
from .contracts import Prediction


class NestedGridPredictor:
    """Restrict a fine state by cell averaging and repeat coarse predictions.

The plant, observer and scoring stay on the caller's fine grid. This measures
the complete grid conversion path, not a pure internal neural-resolution effect.
"""
    def __init__(self, predictor, coarse_shape):
        self.predictor = predictor
        self.coarse_shape = tuple(coarse_shape)
        if len(self.coarse_shape) != 2 or min(self.coarse_shape) < 2:
            raise ValueError("coarse_shape must contain two positive grid dimensions")

    def predict(self, field_K, zones_K, candidates_W, dt_s):
        field = np.asarray(field_K)
        if field.ndim != 2:
            raise ValueError("field must have two spatial dimensions")
        cy, cx = self.coarse_shape
        ny, nx = field.shape
        if ny % cy or nx % cx or ny < cy or nx < cx:
            raise ValueError("fine grid must be an integer refinement of the coarse grid")
        sy, sx = ny//cy, nx//cx
        coarse = field.reshape(cy,sy,cx,sx).mean(axis=(1,3))
        prediction = self.predictor.predict(coarse, zones_K, candidates_W, dt_s)
        prediction.validate(len(candidates_W), candidates_W.shape[1], cy, cx, len(zones_K))
        expanded = np.repeat(np.repeat(prediction.temperature_K, sy, axis=2), sx, axis=3)
        return Prediction(expanded, prediction.zone_temperature_K.copy(),
                          prediction.model_version+f"-area{cy}x{cx}-repeat{ny}x{nx}")
