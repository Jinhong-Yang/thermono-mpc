# Extending a predictor

`examples/custom_predictor.py` implements the public `Predictor` protocol and
passes it directly to `CEMMPC`. Run it on CPU with:

```text
python examples/custom_predictor.py
```

The example persists the initial temperatures. It demonstrates the array
contract and optimizer integration; it is not a useful heating model because
its forecasts do not respond to heater power.

## Contract

The predictor implements:

```python
def predict(self, field_K: np.ndarray, zones_K: np.ndarray,
            candidates_W: np.ndarray, dt_s: float) -> Prediction:
    ...
```

| Value | Shape | Meaning |
|---|---|---|
| `field_K` | `(Ny, Nx)` | Current workpiece field in kelvin |
| `zones_K` | `(n_zones,)` | Current zone temperatures in kelvin |
| `candidates_W` | `(B, H, n_zones)` | Batch of complete future heater plans in watts |
| `dt_s` | scalar | Seconds per predicted control step |
| `Prediction.temperature_K` | `(B, H, Ny, Nx)` | Fields after each future step |
| `Prediction.zone_temperature_K` | `(B, H, n_zones)` | Zones after each future step |
| `Prediction.model_version` | string | Predictor identity |

The first forecast is after the first proposed input interval. Implementations
must return finite positive temperatures and must not mutate input arrays.
The optimizer evaluates whole plans. An accurate one-step predictor needs an
explicit, tested rollout adapter before it can satisfy this interface.
`tests/test_custom_predictor.py` checks shapes, independent output storage,
invalid intervals and integration with the shared optimizer.

## Reuse and retraining

| Change | Reusable components | Work required for a learned model |
|---|---|---|
| New thermal geometry or material | Process configuration, data generation, observer, physical controllers | Generate relevant trajectories and validate or retrain |
| Three to four heating zones | Process and controller interfaces; `reuse_second_case.py` | Retrain; the released three-zone weights have incompatible input/output dimensions |
| New grid at the same geometry | Finite-volume plant and operator implementation | Evaluate accuracy explicitly; direct execution on another grid does not establish accuracy |
| New control interval or prediction horizon | Contract design and physical components | Adapt the architecture/rollout and train for the new interval or horizon |
| Alternative predictor | CEM, command records, validation and scoring | Implement the contract and compare under the same observation and optimization settings |

Released FNO checkpoints predict 12 steps at 10 seconds per step. Their
implementation accepts the shared `dt_s` argument but the learned output is
tied to that trained interval. Inference uses nominal material channels.
