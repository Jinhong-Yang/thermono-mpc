# ThermoNO-MPC

Development-stage research software for synthetic multi-zone heat treatment.
The current package implements a conservative numerical plant and is not a
validated manufacturing controller. It must not be connected to equipment.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
.venv\Scripts\python -m pytest
```

All internal temperatures use kelvin, dimensions metres, time seconds, and
heater inputs watts. The 2D cross-section assumes insulated front and back
faces with a declared effective thickness. Process coefficients are synthetic
design assumptions, not measured material properties.

Publication, copyright, and license approval are pending.
