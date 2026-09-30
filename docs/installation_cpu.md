# CPU installation

ThermoNO-MPC requires Python 3.11 or newer. The core package needs NumPy, SciPy and PyYAML; PyTorch is optional for the CPU-only plant, PID and ROM examples.

From a checkout, create a new environment and install the package:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install --upgrade pip
python -m pip install -e .
thermono-mpc doctor
thermono-mpc demo
python examples/cpu_quickstart.py
python examples/reuse_second_case.py
```

For tests, use `python -m pip install -e ".[dev]"` and `python -m pytest`. The development extra includes PyTorch because the operator and differentiable-residual tests import it. A build can be made with `python -m build`; install the wheel in a new environment outside the checkout to verify that the CLI and CPU path do not rely on editable source files.

The small examples are demonstrations of a synthetic process. They do not load a published furnace model or connect to equipment.
