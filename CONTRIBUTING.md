# Contributing

Contributions should include a clear synthetic or measured process scope, units, API compatibility and evidence for changed numerical/control behavior. Keep new tests at whole-trajectory or whole-scenario grain where applicable. Do not use hidden plant state in a controller or silently change a frozen protocol after test access.

Run `python -m pytest -q` and the CPU quickstart before proposing code changes. A new process geometry, sensor layout or actuator interface needs an explicit validation plan. Do not connect unreviewed code to live equipment. The project has not yet published a supported security or hardware deployment process.
