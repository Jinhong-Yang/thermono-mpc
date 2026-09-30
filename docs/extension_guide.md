# Reuse on another synthetic thermal case

1. Define a new `ProcessConfig` with geometry, thickness, zone count, material coefficients, heat-transfer parameters, actuator bounds and temperature limits. All temperatures are kelvin, powers watts, time seconds and distances metres.
2. Run the numerical checks for the new grid and integration step. Confirm heating direction and total-energy balance. The supplied values are synthetic and do not identify an industrial process.
3. Exercise PID and ROM-CEM first. `examples/reuse_second_case.py` demonstrates a four-zone case without changing the process or controller core.
4. Generate new whole-trajectory splits and train a new FNO for a different zone count or process context. Do not load the three-zone checkpoint into a four-zone case.
5. Freeze a protocol and separate development, calibration and test access. Use a finer scoring grid or independent physical data when available, and compare closed-loop outcomes against the simple controllers.
6. Keep the software command validator between an optimizer and any command transport. A new hardware adapter needs independent safety engineering and authorization before live use.
