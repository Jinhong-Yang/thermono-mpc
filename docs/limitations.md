# Limitations and supported claims

- The process is a two-dimensional synthetic heat-treatment cross-section with insulated front/back faces and declared effective thickness. Numerical self-checks establish internal consistency, not fidelity to a furnace.
- The three-zone trained FNO models did not outperform PID or ROM-CEM in the frozen 12-scenario study. Lower energy in poorly tracking runs must not be described as an efficiency gain.
- The study uses 32×32 scoring grids, a single reference ramp, a single CEM budget and one Windows RTX 5080 host. The sensor-stress category has only two scenarios, so no interval is reported for it alone.
- Full-state oracle forecast results and partially observed closed-loop results have different information conditions. They are reported separately.
- Runtime checks validate version, clock domain, state age, sequence, deadlines, units and power/slew constraints. A timed-out GPU call can continue in the background; there is no hard-real-time or safety certification.
- The optional process-selector demonstration uses a local Python pipe, not a real controller transport. Its child-process startup is completed before sampling, and its tests do not establish worst-case communication latency. Frozen performance measurements used the original thread path.
- No physical plant, PLC, industrial sensor, Jetson, NPU or other edge platform was tested. The examples must not control live equipment.
