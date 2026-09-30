# Tested hardware and software profile

The frozen v2 synthetic benchmark ran on one Windows workstation with an NVIDIA GeForce RTX 5080 (16,303 MiB reported VRAM), driver 591.86, Python 3.11.9 and PyTorch 2.13.0+cu130. The CUDA FFT forward and inverse operations, gradient calculation, training and model inference were exercised on this host. The CPU installation, package smoke checks and small examples were exercised separately.

The recorded online decision time includes the shared sparse observer, controller computation and software command selection. The configured 0.5 s soft software acceptance deadline is 5% of the 10 s control interval. No equipment-derived rationale was recorded for choosing this value. It does not preempt a running GPU kernel or establish worst-case execution time. The released benchmark tables report measured times for this host and workload only.

No Jetson, NPU, PLC, dedicated controller or physical heater was tested. The synthetic process parameters are not calibrated to manufacturing equipment. For another CUDA host, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and rerun the benchmarks before making performance claims.
