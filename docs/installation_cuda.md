# PyTorch and CUDA installation

Install a PyTorch build appropriate to your operating system and accelerator using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install `thermono-mpc[torch]`. Verify the actual build and device before interpreting a GPU benchmark:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The frozen study was run on Windows, an NVIDIA GeForce RTX 5080 with 16,303 MiB reported VRAM, driver 591.86, Python 3.11.9 and PyTorch 2.13.0+cu130. CUDA FFT forward/inverse, gradient, training and inference were exercised on that host. CPU inference was also exercised; other GPU models, ROCm, Metal, edge accelerators, Jetson, NPU, PLC and live equipment were not validated.

Measured response times include the shared sparse observer, controller computation and command selection for the frozen synthetic workload. A software deadline check does not make GPU execution preemptible or provide a worst-case real-time bound.
