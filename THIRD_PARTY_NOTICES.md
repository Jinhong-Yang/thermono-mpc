# Third-party dependencies

The package declares NumPy, SciPy and PyYAML as runtime dependencies and PyTorch as an optional neural-model dependency. Hatchling builds the distribution; pytest and `build` are development tools. Their source code is not copied into this repository. Use each installed distribution under its own license and retain its notices when redistributing a bundled copy.

The local validated environment contained NumPy 2.4.6 (BSD-3-Clause and bundled permissive notices), SciPy 1.17.1 (BSD license classifier and license file), PyYAML 6.0.3 (MIT), PyTorch 2.13.0+cu130 (Apache-2.0 and bundled permissive notices) and Hatchling 1.32.4 (MIT). These are observations from installed package metadata, not a dependency lock or a legal opinion. Check the actual distribution's license files before redistributing binaries.

ThermoNO-MPC uses its own FNO, process, observer and controller implementation. [NeuralOperator](https://github.com/neuraloperator/neuraloperator), [do-mpc](https://github.com/do-mpc/do-mpc) and [acados](https://github.com/acados/acados) are related work, not installed runtime dependencies or vendored source.
