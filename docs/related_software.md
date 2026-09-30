# Related software and scope

This comparison describes the primary documentation inspected on 2026-09-30.
These libraries were not timed or benchmarked on the ThermoNO-MPC process.
An unmentioned feature is not evidence that a package lacks it.

| Software | Documented focus | Relationship to this package |
|---|---|---|
| [NeuroMANCER](https://github.com/pnnl/neuromancer) | Differentiable programming, learned dynamics, constrained optimization and differentiable predictive control | A broader framework for learned control; ThermoNO-MPC supplies a concrete thermal-field process and reproducible comparison protocol |
| [do-mpc](https://github.com/do-mpc/do-mpc) | Nonlinear robust MPC and moving-horizon estimation | A general model-based MPC framework; the present package fixes a shared thermal observation and command interface for its predictor comparisons |
| [CasADi](https://web.casadi.org/) | Symbolic computation, automatic differentiation and numerical optimization | Infrastructure for formulating and solving optimal-control problems |
| [acados](https://github.com/acados/acados) | Fast nonlinear optimal control and MPC solvers | A solver framework; ThermoNO-MPC's released optimizer comparison uses CEM and SciPy SLSQP |
| [NeuralOperator](https://github.com/neuraloperator/neuraloperator) | Learning neural operators, including FNO models | Operator-learning software; the present package connects full-plan thermal predictions to delayed-command control and scoring |
| [safe-control-gym](https://github.com/learnsyslab/safe-control-gym) | Dynamics and benchmarks for learning/control, with constraints and safety filters | Its MPSC/CBF methods differ from this package's timestamp, version, freshness and actuator command checks |

The supported contribution is the implemented combination of a conservative
thermal-field plant, candidate-conditioned operator prediction, matched CEM
comparisons, command validation and traceable results. No first-use claim for
neural operators in MPC, or superiority to these tools, is made.
