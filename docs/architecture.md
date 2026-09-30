# Architecture and public contract

`process.ThermalPlant` is the scored synthetic plant. Its thermal field and
zone states are not passed to a controller except through an explicit state
estimate. The process uses a conservative finite-volume discretization and
checks nonlinear solver convergence. `dataset` generates whole trajectories
and hashes each file in a split manifest.

`operator.DirectFNO` accepts the estimated current field, measured zone
temperatures, known material context, and an entire candidate future heater
sequence. It returns all horizon field and zone predictions. It uses replicated
padding to reduce periodic wraparound at the physical edge; padding alone does
not guarantee an exact boundary condition. `physics_loss` computes a separate
finite-volume residual on predicted states. Data-only and physics-informed
variants share the same architecture and supervised labels.

`controllers.ROMPredictor` is a low-order spatial lump model, implemented
without calling the evaluation plant. `controllers.CEMMPC` can use any
predictor implementing `predict(field_K, zones_K, candidates_W, dt_s)`.
It enforces power and slew limits on candidate plans and rejects plans whose
predicted states exceed the configured nominal temperature limits. Its
sequential clipping is not an exact Euclidean projection. If no feasible plan
exists, it reports `INFEASIBLE_CANDIDATES` and leaves fallback to the runtime.

`runtime.CommandValidator` checks version, clock domain, state age, source
identity, sequence, apply cycle, expiry, deadline, finite values, units,
actuator bounds and slew. `RuntimeSupervisor` executes one prediction job at a
time, invalidates old generations, and uses independent PID fallback. The
current boundary is software-only and shares one host clock; it provides no
hardware isolation, process preemption, or hard real-time guarantee.

The controller-to-actuator object contains only heater setpoints and metadata,
not a full thermal field. `simulation.run_episode` explicitly models one-cycle
application delay by applying the previously held input during the current
interval and selecting a validated command for the next interval.

`process_boundary.ProcessCommandBoundary` is an optional local-process
transport example. The prediction side reduces a causal field estimate to one
temperature mean per heating zone. The child command-selector process receives
those means, state metadata and a signed setpoint command through a single
request/response pipe. It applies the same validator and uses its own compact
PID fallback for rejected or missing commands. Only one request is outstanding;
if the child dies or fails to respond, the caller records an emergency fallback
and closes that boundary. The example waits for child startup before sampling
the state so cold-start time is not hidden inside a decision deadline.
After a process restart, the caller must advance the process version or another
trusted generation identifier; otherwise a fresh validator cannot know all
commands accepted by the old process. The fault tests reject a pre-restart
command under an advanced process version.

The 120 frozen closed-loop runs used the original single-worker thread path,
not this process transport. The process example has its own CPU smoke and fault
tests; it is not a measured PLC transport or a safety-isolated control core.
`scripts/runtime_overload_probe.py` separately measures CPU contention against
a software deadline and records queue rejection, without treating a timed-out
worker as preempted.
