# Post-freeze reviewer diagnostics (v0.1.1)

The six frozen checkpoints, protocol and 120 scored episodes are unchanged.
D9 and D10 are additional diagnostics, not replacement benchmark rows.
Stage the v0.1.0 research archive as described in post_freeze_diagnostics.md.

```console
python scripts/reviewer_diagnostics.py --id D9 --device cuda --workers 4
python scripts/reviewer_diagnostics.py --id D10 --workers 4 --frozen-evidence data/generated/research_inputs
```

D9 reads the full D1 candidate records. Since prediction validation requires
finite outputs, infinite saved CEM cost identifies rejection by predicted
solid/zone limits. Every selected sequence is checked against the saved
minimum-cost candidate, and the D1 frozen-action/state replay receipt is checked.
All 1,890 selection calls (362,880 candidate occurrences) enter screening.
For the 1,620 neural calls, ROM CEM is reselected from the **same projected state
and held command**, with a separate deterministic cold-search stream
`scenario.seed + 100000 + cycle`. Both plans are evaluated by the neural model,
ROM, and a clone of the same delay-aligned true state. The original own-plan
clone is read from D1; the counterfactual ROM-plan clone is newly integrated.
Constant targets 0/500/1000/2000 W are ramped through the same slew limiter;
their endpoint responses test monotonicity at each saved state. These are
correlated diagnostic calls, not independent experimental units.

D10 instruments the original two-start SLSQP comparator over all 12 scenarios.
Its first ten N03 selections (20 local starts) are used for one-factor changes
to finite-difference step, objective scaling and variable scaling. Clipping is
only applied to initial guesses, not inside the SLSQP objective. A separate
single-start comparator scales decision variables by 2000 W, uses eps=0.0005
(1 W physical perturbation), ftol=1e-9 and maxiter=80, and preserves the physical
objective and constraints. It remains in the diagnostic script; the released
default SLSQPMPC still reproduces the original implementation.

Exact executed scripts/core modules/configurations and their SHA-256 hashes
are saved under each diagnostic directory. Existing completed outputs are
reused; changed executed sources require a new output directory rather than
silently overwriting a prior source snapshot. D11 matched-grid closed-loop
runs are optional and are not part of this release's evidence.

For an exact earlier execution use its saved source snapshot. The final release
adds the portable `--frozen-evidence` argument; the executed D10 snapshot used
the author's local staged-input location. This path-only change does not change
the optimizer or simulation. Screening/cross-evaluation runs do not use that path.
