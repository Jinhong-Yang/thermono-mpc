# Model and data artifacts

The source repository omits generated training trajectories and model weights from Git history. A versioned release should carry the six SHA-256-verified `.npz` checkpoints with JSON metadata, the synthetic data manifest and trajectories, and the result archive used by the paper. Verify downloads against the release manifest before running any model.

Until a public release is approved, there is no valid download URL or DOI. The CPU examples run without those assets. Full frozen evaluation requires the artifacts and the exact source commit identified in `docs/benchmark_protocol.md`.
