# AGMM configuration registry

This directory contains the native runtime boundary for a future certified
configuration registry for `all_gather_minimal_matmul_async`. It is deliberately
separate from the one-chip `matmul`/`linear`/`addmm` registry: it has its own
configuration switch (`agmm_registry_mode`), compact key and replay ABI,
compiled table, compatibility contract, circuit breaker, and telemetry.

The production table is empty and the mode defaults to `Off`. `Off` bypasses
lookup, request construction, attestation, telemetry, and the selected-call
guard. Empty-table `Shadow` and `On` preserve the legacy configuration and
launch. An explicit program configuration or compute-kernel configuration is
ineligible and remains observable in `Shadow`; trace capture, including an
unknown trace state, is also ineligible. These preflight reasons intentionally
precede `empty_registry`.

## Exact native contract

The key is an immutable, bounded POD. It distinguishes:

- exact local logical and padded tensor shapes, dtype, layout, 32x32 tile
  metadata, full memory-configuration digest, and tensor-topology digest for
  every input and optional tensor;
- effective logical and padded M/K/N plus batch;
- requested output dtype, layout, tile, memory-configuration digest, and
  output-topology digest;
- TP and FSDP effective topology, ordered mesh, ring sizes, axes, links,
  workers, buffers, chunking, semaphore counts (never semaphore addresses),
  barrier/persistent-buffer presence, transpose, SwiGLU, activation, and exact
  IEEE-754 ternary-scalar bits;
- architecture, board capability, device/mesh/grid dimensions, ordered-mesh
  digest, fabric topology digest, and runtime-capability digest.

The replay descriptor contains only `MinimalMatmulConfig` and the complete
`DeviceComputeKernelConfig`. Materialization validates schema/ABI, default tile
geometry, exact workload consistency, grid bounds, block divisibility, and
destination-register constraints before the public operation is dispatched.
Any validation or materialization failure falls back and opens the per-registry
circuit breaker.

The evidence source schema used by the exporter may evolve independently; the
native compact key and replay schemas start at version 1.

## Attestation boundary

Promotion is intentionally blocked today. Existing reviewed public APIs expose
pieces of tensor distribution, but not a single canonical preimage covering
ordered mesh coordinates, per-device capability/harvest state, fabric routing,
and runtime identity. `production_attestation` therefore returns
`UnsupportedAttestation`; it does not fill missing fields with zeroes or inferred
values. A populated table must not be emitted or enabled until that upstream API
exists, the producer derives the identical canonical preimage independently,
and compatibility digests are generated into the runtime build.

## One-shot execution

`Shadow` can record an exact would-hit but never materializes or applies a
recipe. `On` resolves and materializes at most once. Once a selected public
launch begins there is no baseline retry path: an execution exception propagates
with existing operation semantics, opens the circuit breaker, and does not
increment `completed_hits`. `selected_hits` and `completed_hits` are separate so
rollout tooling cannot mistake an attempted recipe for a successful one.
