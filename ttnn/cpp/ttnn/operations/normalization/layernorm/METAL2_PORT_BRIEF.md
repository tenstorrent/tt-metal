# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/normalization/layernorm/`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ (isolated `get_tile_size(cb_id)` holdovers routed to the Device 2.0 track — do **not** absorb them into the port) · Features ✓ · Factory concept available ✓

Single device-op (`LayerNormDeviceOperation`) with two factories, ported together:
- `LayerNormMultiCoreProgramFactory` (`device/layernorm_op_multi_core.cpp`) — interleaved
- `LayerNormShardedProgramFactory` (`device/layernorm_op_multi_core_sharded.cpp` + `device/sharded_layernorm_factory_helpers.{hpp,cpp}`) — sharded

## Plan — factory concept

Implement `ProgramSpecFactoryConcept` · caching strategy `MaximizeCacheReuse` (default) (→ [`port_op_to_metal2_ttnn_factory.md`](port_op_to_metal2_ttnn_factory.md)). 1:1 with the legacy single-program shape; no op-owned resources (the welford reciprocal LUT is caller-passed, not factory-allocated), no escalation. Strict tensor matching.

## Construct — to do

**Tensor bindings** (per binding; classification can differ per factory — see the audit's Per-DeviceOperation table):

- **input (a)** — *interleaved factory:* **Case 1** → re-express `a.buffer()->address()` RTA as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(ta::input)`. *sharded factory:* **clean** (borrowed-memory DFB — CB 0 on `a_buffer`; keep via `borrowed_from`, do not convert).
- **residual (b)** — *interleaved:* **Case 1**. *sharded:* **clean** (borrowed-memory DFB, CB 1).
- **gamma** — **Case 1** in both factories → re-express the `gamma_dram_addr` RTA; kernels build `TensorAccessor(ta::gamma)`.
- **beta** — **Case 1** in both factories (symmetric to gamma).
- **stats** — *sharded post-all-gather only:* **clean** (borrowed-memory DFB, CB 7).
- **recip LUT** — **clean** in both factories (borrowed-memory DFB, CB 25; welford only).
- **output** — *interleaved:* **Case 1** (`output.buffer()->address()` RTA → `ta::output`). *sharded:* **clean** (borrowed-memory DFB, CB 16 / reshard CB 17).

No Case 2 bindings — every access is plain page-by-page `TensorAccessor` iteration.

**Custom hash:** none — `LayerNormDeviceOperation` has no `compute_program_hash` override (the `compute_program_hash` static in `layernorm_nanobind.cpp:253` is a Python test hook calling the framework default; leave it). Nothing to delete.

## Watch for

- **Notable constructs:**
  - **Borrowed-memory DFBs** — sharded factory binds input/residual/stats/output CBs onto `Buffer` memory via `CBDescriptor::buffer`; both factories do so for the recip LUT (CB 25). Port these via `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`. The causal-link gate already classified them clean; do not force them onto `TensorAccessor`.
  - **Aliased CBs (welford-fp32 path)** — single `CBDescriptor` with two `CBFormatDescriptor`s sharing SRAM (primary + `UnpackToDestFp32` alias index c_29 / c_30 / c_31). Port via `DataflowBufferSpec::advanced_options.alias_with`, one DFB per index, mutually named; same `num_entries * entry_size`; bound to the same kernels. Do **not** split into independent DFBs (shared L1 address is load-bearing). Carry the matching `unpack_to_dest_mode[c_29/30/31] = UnpackToDestFp32` on the compute spec. Sites: `layernorm_op_multi_core.cpp:688-693,709-714,734-739,810-815`; `sharded_layernorm_factory_helpers.cpp:1007-1012,1066-1071`.
- **Cross-op / shared kernels:** all kernel `.cpp` are layernorm-owned (in-family); shared headers come from `kernel_lib/`, `kernel/`, and the in-family `kernel_util/`. No cross-family donor — no co-borrower port-together obligation outside the layernorm op itself. Device-2.0-clean.
- **RTA varargs:** `writer_unary_sharded_ln.cpp:38` reads a runtime-known-count `segment_args` block via `get_arg_addr(9)` + counted loop — supported in Metal 2.0; prefer named RTAs unless the runtime-varying-index read is genuinely needed (per kernel-side whitelist rule 4). Non-gating.
