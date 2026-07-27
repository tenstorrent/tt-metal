# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/transpose`

> **Scoped subset port.** The op is RED at op level — two of eight factories are gated
> (`TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory`). This brief
> covers the **clean six-factory subset only**; do **not** touch the two gated factories or
> their kernels. Full record in `METAL2_PREPORT_AUDIT.md`.

**In-scope factories (port these six):**
`TransposeCNProgramFactory` · `TransposeHCRMProgramFactory` · `TransposeHCTiledInterleavedProgramFactory` · `TransposeHCTiledProgramFactory` · `TransposeWHProgramFactory` (tiled + row-major) · `TransposeWHShardedProgramFactory`

**Out of scope (gated — leave on the descriptor path):**
`TransposeHCShardedProgramFactory` · `TransposeWHShardedRMProgramFactory` — both use the shared device-op `get_dynamic_runtime_args` hook (`Runtime-args update`) and carry `Is safe to port? = no`.

**Gates cleared (for the six):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); each of the six ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all six).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`.
- **Shared device-op coupling:** `get_dynamic_runtime_args` (declared in `transpose_hc_sharded_program_factory.cpp:432`) stays on `TransposeDeviceOperation`; it returns `{}` for these six factories and services only the two gated ones. Do not remove it while porting the subset.

## Construct — to do

**Tensor bindings** (per binding):

- **CN / HC-RM / HC-Tiled-Interleaved / HC-Tiled / WH (tiled + RM)** — input and output are both **Case 1** (via `TensorAccessor`). Today they arrive through the `Buffer*`-binding form (`emplace_runtime_args(core, {tensor.buffer(), ...})`); express each as a `TensorParameter` / `TensorBinding`, kernel builds `TensorAccessor(tensor::name)`, and the `Buffer*` RTA drops out. Mechanical.
- **WH-Sharded** — input CB `c_0` and output CB `c_16` are **borrowed-memory DFBs** (`.buffer = input/output ...buffer()`). Port via `DataflowBufferSpec::borrowed_from` a `TensorParameter`; these are **clean** (no Case-1/2 raw-pointer work).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already 2-arg.

**CB endpoints** (per `(CB, config)`):

- **CN:** `c_0` → bind reader PRODUCER, writer CONSUMER (plain 1:1).
- **HC-RM:** `c_0` → 1:1 (reader P, writer C).
- **HC-Tiled-Interleaved:** `c_0` → 1:1 (reader P, writer C). `c_1` (padding, present only when `C % tile_height != 0`) → 1:1 (reader P, writer C).
- **HC-Tiled:** `c_0` → 1:1 (reader P, donor unary writer C). `c_1` (scratch, present only when misaligned — e.g. Blackhole DRAM 64B) → **self-loop** (only the reader touches it, via `dfb_scratch.get_write_ptr()`; bind it PRODUCER **and** CONSUMER).
- **WH (tiled):** `c_0` → 1:1 (reader P, compute C). `c_16` → 1:1 (compute P, donor unary writer C).
- **WH (row-major):** `c_0` → 1:1 (reader P, compute C). `c_16` → 1:1 (compute P, writer C). `c_24` (im/tilize) → **self-loop** (produced and consumed within the compute kernel). **Drop dead CB `c_25` (im2)** — allocated at `transpose_wh_program_factory.cpp:203-213` (RM branch, marked `// TODO REMOVE`), referenced by no kernel in this factory (`c_25` is used only under `#ifdef SHARDED`, which this factory never sets). Remove the allocation.
- **WH-Sharded:** `c_0` → 1:1 (donor reader P, compute C, borrowed). `c_16` → 1:1 (compute P, donor writer C, borrowed).

No multi-binding CB anywhere in the subset.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader. Nothing to hunt before binding.
- **Cross-op / shared kernels — port the shared kernel as one unit:**
  - `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (HC-Tiled + WH tiled writer) — broadly shared (~42 host files). Its CB→DFB / named-token rewrite must land for all co-borrowers together.
  - `eltwise/unary/.../reader_unary_sharded.cpp` (WH-Sharded reader) — broadly shared (~17 host files).
  - `data_movement/sharded/.../writer_unary_sharded.cpp` (WH-Sharded writer) — broadly shared (~15 host files).
  - In-family helper `data_movement/common/kernels/common.hpp` (`noc_async_read_sharded`/`noc_async_write_sharded`/`fill_with_val`) — Device 2.0 signatures (`Noc&`, `TensorAccessor&`); passes cleanly with `tensor::name`-built accessors.
- **RTA varargs:** none — name every runtime arg.
