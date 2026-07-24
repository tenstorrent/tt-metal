# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize`

> **Scoped brief (config-scoped gate).** The op is RED at op level: `UntilizeMultiCoreBlockProgramFactory` is gated by the readiness sheet (`Is able to port? = "NO (confer with ops)"`). **This brief covers the clean subset — the other 7 factories — only. Do NOT port `UntilizeMultiCoreBlockProgramFactory` in this pass.** Its kernels (`reader_unary_interleaved_wh_multicore.cpp`, `writer_unary_stick_layout_wh_multicore.cpp`, `untilize_wh.cpp`) are not used by the clean subset, so the subset ports independently. The full record is in `METAL2_PREPORT_AUDIT.md`.

**Factories in scope (7):** `UntilizeSingleCoreProgramFactory`, `UntilizeMultiCoreSubCoreGridsProgramFactory`, `UntilizeMultiCoreParallelizeColumnProgramFactory`, `UntilizeMultiCoreProgramFactory`, `UntilizeMultiCoreNDShardInputProgramFactory`, `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`, `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`.

**Gates cleared (for the subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (these 7 rows = `yes`) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `53e5e16e8d0 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all 7 factories return `tt::tt_metal::ProgramDescriptor` from `create_descriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` / `get_dynamic_runtime_args` · pybind `create_descriptor` — all confirmed `no` in both the sheet and the code.

## Construct — to do

**Tensor bindings** (per binding):

- **Interleaved / ND-sharded input paths** (`single`, `sub_core_grids`, `parallelize_column`, `multi_core` interleaved path, `nd_shard_input`): input `src` and output `dst` — **Case 1** (via `TensorAccessor`). Today the base is delivered as a `Buffer*` slot in `emplace_runtime_args(core, {src0_buffer, ...})` / `{dst_buffer, ...}` and fed into `TensorAccessor(args, addr)` in the reader/writer. Express each as a `TensorParameter`/`TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`, and the `Buffer*` RTA slot + the `TensorAccessorArgs(*buffer).append_to(...)` CTA plumbing both disappear.
- **Identical-shard factories** (`InputAndOutputShardType…Identical`, `InputAndOutputNDShardType…Identical`): input and output — **clean (borrowed-memory DFB)**. The CBs are backed by the shard buffers (`cb_src0.buffer = src0_buffer`, `cb_output.buffer = dst_buffer`). Port via `DataflowBufferSpec::borrowed_from`; no `TensorAccessor` and no `Buffer*` RTA slot (readers/writers take only tile counts).
- **`multi_core` even-sharded path:** input is **clean** (backed CB via `cb_backing_buffer = src0_buffer`); output is **Case 1**. This is a per-config split *within* the one factory — the same input binding is clean under even-sharding and Case 1 under the interleaved / block-reader paths. Handle per code path.
- No **Case 2** (raw-pointer) bindings — nothing needs the `get_bank_base_address` bridge.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` is already the 2-arg form.

**CB endpoints:** all legal 1:1 — nothing special to set.
- Each factory has exactly two CBs: input `c_0` and output `c_16`. Port both as ordinary **1 PRODUCER + 1 CONSUMER** DFBs (`c_0`: reader→compute; `c_16`: compute→writer).
- Identical-shard factories: port `c_0`/`c_16` via `borrowed_from` (backed by shard buffers); still 1P+1C.
- `UntilizeMultiCoreProgramFactory`: the full-core and cliff-core compute kernels cover **disjoint** cores — ordinary 1:1 per node, no multi-binding assignment needed.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no dual-instance same-core-range split in the subset.
- **Cross-op / shared kernels (coupling runs both ways — see the audit's port-together section for the full co-borrower lists):**
  - **Kernels this subset borrows:** `eltwise/unary/.../reader_unary_interleaved_start_id.cpp`, `eltwise/unary/.../reader_unary_sharded.cpp`, `data_movement/sharded/.../writer_unary_sharded.cpp` (each broadly shared — ≈8–10 mainline ops incl. `untilize_with_unpadding`, `tilize`, `transpose`, `typecast`, sharded conversions), and `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` (narrow — `untilize_with_unpadding` only). Their CB→DFB / named-token rewrite is a **single change every co-borrower must adopt together** — port the shared kernel as one unit, not per-op.
  - **This op's OWN kernels borrowed elsewhere (rewriting them breaks these ops unless updated in the same change):**
    - `compute/untilize.cpp` (used by `single`, `sub_core_grids`, `parallelize_column`, shard-identical) → **`data_movement/fold`**, **`data_movement/untilize_with_unpadding`** (3 factories), **`pool/upsample`**.
    - `compute/untilize_wh.cpp` (Block, gated) → `untilize_with_unpadding` (block).
    - `compute/untilize_variable_num_blocks.cpp` (used by `multi_core`, `nd_shard_input`, nd-identical) → `untilize_with_unpadding` (nd_sharded).
    - `dataflow/reader_unary_start_id.cpp` (used by `single`, `multi_core` interleaved) → `data_movement/tilize` (retile).
  - **Strong recommendation:** treat the shared kernels as the unit of migration. At minimum, port untilize together with **`untilize_with_unpadding`** (they share kernels in both directions); coordinate `untilize.cpp`'s rewrite with **fold, untilize_with_unpadding, and upsample** as well. The wider shared-kernel graph (typecast, transpose, tilize, pad, sharded↔interleaved, slice_write, reductions, …) is enumerated in the audit's port-together section.
- **RTA varargs:** none — name every runtime arg (all are fixed distinct fields).
