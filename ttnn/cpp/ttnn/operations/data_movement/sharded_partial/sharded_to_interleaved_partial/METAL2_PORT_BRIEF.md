# Metal 2.0 Port Brief — `data_movement/sharded_partial/sharded_to_interleaved_partial`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single `ShardedToInterleavedPartialProgramFactory::create_descriptor` returning a `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind — all `no`/absent on this op.

## Construct — to do

**Tensor bindings** (per binding):

- **input tensor (sharded)** — **clean (borrowed-memory DFB)** → the input CB `c_0` (`src0_cb_index`) is a globally-allocated CB bound to `input.buffer()` (`device/sharded_to_interleaved_partial_program_factory.cpp:140-147`). Port via `DataflowBufferSpec::borrowed_from` the input `TensorParameter`. The reader kernel only advances the FIFO (`dfb.push_back`); there is no `TensorAccessor` to rewrite.
- **output / cache tensor (interleaved)** — **Case 1** (via `TensorAccessor`) → express as a `TensorParameter` / `TensorBinding`; the writer kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(dst_args, dst_addr)`. Drop the `Buffer*` RTA (`writer_rt.push_back(dst_buffer)`, `_program_factory.cpp:243`) and the `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` CTA plumbing (`:177`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor uses the two-argument form; nothing to drop.

**CB endpoints:** all legal 1:1 — bind normally, no special action:
- `convert_df == false`: `c_0` = reader PRODUCER + writer CONSUMER.
- `convert_df == true`: `c_0` = reader PRODUCER + compute CONSUMER; `c_16` = compute PRODUCER + writer CONSUMER.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** the op owns **none** of its kernels — all four are file-path borrows. A shared kernel's Metal 2.0 rewrite must be adopted by every co-borrower in the same change:
  - `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — broadly shared (~18 co-borrowers); port as its own port-together unit.
  - `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` — in-family (`data_movement/sharded`), ~3 co-borrowers.
  - `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` — in-family (RM path; validate-blocked at runtime, see audit Misc anomalies).
  - `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — shared compute-pool kernel, ~4 co-borrowers (only used when `convert_df`).
- **RTA varargs:** none — name each RTA (fixed distinct field at a constant index). Live TILE writer arg map: `[0]` dst base, `[1]` block_height_tiles, `[2]` block_width_tiles, `[3]` unpadded_block_height_tiles, `[4]` unpadded_block_width_tiles, `[5]` output_width_tiles, `[6]` block_num_tiles, `[7]` start_id_offset, `[8]` start_id_base; reader arg `[0]` num_tiles_per_core.
- **Dead RM path:** `validate` forces TILE input, so the RM writer kernel and RM RTA branch are unreachable — do not invest in porting the RM branch beyond what the shared kernel's own port requires (see audit Misc anomalies).
