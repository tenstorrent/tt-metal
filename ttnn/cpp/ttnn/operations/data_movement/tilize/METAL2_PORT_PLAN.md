# Port Plan — `tilize` (Phase 1: sharded factories)

## Scope and phasing

`TilizeDeviceOperation` has 5 factories in its `program_factory_t` variant. Per the framework adapter at `mesh_device_operation_adapter.hpp`, variant arms are dispatched independently — a single op can mix `ProgramDescriptorFactoryConcept` and `ProgramSpecFactoryConcept` factories. This unlocks incremental porting.

This port proceeds in 3 phases:

| Phase | Factories | Kernel forks needed | Status |
|---|---|---|---|
| 1 | `TilizeMultiCoreShardedProgramFactory`, `TilizeMultiCoreWidthShardedProgramFactory` | `writer_unary_sharded_metal2.cpp` (new), `tilize_metal2.cpp` (new); reuses `reader_unary_sharded_metal2.cpp` from s2i port | **DONE — builds green** |
| 2 | `TilizeSingleCoreProgramFactory`, `TilizeMultiCoreDefaultProgramFactory` | `reader_unary_stick_layout_split_rows_{single,multi}core_metal2.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`; reuses `tilize_metal2.cpp` from Phase 1 | **DONE — pending build/test** |
| 3 | `TilizeMultiCoreBlockProgramFactory` | n/a (deferred) | **DEFERRED — see below** |

Phase 1 lands first so the user can build/test the smallest-surface change before further phases propagate to interleaved data movement.

## Phase 1 details

Both sharded factories are structurally identical: same reader, writer, compute, same DFB topology (input borrowed, output borrowed). The only meaningful difference is the work split rule selected upstream by `select_program_factory` based on sharding mode (height-wise vs. width-wise shard). The factories' bodies match exactly except for the `program_id` string.

### Kernel surface

- **Reader** (`reader_unary_sharded_metal2.cpp`) — reused from `sharded_to_interleaved` port. Already-forked in `eltwise/unary/device/kernels/dataflow/`. Uses `dfb::shard` (PRODUCER) and `args::num_tiles_per_core`.
- **Writer** (`writer_unary_sharded_metal2.cpp`) — newly forked in `data_movement/sharded/device/kernels/dataflow/`. Single line of CB-wait logic. Uses `dfb::out` (CONSUMER) and `args::num_units`. Trivial.
- **Compute** (`tilize_metal2.cpp`) — newly forked in `ttnn/cpp/ttnn/kernel/compute/`. Bindings: `dfb::src` (CONSUMER), `dfb::dst` (PRODUCER); CTAs `args::per_core_block_cnt`, `args::per_core_block_tile_cnt`. The `compute_kernel_lib::tilize<>` template accepts DFBs as uint32_t template arguments via `DFBAccessor`'s constexpr `operator uint32_t()` — confirmed by the `tilize_helpers.hpp` documentation showing `dfb_in`, `dfb_out` template params.

### DFB topology

Both factories use `borrowed_from` for both input AND output DFBs — the input tensor's shard is mapped into `SRC_DFB`, and the output tensor's shard is mapped into `OUT_DFB`. This is the same pattern as `sharded_to_interleaved`'s SRC_DFB but extended to the output side. Legacy: `cb_src0.buffer = src_buffer` AND `cb_output.buffer = dst_buffer`. The `borrowed_from` mechanism replaces both bindings.

### Carry-over from `sharded_to_interleaved`

- `Tensor::mesh_tensor()` bridge for `TensorArg::tensor` — applied preemptively.
- Dead-RTA-slot check — not applicable (this op has no RTA legacy dead slots).
- `borrowed_from` for sharded input — applied.

### Phase 1 friction (anticipated)

None expected. The pattern is the proven s2i pattern extended to a paired-borrowed-DFB topology. Build errors would most likely be:

- `DFBAccessor` not constexpr in template position — *resolved*: confirmed `operator uint32_t() const noexcept` is constexpr in `dataflow_buffer.h:53`.
- `compute_kernel_lib::tilize<>` template signature mismatch — *resolved*: signature inspected at `tilize_helpers.hpp:153`.

## Phase 2 details (completed)

The Phase 2 factories ported as written. Key restructuring uncovered while building Phase 2:

### `per_core_block_cnt` moved from CTA to RTA in `tilize_metal2.cpp`

Legacy `tilize.cpp` had `per_core_block_cnt` as CTA[0]. In Metal 2.0, the DFB invariant ("a local DFB has exactly one producer kernel and one consumer kernel" — `dataflow_buffer_spec.hpp:46`) means a single DFB cannot have multiple compute KernelSpec consumers. If `per_core_block_cnt` stayed a CTA, multi_core_default (which has full + cliff cores with different block counts) would need two compute KernelSpecs both consuming from SRC_DFB → invariant violation.

**Fix**: move `per_core_block_cnt` to a per-core RTA in the kernel. A single compute KernelSpec serves all cores; the framework dispatches the right block count per node at runtime. Phase 1 sharded factories also adopted this change (uniform RTA value across cores) for consistency, but the actual driver was Phase 2.

### Phase 3: multi-core-block — DEFERRED

`TilizeMultiCoreBlockProgramFactory` selects on the WH-tilize block code path with 4 distinct compute kernel CT-arg signatures (one per core-range type: full, cliff-row, cliff-col, cliff-row-col), plus 2 distinct DFB sizes across those ranges. Two structural blockers prevent a clean Metal 2.0 port:

1. **The compute kernel's `block_size_row` is a template parameter** of `compute_kernel_lib::tilize<block_size_row, ...>` — a constexpr CTA, not a runtime value. The four core ranges have different `block_size_row` values, so they need four KernelSpecs. The `per_core_block_cnt`-as-RTA trick used in Phase 2 doesn't help here because `block_size_row` is a template arg, not a function arg.

2. **The DFB invariant rejects four compute consumers on one DFB.** Even if the kernel could be templated differently per core range, the four KernelSpecs would all need to consume from SRC_DFB → invariant violation. The clean fix would be 4 DFB pairs (8 DFBs total), each with its own producer/consumer pair — but this also requires forking the reader and writer KernelSpecs four times since each DFB has exactly one producer/consumer kernel pair too. Total: ~20 kernel/DFB resources for one factory. Not a clean port.

**What unblocks Phase 3:**

- Add a runtime-block-width variant to `compute_kernel_lib` (e.g., `compute_kernel_lib::tilize_runtime<dfb::src, dfb::dst>(per_core_block_cnt, block_size_row)`). Then the same RTA-based pattern applied in Phase 2 works for the 4-way split.

  *Or:*

- Relax the DFB "one producer, one consumer" invariant for the case of disjoint-node WUs. The legacy code already implements this correctly (per-range CBs); the Metal 2.0 invariant is stricter than the underlying hardware requires.

The framework adapter (`mesh_device_operation_adapter.hpp:204-213`) dispatches each `program_factory_t` variant arm independently against its concept, so leaving `TilizeMultiCoreBlockProgramFactory` on `ProgramDescriptorFactoryConcept` while the other 4 are on `ProgramSpecFactoryConcept` is a valid mixed state. No code changes needed to defer.

The Phase 3 kernel forks (`tilize_wh_metal2.cpp`, `writer_unary_interleaved_start_id_wh_metal2.cpp`, `reader_unary_pad_multicore_both_dims_metal2.cpp`) were written speculatively before the structural issues surfaced; they remain in the tree as starting points for the eventual port.

## Doc-evolution suggestions

- `borrowed_from` for **output** DFBs (in addition to input DFBs) should appear in the migration guide's examples. The guide currently leads with input borrowing; this op is a witness for symmetric (input + output) borrowing where both tensors are sharded and L1-resident.
- Confirming the DFBAccessor's constexpr-as-template-arg path was non-obvious from the migration guide. Suggest a sentence in the "DFB → Direct use in LLK compute APIs" section: "`DFBAccessor`'s constexpr `operator uint32_t()` also makes it usable as a `uint32_t` non-type template argument, so existing template-on-CB-index APIs (e.g. `compute_kernel_lib::tilize<...>`) accept DFBs unchanged."
