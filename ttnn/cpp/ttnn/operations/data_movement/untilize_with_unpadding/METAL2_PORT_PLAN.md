# Port Plan — `untilize_with_unpadding`

Port plan for `untilize_with_unpadding`, ported from `ProgramDescriptor` (`create_descriptor`) to
Metal 2.0 (`create_program_artifacts`, `MetalV2FactoryConcept`).

Audit: `METAL2_PREPORT_AUDIT.md` (GREEN, all 5 remaining factories). Brief: `METAL2_PORT_BRIEF.md`.
Reference port: `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory`
(`untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`) — already on Metal 2.0; this
port reuses its named DFB/TensorParameter binding style and the `_metal2` kernel-fork convention.

Atomic unit = one ProgramFactory. Factories are ported one at a time; each is a complete sub-port.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor`) for all 5 remaining factories.
- Variants (device-op `program_factory_t`): SingleCore, MultiCoreInterleaved (already ported),
  MultiCoreSharded, MultiCoreColInterleaved, MultiCoreBlockInterleaved, MultiCoreNDSharded.
- Custom `compute_program_hash`: **none** (default reflection-based hash; audit Q5). Nothing to delete.
- Pybind: only the user op function is bound (`untilize_with_unpadding_nanobind.cpp`); no
  `create_descriptor`/factory pybind. Nothing to delete on the device-op / nanobind side.

### Kernels (per factory)

**SingleCore**
| unique_id | source | CTAs | RTAs | config |
|---|---|---|---|---|
| reader | eltwise/unary `reader_unary_interleaved_start_id.cpp` | TA(src) | src0_buffer, num_tiles, start_id=0 | Reader |
| writer | own `writer_unary_unpad_dims_split_rows.cpp` | float32_dtype, unpadded_stick_size(dead), TA(dst) | dst_buffer + 14 scalars | Writer |
| compute | untilize `compute/untilize.cpp` | per_core_block_cnt, per_core_block_tile_cnt, c_0, c_16 | — | Compute (fp32_dest_acc_en, unpack_to_dest_mode) |
- CBs: c_0 (in, num_tiles_per_block tiles), c_16 (out, num_tiles_per_block tiles).
- Work split: single core (`{0,0}` or first of `sub_core_grids`).

**MultiCoreBlockInterleaved**
| unique_id | source | CTAs | RTAs |
|---|---|---|---|
| reader | eltwise/unary `reader_unary_interleaved_wh_multicore.cpp` | num_tiles_2d, third_dim, total_tiles_per_row, TA(src) | src_addr, start_id, single_block_size_row, single_block_size_col |
| writer | own `writer_unary_stick_layout_wh_multicore.cpp` (**shared w/ untilize op → fork**) | total_num_rows, third_dim, TILE_HEIGHT, unpadded_X_size, TA(dst) | dst_addr, width_size, start_row_id, start_col_id, sbs_row, sbs_col, sub_block_width_size, ssbs_row |
| compute (×up to 4) | untilize `compute/untilize_wh.cpp` | block_size_col, block_size_row, third_dim | — |
- CBs: c_0/c_16 pair emitted per non-empty sub-region (full, cliff-row, cliff-col, cliff-col-row).
- Work split: `split_blocks_for_tilize_wh` (full + up to 3 cliff compute KernelSpecs).
- **Hazard:** legacy pushes `buffer->address()` directly into the RTA list (silent-wrong-on-cache-hit).

**MultiCoreColInterleaved**
| unique_id | source | CTAs | RTAs |
|---|---|---|---|
| reader | eltwise/unary `reader_unary_interleaved_col_multicore.cpp` | num_tiles_2d, third_dim, nblocks_per_core, TA(src) | src_addr, core#, tiles_per_row, num_blocks |
| writer | own `writer_unary_stick_layout_col_multicore.cpp` (not shared) | total_num_rows, ncores, third_dim, TILE_WIDTH, unpadded_X_size, TA(dst) | dst_addr, core#, size_per_row_per_block, blocks_per_core, width_size |
| compute (×up to 2) | untilize `compute/untilize_w.cpp` | nblocks_per_core, num_tiles_per_col, third_dim | — |
- CBs: c_0 (in, num_tiles_per_col), c_16 (out, num_tiles_per_col), all_cores.
- Work split: `split_blocks_for_tilize` (full + cliff compute KernelSpecs).
- **Hazard:** same `->address()`-in-RTA as block.

**MultiCoreSharded** / **MultiCoreNDSharded** — ported in this pass (see Delivered below).

**MultiCoreSharded**
| unique_id | source | CTAs | RTAs |
|---|---|---|---|
| reader | eltwise/unary `reader_unary_sharded.cpp` → fork `_metal2` | DFB(in) | num_tiles_per_core |
| writer (out-sharded) | own `writer_unary_unpad_batch_rows_sharded.cpp` (in-place) | DFB(out), DFB(out_sharded ×self-loop), aligned_page_size | num_unpadded_output_rows, num_padded_tiles_per_batch, num_unpadded_rows_per_batch, padded/unpadded_block_row_size_bytes, batch |
| writer (W=16) | own `writer_unary_unpad_width_16_sharded.cpp` (in-place) | DFB(out), DFB(out_sharded ×self-loop), tile_size_in_bytes | num_unpadded_output_rows, num_padded_tiles_per_core |
| writer (interleaved-out) | shared `kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` → fork `_metal2` | DFB(out), TA(output), float32_dtype | num_rows_block, block_row_size, batch, num_blocks_h/w, last_block_row_size_unpadded, num_output_rows_unpadded, block_start_row_id/offset |
| compute (general) | untilize `compute/untilize.cpp` → reuse `untilize_compute_metal2.cpp` | per_core_block_cnt, per_core_block_tile_cnt | — |
| compute (W=16) | shared `kernel/compute/eltwise_copy.cpp` → fork `_metal2` | per_core_tile_cnt | — |
- DFBs: IN c_0 (**borrowed from input**), OUT c_16 (regular), OUT_SHARDED c_17 (**borrowed from output**, out-sharded only).
- The c_17 edge is **producer-only** (writer pushes the resident output); bound as a writer
  producer+consumer **self-loop** (same accessor name) to satisfy the one-producer/one-consumer
  DFB invariant. (No fake scratch-CB needed — the DM self-loop is accepted by the validator.)

**MultiCoreNDSharded**
| unique_id | source | CTAs | RTAs |
|---|---|---|---|
| reader | sharded `reader_unary_nd_sharded_blocks.cpp` → fork `_metal2` | DFB(in), TA(input), num_tiles_per_input_block, num_shards, num_cores | start_shard_id |
| writer | own `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` (in-place) | DFB(out), TA(output), TA(input), 14 named CTAs (slots 1/8 dead-dropped) | start_shard_id + common varargs (out/in padded shapes) |
| compute | untilize `compute/untilize_variable_num_blocks.cpp` → reuse `_metal2` | per_core_block_tile_cnt | per_core_block_cnt |
- DFBs: IN c_0 (regular — ND input staged through L1, **not** borrowed), OUT c_16 (regular).
- Writer per-block tensor shapes (positional `get_common_arg_val` loop) → **common runtime varargs**
  (`num_common_runtime_varargs`, read via `get_common_vararg`).

### Cross-op / shared kernels (fork on port)
- `reader_unary_interleaved_start_id.cpp` (eltwise/unary) — SingleCore reader. The interleaved
  reference reuses untilize's own already-ported `reader_unary_start_id.cpp` instead (functionally
  identical for tile layout); SingleCore does the same — no new fork needed.
- `reader_unary_interleaved_wh_multicore.cpp` (eltwise/unary) — Block reader → fork `_metal2`.
- `reader_unary_interleaved_col_multicore.cpp` (eltwise/unary) — Col reader → fork `_metal2`.
- `compute/untilize.cpp` (untilize) — SingleCore compute. `untilize_compute_metal2.cpp` fork already
  exists (reference uses it) → reuse.
- `compute/untilize_wh.cpp` (untilize) — Block compute → fork `_metal2`.
- `compute/untilize_w.cpp` (untilize) — Col compute → fork `_metal2`.
- `writer_unary_stick_layout_wh_multicore.cpp` (own, but shared w/ untilize op) → fork `_metal2`.

### Flags
- Dead struct `UntilizeWithUnpaddingMultiCoreSharedVariables` (audit Misc) — not port work; leave.
- `writer_unary_unpad_dims_split_rows.cpp` CTA slot 1 (`unpadded_stick_size`) is emitted by the host
  but never read by the kernel — dead. Dropped in the port (declare only CTAs the kernel reads).

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: per-factory `create_program_artifacts` returning `ProgramArtifacts{spec,
  run_params}`; no op-owned tensors. The device-op `program_factory_t` variant is valid with ported
  and un-ported factories coexisting (framework dispatches per-factory).

## Planned Spec Shape (per factory)
- **SingleCore**: 3 KernelSpecs (reader/writer/compute); 2 DFBs (IN c_0, OUT c_16); 2 TensorParameters
  (input/output); 1 WorkUnitSpec.
- **Block**: reader + writer + 1..4 compute KernelSpecs (one per non-empty sub-region, preserving the
  legacy per-region CTA multiplicity); 2 DFBs; 2 TensorParameters; 1..4 WorkUnitSpecs (reader+writer
  span `all_cores`; each compute in its own sub-region WU). DFBs sized to the max region tile count.
- **Col**: reader + writer + 1..2 compute KernelSpecs (full + cliff); 2 DFBs (num_tiles_per_col);
  2 TensorParameters; 1..2 WorkUnitSpecs.

## Preserved Multiplicity
| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| Block: compute_wh ×(1..4) per sub-region (different CTAs) | compute_full/_cliff_row/_cliff_col/_cliff_col_row | one WU per sub-region | IN (consumer), OUT (producer) bound on each |
| Col: compute_w ×(1..2) (full + cliff) | compute_full/_cliff | full + cliff WU | IN/OUT bound on each |
| SingleCore: none | — | — | — |

## Dropped Plumbing
| legacy form | Metal 2.0 replacement |
|---|---|
| reader `src0_buffer`/`src_addr` RTA + `TensorAccessorArgs(src)` CTA | `TensorParameter INPUT` + `TensorBinding ta::input`; kernel `TensorAccessor(ta::input)` |
| writer `dst_buffer`/`dst_addr` RTA + `TensorAccessorArgs(dst)` CTA | `TensorParameter OUTPUT` + `TensorBinding ta::output`; kernel `TensorAccessor(ta::output)` |
| `cb_id = 0` / `c_16` magic CB indices (CTA + kernel constexpr) | `DFBBinding` IN/OUT; kernel `DataflowBuffer(dfb::in/out)` |
| positional CTAs (`get_compile_time_arg_val(N)`) | named CTAs (`get_arg(args::name)`) |
| positional RTAs (`get_arg_val<uint32_t>(N)`) | named RTAs (`get_arg(args::name)`) |
| writer dead CTA `unpadded_stick_size` (single core) | dropped (kernel never read it) |

## Applied Patterns
- [Multi-variant / work-split multiplicity](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta):
  Block (per-region compute) and Col (full+cliff) keep per-group CTAs as multiple KernelSpecs +
  WorkUnitSpecs — no CTA→RTA demotion.
- [Pass DFB handles directly to LLKs](metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  compute forks pass `dfb::in`/`dfb::out` to `compute_kernel_hw_startup` and `compute_kernel_lib::untilize<...>`.
- [Modifying a shared dataflow kernel](metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel):
  `writer_unary_stick_layout_wh_multicore.cpp` + the eltwise/unary readers + untilize computes are
  cross-op → forked `_metal2`, legacy left for un-ported co-borrowers.

## Delivered this pass / Deferred

**Ported (all reachable from `select_program_factory`):**
- **SingleCore** — `use_multicore = false`.
- **MultiCoreBlockInterleaved** — `!enough_space_height`, and the WH block-decision path.
- **MultiCoreInterleaved** — the height-split interleaved path.
- **MultiCoreSharded** — sharded in/out (out-sharded, W=16, and interleaved-out writer variants).
- **MultiCoreNDSharded** — ND-sharded (`buffer_distribution_spec`) path.

**Capitulated (left on legacy `ProgramDescriptorFactoryConcept`; variant supports mixed concepts, so
the op still builds and runs):**
- **ColInterleaved — CAPITULATED.** Two blockers: (1) `select_program_factory` never returns this
  factory — it is **unreachable** dead code; (2) a **pre-existing host/kernel RTA index mismatch** in
  `writer_unary_stick_layout_col_multicore.cpp`: the kernel reads RTA slots `0,1,3,4,5` (skips slot
  2, reads out-of-bounds slot 5) while the host supplies only slots `0..4`. A faithful positional→
  named conversion cannot reproduce an out-of-bounds read, and the recipe forbids "fixing" the legacy
  kernel during a port. Routed to the report (Handoff points / Open items) for the op owner.

**Convenience binding factories:** the reference port uses `ProducerOf`/`ConsumerOf`; the recipe
prefers full `DFBBinding{}` designated-initializers. This port matches the in-repo reference (per the
brief) for consistency. Noted in the report.
