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

**MultiCoreSharded** / **MultiCoreNDSharded** — inventoried below; ported in a later pass (more
complex: borrowed-memory DFBs, producer-only output edge / fake-CB, multiple writer variants).

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

**Ported this pass (both reachable from `select_program_factory`, both clean):**
- **SingleCore** — `use_multicore = false`.
- **MultiCoreBlockInterleaved** — `!enough_space_height`, and the WH block-decision path.

**Capitulated / deferred (left on legacy `ProgramDescriptorFactoryConcept`; variant supports mixed
concepts, so the op still builds and runs):**
- **ColInterleaved — CAPITULATED.** Two blockers: (1) `select_program_factory` never returns this
  factory — it is **unreachable** dead code; (2) a **pre-existing host/kernel RTA index mismatch** in
  `writer_unary_stick_layout_col_multicore.cpp`: the kernel reads RTA slots `0,1,3,4,5` (skips slot
  2, reads out-of-bounds slot 5) while the host supplies only slots `0..4`. A faithful positional→
  named conversion cannot reproduce an out-of-bounds read, and the recipe forbids "fixing" the legacy
  kernel during a port. Routed to the report (Handoff points / Open items) for the op owner.
- **Sharded + NDSharded — DEFERRED.** Borrowed-memory DFBs (`borrowed_from`), producer-only output
  edge (fake-CB self-loop if validator rejects), multiple runtime-selected writer variants. A larger
  sub-port; deferred to a fresh pass with full inventory. See `METAL2_PORT_REPORT.md` Open items.

**Convenience binding factories:** the reference port uses `ProducerOf`/`ConsumerOf`; the recipe
prefers full `DFBBinding{}` designated-initializers. This port matches the in-repo reference (per the
brief) for consistency. Noted in the report.
