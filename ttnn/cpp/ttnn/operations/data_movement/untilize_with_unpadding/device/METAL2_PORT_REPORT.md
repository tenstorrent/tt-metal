# Metal 2.0 Port Report — untilize_with_unpadding (single-core factory)

Status: **PORTED** (single-core factory only; not built — this worktree has no build dir).

## TTNN ProgramFactory
- **Concept realized**: `MetalV2FactoryConcept` (`ProgramSpecFactoryConcept`). `UntilizeWithUnpaddingSingleCoreProgramFactory::create_descriptor` → `create_program_spec`, returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`.
- **Device-op-class edits**: none required.
  - `select_program_factory` returns the factory by value; the variant routing in `mesh_device_operation_adapter` dispatches on the concept, so the only change to route to the new path is the factory itself swapping `create_descriptor` → `create_program_spec`. No edit to `untilize_with_unpadding_device_operation.cpp` / `.hpp`.
  - Custom `compute_program_hash`: none (default reflection hash) — nothing to delete.
- **Pybind entry points removed**: none. Grepped the op's pybind/nanobind surface for `create_descriptor` / `create_program` — no exposure of this factory's entry point.
- **Open items (concept fit)**: the device-op's other 5 factories stay on the legacy `ProgramDescriptorFactoryConcept`. The `program_factory_t` variant is mixed-concept and the framework dispatches per-factory, so the op keeps building/running. Those are the remaining work for a future pass.

## Handoff points
- **No worked examples on this branch.** PORT_INSTRUCTIONS.md described worked `rand` / `matmul` Metal 2.0 ports on branch `dgomez/rand-metal2`, but this worktree is on `worktree-agent-...` off a commit where NO in-tree factory uses `create_program_spec` / `device_operation::ProgramArtifacts` (grep returned zero). The Metal 2.0 *framework* headers (`metal2_host_api/*`, `ttnn/metal2_artifacts.hpp`, adapter `ProgramSpecMeshWorkloadFactoryAdapter`, device-side `dataflow_buffer.h` / `tensor_accessor.h` / `kernel_args.h`) ARE all present and complete. The port was written directly against those headers + the recipe/patterns docs. Flagging the missing exemplars so the invoker can confirm the branch is the intended one before building.

## Successes
- **Caution: Modifying a shared dataflow kernel** (patterns catalog) fired correctly. The reader (`reader_unary_interleaved_start_id.cpp`, ~12 consumers) and compute (`untilize/.../compute/untilize.cpp`, ~9 consumers) are heavily shared; both were FORKED to `_m2` copies inside this op's `device/kernels/` tree rather than edited in place, so no sibling op breaks. The writer (`writer_unary_unpad_dims_split_rows.cpp`, this op's dir, single consumer) was ported in place.
- **Pass DFB handles directly** pattern: the compute kernel feeds `dfb::in`/`dfb::out` straight into `compute_kernel_hw_startup(...)` and `compute_kernel_lib::untilize<per_core_block_tile_cnt, dfb::in, dfb::out, ...>(...)` (template-parameter position), relying on the constexpr `DFBAccessor::operator uint32_t()`. No `.id` extraction.

## Friction
- **Gap — fork location for cross-op kernels.** The patterns catalog says fork "alongside the original" (`*_metal2.cpp` next to the legacy file), but the recipe's hard scope fence is "stay within `ttnn/cpp/ttnn/operations/<op>/`". For cross-op shared kernels these conflict (the original lives in another op's dir). Resolved by placing the forks inside THIS op's `device/kernels/` tree with an `_m2` suffix and pointing the factory at them. The docs should state explicitly where a cross-op fork lands when the porter may not write outside the op dir.
- **Confusion — writer CTA slot 1.** The legacy host emitted `unpadded_stick_size` as writer CTA slot 1, but the kernel reads only CTA slot 0 (`FLOAT32_DTYPE`) and `TensorAccessorArgs<2>()`. The slot-1 value is dead. Dropping it (no named CTA) is behavior-preserving; documented in the plan's Dropped Plumbing.

## Open items for downstream
- **Fake-CB self-loops**: none. Both DFBs (IN, OUT) have real producer/consumer pairs (reader→compute, compute→writer).
- **Compute-kernel pointer escape valve**: none. The compute kernel needs no raw L1 base address; the writer's only base-pointer read is `cb_out0.get_read_ptr()` on its own DFB (`CoreLocalMem` over the DFB's L1) — framework-managed, refreshed per execution, no smuggled pointer.
- **Cross-op kernel forks (sunset checklist):**
  - `reader_unary_interleaved_start_id.cpp` — FORKED to `untilize_with_unpadding/device/kernels/dataflow/reader_unary_interleaved_start_id_m2.cpp`. Legacy original (`eltwise/unary/device/kernels/dataflow/`) unchanged; ~11 other consumer op dirs remain unmigrated. Delete the fork only when this op no longer needs it AND the shared original is itself Metal-2.0-ported.
  - `untilize/device/kernels/compute/untilize.cpp` — FORKED to `untilize_with_unpadding/device/kernels/compute/untilize_m2.cpp`. Legacy original unchanged; ~8 other consumer op dirs remain unmigrated.
- **Sibling factories (carry-over):** the 5 remaining factories in this device-op are candidates for the same treatment in a follow-up pass; several share these same reader/compute kernels, so the forks created here can be reused.
- **Test coverage note:** not built or run (no build dir per instructions). Verification (gtests + pytests at `tests/ttnn/unit_tests/operations/data_movement/` and under the simulator) is required before merge.

---

# Metal 2.0 Port Report — untilize_with_unpadding (multi-core interleaved factory)

Status: **PORTED** (multi-core interleaved factory — the default-selected non-sharded multicore
path; not built — this worktree has no build dir).

## TTNN ProgramFactory
- **Concept realized**: `MetalV2FactoryConcept`. `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_descriptor` → `create_program_spec`, returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. Header updated (`tt-metalium/program_descriptors.hpp` include → `ttnn/metal2_artifacts.hpp`; return type `ProgramDescriptor` → `ProgramArtifacts`).
- **Device-op-class edits**: none. `select_program_factory` returns this factory by value (unchanged); the adapter dispatches per-factory on the concept. No custom `compute_program_hash` to delete.
- **Pybind entry points removed**: none — grepped `untilize_with_unpadding_nanobind.{cpp,hpp}`; no `create_descriptor`/`create_program` exposure for any factory.

## Kernels (multi-core)
- **reader** — REUSES `reader_unary_interleaved_start_id_m2.cpp` (the cross-op `_m2` fork already created for single-core). The multicore reader emits `{src, num_tiles_per_core, tile_start_id}` → the m2 reader's `num_pages`/`start_id` named RTAs + `ta::src`. No new fork.
- **compute** — REUSES `untilize_m2.cpp` (the cross-op `_m2` fork already created for single-core). Multicore CTAs `{nblocks_per_core, num_tiles_per_row}` map onto the m2 compute's `per_core_block_cnt`/`per_core_block_tile_cnt`; cb ids become `dfb::in`/`dfb::out`. No new fork.
- **writer** — `writer_unary_stick_layout_split_rows_multicore.cpp` ported **in place** (op-local, single consumer = this factory). dst addr → `ta::dst`; cb 16 → `dfb::out`; FLOAT32_DTYPE/unpadded_X_size → named CTAs; padded_X_size/start_stick_id/n_block_reps → named RTAs; per-core block-rep 5-tuples → runtime varargs (`get_vararg`).

## Successes
- **Multi-group work split → multi-KernelSpec / multi-WorkUnitSpec** (matmul-multicore exemplar shape) applied cleanly: full + cliff compute are two KernelSpecs in two WUs (`uwu_full`, `uwu_cliff`); reader + writer are members of BOTH WUs (so their derived node set = all_cores). The per-group block count stays a CTA — no CTA→RTA demotion.
- **Cross-op fork reuse**: the single-core port's `reader_*_m2.cpp` and `untilize_m2.cpp` forks were directly reusable by the multicore factory (same kernel sources, same named bindings). No additional cross-op kernels touched.
- **Varargs** modeled the writer's variable-length per-core block-rep tuples faithfully (the kernel already bounds its read with the `n_block_reps` named RTA), reusing the `get_vararg` mechanism proven by slice's m2 reader.

## Friction / Open items for downstream
- **BLOCKER-adjacent — deprecated API is the only fit.** The writer needs a DIFFERENT number of runtime varargs per core. The non-deprecated scalar `KernelAdvancedOptions::num_runtime_varargs` only supports a UNIFORM count across all of a kernel's nodes; the per-node-varying case requires `num_runtime_varargs_per_node`, which is marked `[[deprecated]]` ("will be removed once existing uses are refactored"). This port is (per grep) the FIRST in-tree use of that field. It compiles clean because the repo sets `-Wno-deprecated-declarations`. Surfacing precisely so the API owners can decide: either (a) keep a supported per-node-varying-vararg mechanism, or (b) provide guidance to pad every core to the max count via the scalar (inert padding — the kernel reads only `n_block_reps*5` varargs). I chose the deprecated per-node field over padding because padding adds dispatch-buffer bloat and obscures the real per-core count. File: `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp` (the `writer.advanced_options.num_runtime_varargs_per_node[...]` line).
- **Vararg use (report-required)**: writer runtime varargs retained (genuinely variable-length, loop-indexed) — this is the sanctioned vararg case, not a positional-RTA carry-over.
- **Cross-op kernel forks (sunset checklist)**: unchanged from single-core — `reader_unary_interleaved_start_id_m2.cpp` and `untilize_m2.cpp` remain forks; now used by BOTH the single-core and multi-core m2 factories. Delete only when this op no longer needs them AND the shared originals are themselves m2-ported.
- **Compute-kernel pointer escape valve**: none. Writer's only base-pointer read is `cb_out0.get_read_ptr()` on its own DFB (framework-managed). No smuggled addresses.
- **Remaining factories**: 4 of 6 (sharded / col-interleaved / block-interleaved / nd-sharded) stay on the legacy concept; the op keeps building/running (mixed-concept variant, per-factory dispatch).
- **Test coverage**: not built or run (no build dir). gtests + pytests + simulator verification required before merge.

---

## Block-interleaved factory port

Status: **PORTED** (multi-core block-interleaved factory; not built — no build dir in this worktree).

### TTNN ProgramFactory
- **Concept realized**: `MetalV2FactoryConcept`.
  `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create_descriptor` → `create_program_spec`,
  returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. Header updated
  (`tt-metalium/program_descriptors.hpp` → `ttnn/metal2_artifacts.hpp`; return type `ProgramDescriptor` →
  `ProgramArtifacts`).
- **Device-op-class edits**: none. The factory is selected by value in `select_program_factory` and listed in
  `program_factory_t`; the adapter dispatches per-factory on the concept, so swapping the entry point is the
  only change. No custom `compute_program_hash`. No pybind exposure of this entry point.
- All host scalar/shape arithmetic (the full `split_blocks_for_tilize_wh` destructuring,
  `total_tiles_per_row`, `el_size`/`padded`/`unpadded` row bytes, `third_dim`, the per-core RTA loop with
  `single_block_size_row_arg`/`col_arg`/`sub_block` selection and the `start_row_id`/`start_column_id`/
  `tile_start_id` bookkeeping) is copied **verbatim**. Only resource declaration changed.

### Kernels — ported vs forked (all three FORKED)
| role | legacy source | fork (new path, all in this op's `device/kernels/`) | reason |
|------|---------------|------|--------|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | `.../untilize_with_unpadding/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_m2.cpp` | shared (lives outside op dir; also used by untilize) |
| writer | `.../untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | `.../writer_unary_stick_layout_wh_multicore_m2.cpp` | op-local BUT also referenced by `untilize/.../untilize_multi_core_block_program_factory.cpp:196` (still legacy) → forked so untilize keeps its legacy copy untouched |
| compute | `untilize/device/kernels/compute/untilize_wh.cpp` | `.../untilize_with_unpadding/device/kernels/compute/untilize_wh_m2.cpp` | shared (lives outside op dir) |

Kernel port mechanics (logic/loops/`#ifdef`s unchanged, only access mechanism):
- **reader**: `TensorAccessorArgs<3>()` + `src_addr` RTA → `TensorAccessor(ta::src)` (src_addr slot 0 dropped);
  `cb_id_in0=0` → `dfb::in`; `get_tile_size` → `get_local_cb_interface(dfb::in).fifo_page_size`; CTAs
  (num_tiles_per_2d, third_dim, total_tiles_per_row) and RTAs (start_id, single_block_size_row_arg,
  single_block_size_col_arg) → named `get_arg(args::...)`. BACKWARDS `#ifdef` preserved.
- **writer**: `TensorAccessorArgs<4>()` + `dst_addr` RTA → `TensorAccessor(ta::dst)` (dst_addr slot 0 dropped);
  `cb_id_out0=16` → `dfb::out`; kept `api/core_local_mem.h` (`CoreLocalMem<uint32_t>`); CTAs (total_num_rows,
  third_dim, tile_height, unpadded_X_size) and RTAs (width_size + the in-loop start_row_id, start_column_id,
  single_block_size_row_arg, single_block_size_col_arg, sub_block_width_size, single_sub_block_size_row_arg)
  → named args. The in-loop RTAs were the SAME positional slots re-read each `third_dim` iteration; with named
  args they are read once per iteration via `get_arg(args::name)` returning the same value — identical behavior.
- **compute**: `c_0/c_16` → `dfb::in/dfb::out` (fed straight into `compute_kernel_hw_startup` and the
  `compute_kernel_lib::untilize<block_size_row, dfb::in, dfb::out, ...>` template params); CTAs
  (block_size_col, block_size_row, third_dim) → named `constexpr` args (constexpr required because
  `block_size_row` is a template argument). `DST_ACCUM_MODE` define + `unpack_to_dest_mode` handled host-side
  per the interleaved exemplar.

### CB-SIZE CONSOLIDATION DECISION (prominent)
Legacy emitted a PAIR of (c_0 input, c_16 output) CBs on each non-empty core sub-region via `push_cb_pair`,
and the per-region `num_tiles` differed:
- `core_range` → `single_sub_block_size`
- `cliff_col_row_core_range` → `single_block_size_cliff_row`
- `cliff_row_core_range` → `single_block_size_cliff_row`
- `cliff_col_core_range` → `single_sub_block_size`

A Metal 2.0 `DataflowBufferSpec` fixes `entry_size`/`num_entries` ONCE per spec (the per-execution
`dfb_run_overrides` does NOT support resizing — only addresses), so one DFB per logical buffer must pick a
`num_entries` that is safe (>=) for ALL emitted regions. **Resolution**: `num_entries` = MAX over the
emitted regions of `{single_sub_block_size, single_block_size_cliff_row}`, computed under the SAME emission
guards (`!core_range.empty()`, `has_cliff_col && has_cliff_row`, `has_cliff_row`, `has_cliff_col`) so an
unused region never inflates the reservation. IN uses `entry_size = input_single_tile_size`, OUT uses
`output_single_tile_size`; both use this shared `max_cb_num_tiles`.

**Footprint/correctness judgment (not a blocker)**: each per-region CB L1 reservation can only INCREASE to the
max, never decrease, so no region under-reserves — correctness is preserved. The footprint increase is bounded
by `(max - per_region) * (input_single_tile_size + output_single_tile_size)` per core on the smaller regions,
and the host-side split itself sizes everything under `cb_block_size_limit = max_l1_size / (in+out tile size)`
(so the max single block already fits L1). I judged this does NOT materially risk correctness or L1 overflow,
hence proceeded with one max-sized DFB rather than treating it as a blocker. If a future requirement needs the
exact per-region sizing back, that would require per-WorkUnit DFB sizing, which the current
`DataflowBufferSpec`/`dfb_run_overrides` API does not provide.

### Compute work-split multiplicity
Preserved the up-to-4-group structure: one compute `KernelSpec` per region (full / cliff-col-row / cliff-row /
cliff-col) with distinct `KernelSpecName`s and its own `{block_size_col, block_size_row, third_dim}` CTAs, each
in its own `WorkUnitSpec` (`uwu_full`/`uwu_cliff_col_row`/`uwu_cliff_row`/`uwu_cliff_col`), guarded by the same
conditions and CTA values as the legacy `push_compute` calls. Reader + writer run on `all_cores`, so they are
members of EVERY `WorkUnitSpec` (Local-DFB rule). A `make_compute` lambda parameterized by unique_id, target
CoreRangeSet, the two block-size CTAs, and the WU name builds the `KernelSpec` and pushes the `WorkUnitSpec`,
generalizing the row exemplar's full/cliff pattern to up to 4 groups. No CTA→RTA demotion.

### Buffer-address / TensorAccessor plumbing
`src0_buffer->address()`/`dst_buffer->address()` RTAs and the `TensorAccessorArgs(...).append_to(...)` CTAs
DISAPPEAR → `TensorParameter` SRC/DST + `TensorArgument{input.mesh_tensor()}` / `{output.mesh_tensor()}` +
per-kernel `TensorBinding`s. The `Buffer* src0_buffer/dst_buffer` locals and the `TT_FATAL(dst_buffer != ...)`
(framework-enforced now) are gone.

### Blockers
- **None.** No missing framework API was hit; the CB consolidation is a clean (and judged-safe) modeling
  decision, not a gap. This factory does NOT use per-node-varying varargs (unlike the interleaved factory's
  writer), so it does not touch the deprecated `num_runtime_varargs_per_node` field.

### Deviations
- CB sizes consolidated to one max-sized IN/OUT DFB pair (documented above) — the only behavioral modeling
  deviation, judged footprint-safe.
- Reader/writer RTA push order: reader pushed before writer (matching the exemplar's KernelRunArgs grouping);
  legacy computed `writer_rt_args` first then pushed reader-then-writer. Net per-core arg sets are identical;
  only the local construction order differs. Comments adjusted to match.

### Cross-op kernel forks (sunset checklist)
- `reader_unary_interleaved_wh_multicore_m2.cpp` — fork of `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp`. Legacy original unchanged.
- `writer_unary_stick_layout_wh_multicore_m2.cpp` — fork of this op's `writer_unary_stick_layout_wh_multicore.cpp`; legacy original retained because `untilize/.../untilize_multi_core_block_program_factory.cpp` still uses it. Delete the fork only once untilize's block factory is m2-ported (it can then share/own the kernel) AND this op no longer needs it.
- `untilize_wh_m2.cpp` — fork of `untilize/.../compute/untilize_wh.cpp`. Legacy original unchanged.

### Test coverage
Not built or run (no build dir). gtests + pytests + simulator verification required before merge.

## Col-interleaved factory port

**STATUS: PORTED** (multi-core COL interleaved factory; not built — this worktree has no build dir).

### Files changed
- `device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.hpp` — swapped `program_descriptors.hpp` include for `ttnn/metal2_artifacts.hpp`; method now `static ttnn::device_operation::ProgramArtifacts create_program_spec(...)`.
- `device/factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp` — rewritten `create_descriptor` → `create_program_spec`. All host scalar/shape arithmetic (`num_blocks`, `split_blocks_for_tilize`, `el_size`, `unpadded_row_size_bytes`, `num_tiles_2d`, `third_dim`, `total_num_rows`, per-core `size_per_row_per_block`/`number_blocks_per_core`) copied verbatim. Resource tail converted to DataflowBufferSpec / TensorParameter / KernelSpec / WorkUnitSpec / ProgramSpec / ProgramRunArgs, mirroring the row-interleaved exemplar.

### Kernels: ported vs forked
- READER — **forked** (shared, lives outside op dir): `device/kernels/dataflow/reader_unary_interleaved_col_multicore_m2.cpp` (NEW), forked from `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_col_multicore.cpp`. Ported to DFB `dfb::in` + `TensorAccessor(ta::src)` + named args; CTAs (num_tiles_per_2d/third_dim/number_blocks_per_core) and RTAs (core_number/tiles_per_row/num_blocks) named; loops/#ifdefs verbatim.
- WRITER — **ported in place** (op-local): `device/kernels/dataflow/writer_unary_stick_layout_col_multicore.cpp`. DFB `dfb::cb_id_out0`, `TensorAccessor(ta::dst)`, named CTAs (total_num_rows/ncores/third_dim/tile_width/unpadded_X_size) + named RTAs (core_number/size_per_row_per_block/blocks_per_core/width_size). `write_block` lambda/pad logic/loops verbatim. Kept `api/core_local_mem.h`.
- COMPUTE — **forked** (shared with untilize): `device/kernels/compute/untilize_w_m2.cpp` (NEW), forked from `data_movement/untilize/device/kernels/compute/untilize_w.cpp`. CB ids → `dfb::in`/`dfb::out`; CTAs (per_core_block_cnt/per_core_block_tile_cnt/third_dim) named; `untilize<1,...>(per_core_block_cnt * per_core_block_tile_cnt * third_dim)` preserved verbatim. Two compute KernelSpecs (COMPUTE_FULL / COMPUTE_CLIFF) preserve the per-group CTA multiplicity; reader+writer are members of both WUs (uwu_full / uwu_cliff) per the Local-DFB rule.

### Metal blocks
None. No missing host/kernel API encountered.

### Deviations / findings (must-read)
1. **Pre-existing off-by-one RTA bug in the legacy descriptor writer (NOT introduced here).** The descriptor `create_descriptor` wrote writer RTAs `{dst_addr(0), i(1), size_per_row_per_block(2), number_blocks_per_core(3), TILE_WIDTH*el_size(4)}` (5 values), but `writer_unary_stick_layout_col_multicore.cpp` read `core_number=arg(1)`, `size_per_row_per_block=arg(3)`, `blocks_per_core=arg(4)`, `width_size=arg(5)`. After the `dst_addr` slot, that is an off-by-one: it read `number_blocks_per_core` as `size_per_row_per_block`, `width_size` as `blocks_per_core`, and `width_size` from an OUT-OF-BOUNDS index 5 (host never wrote it). The original pre-descriptor (PR #17538) `SetRuntimeArgs` layout `{dst_addr, unpadded_X_size, i, size_per_row_per_block, number_blocks_per_core, width_size}` matched the kernel correctly; the #45071 descriptor migration dropped `unpadded_X_size` from the RTAs (folded it into a CTA) without re-indexing the kernel reads. Since Metal 2.0 named bindings cannot reproduce an OOB positional read, this port wires each kernel variable to the host's **intended** value by name (the correct pre-descriptor mapping). This is a behavior *fix* relative to the buggy descriptor variant, surfaced and documented rather than silently propagated. A matching comment lives at the top of the writer kernel.
2. **This factory is dead in dispatch.** `select_program_factory` (device/untilize_with_unpadding_device_operation.cpp:18-63) never returns `UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory` — it routes only to single-core / multi-core-interleaved / block-interleaved / sharded / nd-sharded. The col-interleaved variant is registered in the `program_factory_t` variant but unreachable, which is why finding (1) was never caught at runtime. The port is faithful and ready, but this path is not exercised by current dispatch or tests.
3. **DFB num_entries.** Legacy CBs `c_0`/`c_16` used `total_size = num_tiles_per_col * single_tile_size`; ported to `entry_size = single_tile_size`, `num_entries = num_tiles_per_col` (per rule 4).

### Test coverage
Not built or run (no build dir in this worktree). gtests + pytests + simulator verification required before merge; note the dead-dispatch caveat (finding 2) — exercising this factory requires forcing its selection.

## ND-sharded factory port

**STATUS: PORTED** (multi-core ND-sharded factory; not built — this worktree has no build dir). Behavior-preserving; all host arithmetic copied verbatim.

### TTNN ProgramFactory
- `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory::create_descriptor` (legacy `ProgramDescriptor`) → `create_program_spec` returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. `.hpp` decl + include `ttnn/metal2_artifacts.hpp` updated (dropped `program_descriptors.hpp`).
- No device-op-class edits (variant dispatch is per-factory concept, same as the already-ported single-core / interleaved factories).
- `.cpp` includes: dropped `program_descriptors.hpp`, `tensor_accessor_args.hpp`, `cb_utils.hpp`, `host_api.hpp`, and the unused `untilize/device/untilize_device_operation.hpp`; added the metal2 `program_spec.hpp` / `program_run_args.hpp` + `using namespace tt::tt_metal::experimental;`. Kept `buffer_distribution_spec.hpp`, `ccl/sharding_addrgen_helper.hpp`, `work_split.hpp`, `work_split_tilize.hpp`.

### Kernels — ported vs forked
- **Reader — FORKED** (shared with `untilize/` factories, still legacy):
  `.../untilize_with_unpadding/device/kernels/dataflow/reader_unary_nd_sharded_blocks_m2.cpp` (new), source of fork `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` (untouched). `cb_id_in0` CTA → `dfb::in`; `TensorAccessorArgs<4>()+src_addr` → `TensorAccessor(ta::src)`; `src_addr` RTA dropped (TensorBinding); `start_shard_id` named RTA; `num_tiles_per_input_block`/`num_shards`/`num_cores` named CTAs. `get_tile_size(cb_id_in0)` → `get_local_cb_interface(dfb::in).fifo_page_size` (established m2 pattern; tile CB ⇒ identical value). Kept `ccl/kernel_common/sharding_addrgen.hpp` + `shard_pages` loop.
- **Writer — PORTED IN PLACE** (op-local; only this factory references it):
  `.../untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` (rewritten). `cb_id_out0` → `dfb::out`; both `TensorAccessorArgs` → `ta::dst` then `ta::src`; `dst_addr`/`src0_addr` RTAs dropped (two TensorBindings); `start_shard_id` named RTA; scalar CTAs named; common-arg loop reads `get_common_vararg(i)`. Logic/loops/`div_up` helper unchanged.
- **Compute — FORKED** (shared with `untilize/` factories, still legacy):
  `.../untilize_with_unpadding/device/kernels/compute/untilize_variable_num_blocks_m2.cpp` (new), source of fork `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` (untouched). cb-id CTAs → `dfb::in`/`dfb::out`; `per_core_block_tile_cnt` named CTA; `num_input_blocks_to_process` named RTA. `DST_ACCUM_MODE` define preserved; early-return-on-0-blocks guard preserved.

### Two TensorParameters on the writer (dst + src binding order)
Two `TensorParameter`s declared: `SRC` (`input.tensor_spec()`), `DST` (`output.tensor_spec()`). Reader binds `SRC` only. Writer binds **both**, in order **DST then SRC** — matching the legacy append order (`TensorAccessorArgs(*dst_buffer)` THEN `TensorAccessorArgs(*src0_buffer)`, "for ND sharded input we need info on the input buffer distribution") and the kernel's accessor construction order (`accessor_dst` first, `accessor_src` second) so the auto-injected CTA layout aligns. `tensor_args` = `{SRC, input.mesh_tensor()}, {DST, output.mesh_tensor()}`. The two buffer-address RTAs (dst_buffer, src0_buffer) disappear into these bindings; only `start_shard_id` remains a per-node named RTA.

### Common runtime args handling — VARARGS (rationale)
The writer's legacy `common_runtime_args` (output padded-shape dims, then input padded-shape dims) is **rank-dependent** (count = `2 * tensor_rank`, ≤ 16 after ND→4D squeezing). The kernel reads them in a runtime loop (`for i in [0, tensor_rank): get_common_vararg(i)` / `(i + tensor_rank)`), with `i` a runtime variable — so by the named/vararg test in the patterns catalog this is the honest vararg case, not named args.
- Schema: `writer.advanced_options.num_common_runtime_varargs = 2 * tensor_rank;`
- Values: `writer_run_args.advanced_options.common_runtime_varargs = {out dims..., in dims...}` (broadcast to all nodes).
- Kernel: `get_common_arg_val<uint32_t>(i)` → `get_common_vararg(i)`. The writer declares **no** named common RTAs, so the genfiles `get_common_vararg` base offset is 0 ⇒ index-for-index identical to legacy.
- **Caution (per patterns doc):** varargs retained deliberately because the count is rank-dependent. If `tensor_rank` were fixed this could be fixed named common args, but the kernel's runtime-indexed loop makes varargs the faithful mapping.

### has_compute / DFB producer-consumer edge case
Legacy always pushes reader+writer and gates compute on `has_compute = !compute_core_range.ranges().empty()`. The port mirrors this exactly: one `WorkUnitSpec` (`target_nodes = compute_core_range`) lists `{READER, WRITER}` and appends `COMPUTE` only when `has_compute`. The local IN/OUT DFBs are PRODUCER reader→IN / CONSUMER compute←IN, PRODUCER compute→OUT / CONSUMER writer←OUT.
- **Edge case noted:** if `has_compute` were false, IN would have a producer (reader) but no consumer, and OUT a consumer (writer) but no producer — violating the DFB producer-and-consumer invariant; the spec validator would reject it. In the ND-sharded path `compute_core_range` is exactly `cores_with_data()`, which is non-empty whenever the op runs, so `has_compute` is effectively always true and the degenerate case is unreachable (the legacy op would itself be a no-op). The DFBs and bindings are only assembled consistent with the present kernels (compute pushed iff `has_compute`), so the invariant holds for every reachable input. No structural blocker.

### Deviations / notes
- Unused legacy writer CTAs `output_stick_size` (legacy idx 1) and `input_single_tile_size` (legacy idx 8) are not read by the kernel; retained as named CTAs for fidelity (named-arg resolution is by name, so dead CTAs are harmless and shift no offsets).
- No metal/framework blockers. The common-vararg path (`num_common_runtime_varargs` / `AdvancedKernelRunArgs::common_runtime_varargs` / device `get_common_vararg`) exists and is wired end-to-end (advanced_options.hpp, program_run_args.hpp/.cpp, genfiles.cpp).
- Not built: this worktree has no build dir; not compiled or run.

---

## Sharded factory port

### STATUS
DONE (not built — this worktree has no build dir; not compiled or run). `UntilizeWithUnpaddingMultiCoreShardedProgramFactory::create_descriptor` → `create_program_spec` returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. Header swapped `tt-metalium/program_descriptors.hpp` → `ttnn/metal2_artifacts.hpp`. No edit to `untilize_with_unpadding_device_operation.{cpp,hpp}` (adapter dispatches on the concept). Pybind: none exposed for this factory.

All host scalar/shape arithmetic (num_rows_block, block_row_size, last_block_row_size_unpadded, num_output_rows_unpadded, last_idx/end_core, ntiles_per_batch, aligned_page_size, and the full per-core `!out_sharded` writer-RTA loop with block_start_row_offset/block_start_row_id_offset) copied **verbatim**.

### The 3 kernel-config branches (selected at runtime inside `create_program_spec`)
- **Config A — `out_sharded && !unpad_tensor_w_16`**: compute = untilize; writer = `writer_unary_unpad_batch_rows_sharded.cpp`. DFBs: IN, OUT, SHARDED_OUT.
- **Config B — `out_sharded && unpad_tensor_w_16`**: compute = eltwise_copy (data-type-convert only, skip untilize); writer = `writer_unary_unpad_width_16_sharded.cpp`. DFBs: IN, OUT, SHARDED_OUT.
- **Config C — `!out_sharded`**: compute = untilize; writer = `writer_unary_stick_layout_interleaved_blocks_m2.cpp` (writes to interleaved DST via TensorAccessor). DFBs: IN, OUT (no SHARDED_OUT).

Reader is the same in all three (`reader_unary_sharded_m2.cpp` — push_back resident input tiles into IN).

### Kernels: ported / forked / reused (absolute paths)
- **Reader — FORKED** (shared by ~12 ops): new `.../untilize_with_unpadding/device/kernels/dataflow/reader_unary_sharded_m2.cpp` from `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (legacy untouched). `CircularBuffer cb(cb_id_in0); cb.push_back(...)` → `DataflowBuffer dfb_in(dfb::in); dfb_in.push_back(get_arg(args::num_tiles_per_core))`. cb-id CTA → `dfb::in`.
- **Writer A — PORTED IN PLACE** (op-local; grep confirmed only this factory referenced it): `.../kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp`. `cb_id_untilize_out`→`dfb::out`, `cb_id_out`→`dfb::sharded_out`; `aligned_page_size` named CTA; 6 named RTAs. CoreLocalMem/Noc/endpoints logic unchanged.
- **Writer B — PORTED IN PLACE** (op-local): `.../kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp`. Same dfb mapping; `get_tile_size(cb_id_out)` → `get_tile_size(dfb::sharded_out)` (constexpr free function + implicit `DFBAccessor::operator uint32_t`); 2 named RTAs; all face-copy logic unchanged.
- **Writer C — FORKED** (generic, lives in `ttnn/cpp/ttnn/kernel/dataflow/`): new `.../untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks_m2.cpp` from `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` (legacy untouched). `TensorAccessorArgs<2>()+dst_addr` → `TensorAccessor(ta::dst)`; cb-id c_16 → `dfb::out`; `FLOAT32_DTYPE` named CTA; 9 named RTAs; the template helper `write_tiles_in_block` takes `dfb::out` (implicit conv). The `dst_addr` RTA disappears.
- **Compute untilize — REUSED**: existing `.../kernels/compute/untilize_m2.cpp` (forked for single-/multi-core interleaved). Legacy `untilize.cpp` uses 4 CTAs (`per_core_block_cnt`, `per_block_ntiles`, `src_cb_id`, `out_cb_id`); `untilize_m2.cpp` uses named `per_core_block_cnt`/`per_core_block_tile_cnt` + `dfb::in`/`dfb::out` and the **identical** `compute_kernel_lib::untilize` template params. Reused as-is (configs A, C).
- **Compute eltwise_copy — FORKED** (generic, `ttnn/cpp/ttnn/kernel/compute/`): new `.../kernels/compute/eltwise_copy_m2.cpp` from `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (legacy untouched). `tt::CBIndex::c_0`→`dfb::in`, `c_16`→`dfb::out`; `per_core_tile_cnt` named CTA (host sets it to `num_input_tiles`, matching legacy `compute_args[0] = num_input_tiles`). LLK calls take `dfb::` directly (config B).

### Borrowed DFBs (sharded I/O in L1)
- **IN** (`c_0`): `borrowed_from = SRC` **iff `src_sharded`** (branch preserved exactly — when not src_sharded it is a plain allocated DFB). PRODUCER on reader, CONSUMER on compute — a real producer/consumer pair.
- **OUT** (`c_16`): plain DFB (no borrow). PRODUCER on compute, CONSUMER on writer.
- **SHARDED_OUT** (`c_17`): present **only** when `out_sharded` (configs A, B). `borrowed_from = DST`, `entry_size = aligned_page_size` (NOT a tile size — the kernel advances the write ptr by aligned_page_size), `num_entries = num_output_rows_unpadded`.

Backing L1 addresses for IN/SHARDED_OUT resolve at runtime from the matching `TensorArgument` (SRC/DST); no `dfb_run_overrides` needed.

### SHARDED_OUT self-loop fake-CB workaround (PROMINENT — interim)
**SHARDED_OUT is a one-ended FIFO.** In configs A/B the writer is the *only* kernel touching it: `reserve_back` / `get_write_ptr` (member) / `push_back`. Nothing consumes it — it **is** the sharded output buffer in L1. A Metal 2.0 DFB requires ≥1 PRODUCER and ≥1 CONSUMER binding, so to satisfy the validator SHARDED_OUT is bound as a **SELF-LOOP**: both a PRODUCER and a CONSUMER `DFBBinding` on the **writer**, sharing `accessor_name = "sharded_out"` (one device-side `dfb::sharded_out` handle). This is the [Fake CB → self-loop DFB] interim pattern — a deliberate validator-satisfying white lie, **not** a real FIFO. To be replaced by the forthcoming "local" TensorAccessor variant when it lands. (IN in sharded mode is *not* a fake CB — reader push_back / compute wait_front is a genuine producer/consumer pair.)

### Per-node vs common RTA choice
**Per-node for all kernels** (one `NodeRuntimeArgs` per core), matching the legacy `SetRuntimeArgs(program, kernel, all_cores, args)` / per-core enumeration exactly. The reader args and the `out_sharded` writer args are in fact identical across cores (could be `common_runtime_arg_values`), but per-node is always safe and is the faithful 1:1 of the legacy code; the `!out_sharded` (config C) writer args genuinely vary per core and must be per-node.

### Work units / local-DFB rule
One `WorkUnitSpec` (`uwu_sharded`, `target_nodes = all_cores = shard_spec.grid`) listing `{READER, WRITER, COMPUTE}`. All kernels run on all_cores, so every DFB's producer & consumer share the same WU — the local-DFB invariant holds trivially.

### CTAs / hw_config
Positional CTAs → named: `aligned_page_size` (config A), `float32_dtype` (config C), `per_core_block_cnt`/`per_core_block_tile_cnt` (untilize), `per_core_tile_cnt` (eltwise_copy). CB-index CTAs became `dfb::` handles (not named args). Reader/writer `hw_config = DataMovementHardwareConfig{.role = READER/WRITER}`; compute `ComputeHardwareConfig{.fp32_dest_acc_en}` + `unpack_to_dest_mode.insert({IN, UnpackToDestFp32})` iff fp32_dest_acc_en. `DST_ACCUM_MODE` define via `CompilerOptions::Defines` (Int32/UInt32/Float32), preserved verbatim.

Unused-CTA note: config C's legacy writer_ct_args included `output_row_size` (slot 1) which the kernel never read — dropped (named-arg resolution is by name; behavior unchanged). Config B's legacy writer_ct_args included `aligned_page_size` (slot 2) which writer B never read — dropped.

### Deviations / blockers
- **No framework blockers.** All required APIs present: `DataflowBufferSpec.borrowed_from`, self-loop dual-binding (per-kernel accessor-name dedup), `get_tile_size(dfb::)` constexpr free function, `TensorAccessor(ta::dst)`, per-node RTAs.
- Config C `dst_addr` RTA (legacy slot 0) folds into `ta::dst`; the three constant-`1` RTAs map to `batch`/`num_blocks_h`/`num_blocks_w` named RTAs.
- Not built: no build dir in this worktree; not compiled, not run on hardware/sim.
