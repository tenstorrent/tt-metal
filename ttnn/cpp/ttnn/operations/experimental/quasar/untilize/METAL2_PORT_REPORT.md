# Metal 2.0 Port Report — `experimental/quasar/untilize`

**Pass 1: `UntilizeSingleCoreProgramFactory` ported (build pending).** create_descriptor → create_program_artifacts.
- Forks: `compute/untilize_metal2.cpp` + `dataflow/reader_unary_start_id_metal2.cpp` (both shared with not-yet-ported factories — legacy retained). Writer `writer_unary_stick_layout_split_rows_single_core.cpp` ported in place. Forked kernels are JIT (no cmake change).
- No device-op edits (no custom hash). Mixed-concept program_factory_t variant (single_core on MetalV2, rest legacy).
- Dropped: buffer-addr RTAs (reader src/writer dst → TensorBindings), TensorAccessorArgs, CB-index CTAs → DFBBindings, positional CTAs → named, dead writer CTA `output_stick_size`.
- Self-audit clean: all factory binding/CTA/RTA names match kernel dfb::/tensor::/args:: tokens; no CircularBuffer/CBDescriptor/positional-CTA/ta:: artifacts.
- Remaining 7 factories enumerated in METAL2_PORT_PLAN.md.

---
## Pass 2: multi_core_parallelize_column factory (build+test pending)
Ported `UntilizeMultiCoreParallelizeColumnProgramFactory` → create_program_artifacts. Clean reader→compute→writer, full+cliff compute (preserved multiplicity), per-core fixed RTAs (no varargs).
- Forks (both shared with multi_core_sub_core_grids): `reader_unary_interleaved_start_id_metal2.cpp` (= uwu's ported reader), `writer_unary_stick_layout_split_rows_interleaved_parallel_columns_metal2.cpp`. Compute reuses `untilize_metal2.cpp` (pass-1 fork). Forked kernels are JIT.
- Dropped: buffer-addr RTAs→TensorBindings, TensorAccessorArgs, CB indices→DFBBindings, positional CTAs→named.
- **⚠ LATENT LEGACY BUG FLAGGED (owner review):** the legacy cliff-core writer passed 7 positional RTAs (an extra `stick_size`/`block_size_nbytes` at index 2) but `writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` reads only 6 — so on a cliff core the legacy mis-aligned `num_tiles_per_core`/`tile_width_size`/`start_stick_id`/`offset_within_stick` (full-core path is correct). Metal 2.0 named args bind by name, so this port emits the cliff writer with the 5 named args the kernel actually reads → corrects the cliff-core behavior (a behavior change vs legacy on cliff cores, per user direction "port cliff as kernel-correct + flag"). Recommend the op owner verify/fix the legacy factory's cliff arg list separately.
- Self-audit clean. Remaining untilize factories: multi_core (multi-source: 3 readers), multi_core_block (untilize_wh), multi_core_sub_core_grids (reuses both pass-2 forks!), + 3 sharded (nd_shard_input, input_and_output_{shard,nd_shard}_identical — check borrowed-mem/scratch first).

---
## Pass 3: multi_core_sub_core_grids factory (build pending)
Ported `UntilizeMultiCoreSubCoreGridsProgramFactory` → create_program_artifacts. **Zero new kernels** — reuses all three pass-2 forks (reader_unary_interleaved_start_id_metal2, writer_..._parallel_columns_metal2, untilize_metal2). Structurally parallelize_column minus the full/cliff split: sub-core-grid core computation (corerange_to_cores + num_cores_to_corerangeset_in_subcoregrids, uniform nblocks_per_core), a single compute KernelSpec, a single WorkUnitSpec over all_cores, and uniform per-core RTAs (no cliff branch).
- Dropped: buffer-addr RTAs→TensorBindings, TensorAccessorArgs, CB indices→DFBBindings, positional CTAs→named.
- .hpp updated to create_program_artifacts. No device-op edit (already in program_factory_t variant + selector; framework auto-detects MetalV2 vs legacy concept per factory).
- Self-audit clean: no create_descriptor/ProgramDescriptor/CBDescriptor/TensorAccessorArgs/ta:: artifacts; all reader(2 RTA)/writer(1 CTA+5 RTA)/compute(2 CTA) names match fork tokens. Factory already in sources.cmake (name unchanged).
- Not python-routable as a unit test (sub_core_grids requires the op be invoked with an explicit sub_core_grids attribute; exercised via demo/integration). 3 forks now cover single_core + parallelize_column + sub_core_grids = 3 of 8 factories.

---
## Pass 6: the 3 sharded factories (build pending) — untilize now 8/8 on Metal 2.0
All three remaining sharded factories ported to create_program_artifacts:
- **input_and_output_shard_type_and_shard_spec_identical**: input & output share a shard spec → BOTH DFBs zero-copy (`borrowed_from` IN/OUT). Reader push-only, compute untilizes in place, writer is a wait/pop handshake. New fork `writer_unary_sharded_metal2.cpp`; reuses reader_unary_sharded_metal2 + untilize_metal2.
- **input_and_output_nd_shard_...identical**: same shape, nd-shard; per-core shard count varies (idle cores → 0, compute fork early-returns). Reuses all forks (reader_unary_sharded_metal2 + writer_unary_sharded_metal2 + untilize_variable_num_blocks_metal2) — zero new kernels.
- **nd_shard_input**: nd-sharded input → row-major output. Normal (non-borrowed) double-buffered DFBs: block reader in, NoC writer out. New forks `reader_unary_nd_sharded_blocks_metal2.cpp` + `writer_unary_stick_layout_split_rows_multi_core_nd_shard_metal2.cpp` (the writer binds BOTH tensors — input for shard-page iteration, output for writes; dead CTAs at idx 1 output_stick_size + idx 8 input_single_tile_size dropped). Reuses untilize_variable_num_blocks_metal2.
- All .hpp → create_program_artifacts. Self-audit clean (no legacy artifacts; all fork tokens match factory bindings; all 8 factories in cmake). **untilize is fully ported: 8/8 factories on Metal 2.0.** Sharded factories are build-validated (need sharded input not in the unit test; exercised via resnet demo / integration). `DataflowBufferSpec.borrowed_from` carries the zero-copy backed-CB pattern.

---
## Pass 5: multi_core factory (build pending) — the default interleaved multi-core path + both sharded modes
Ported `UntilizeMultiCoreProgramFactory` → create_program_artifacts. Three host-selected input modes, all on Metal 2.0:
- (C) **interleaved**: reuses `reader_unary_start_id_metal2` (the single_core fork).
- (A) **block reader** (uneven/DRAM sharding): new fork `reader_unary_sharded_blocks_metal2.cpp`.
- (B) **even sharded** (zero-copy): new fork `reader_unary_sharded_metal2.cpp` (push-only) + input DFB `borrowed_from = INPUT` (DataflowBufferSpec borrowed-memory → the shard's L1 backs the CB; address supplied via tensor_args).
Common kernels: new forks `writer_unary_stick_layout_split_rows_multi_core_metal2.cpp` (dead legacy CTA index 1 `output_stick_size` dropped) + compute `untilize_variable_num_blocks_metal2.cpp` (named CTA per_core_block_tile_cnt + named RTA per_core_block_cnt, idle-core early-return preserved). Full + optional interleaved-cliff compute KernelSpecs, each its own WorkUnitSpec; reader/writer span all cores. Host work-distribution loop (incl. uneven-shard width/height handling) copied verbatim; buffer-addr RTAs → TensorBindings / borrowed_from.
- .hpp → create_program_artifacts. No device-op edit (already in variant + selector). Self-audit clean (no PD/CBDescriptor/KernelDescriptor/TensorAccessorArgs/ta::/cb_id; all reader(per-mode)/writer(6 CTA+5 RTA)/compute(1 CTA+1 RTA) names match fork tokens; factory in cmake).
- Test: added `test_quasar_untilize_multi_core` (interleaved moderate-width multi-row shapes → the model's path). Sharded modes (A/B) are build-validated only (need sharded input, exercised via demo/integration). **5 of 8 untilize factories done.** Remaining: 3 sharded (nd_shard_input, input_and_output_{shard,nd_shard}_identical — check borrowed-mem/scratch first).

---
## Pass 4: multi_core_block factory (build pending) — also CLOSES the pre-existing bug below
Ported `UntilizeMultiCoreBlockProgramFactory` → create_program_artifacts. The WH block path (4 sub-regions: full + col/row/col-row cliffs). **Zero new kernels** — binds uwu's already-m2 WH kernels (`untilize_with_unpadding/.../reader_unary_interleaved_wh_multicore.cpp`, `untilize_wh.cpp`, `writer_unary_stick_layout_wh_multicore.cpp`); the legacy factory already cross-referenced uwu's writer, so pointing reader+compute there too is consistent and resolves the bug (m2 factory + m2 kernels). Mirrors uwu's block_interleaved factory; untilize specifics preserved (tensor tile dims, single row_size_bytes; for plain untilize unpadded_X_size == row_size_bytes).
- **Cliff classification — two fixes.** (1) Use uwu's CoreRangeSet split (`available_grid` = sub_core_grids or full default grid) + `corerange_to_cores(available_grid)` so the per-core RTA loop visits cores in the same order `split_blocks_for_tilize_wh::addCore` built the region sets (correct running offsets). (2) **Classify each core by actual region-set membership** (`cliff_*_core_range.contains(core)`), NOT by reconstructing the region from the linear index. uwu's index heuristic (`(i+1)%(full_cores_per_row+1)==0` → cliff_row) is latently wrong for single-tile-high shapes: when `full_cores_per_col == 0` every core lands in cliff_col, but the modulo still flags one as cliff_row → that core runs COMPUTE_CLIFF_COL while getting cliff_row writer RTAs → writer pops a different tile count than compute pushes and hangs at CWFW (seen on `wide_1tilerow`). Membership matches exactly what the WorkUnitSpecs use to place compute, so RTAs and compute always agree. Fixes the single-tile-high hang (wide_1tilerow) and the cliff-col RTA mismatch.
- **Known shared-kernel limit (NOT this port):** a much taller+wider shape ((1,1,256,4096), num_tiles_per_col==8) trips a compile-time assert inside `untilize_wh` — split_blocks_for_tilize_wh produces a large square block whose `block_width_tiles` exceeds the pack_untilize DEST limit. The assert is unchanged across every host RTA/region variant tried (it's driven by the compile-time block geometry, not host args) and the compute CTA binding is identical to uwu's proven block_interleaved factory, so it reproduces there too. Test covers cliff_row (wide_1tilerow) + small cliff_col (wide_2tilerows, num_tiles_per_col==2), which pass; the pathological 2D-block shape is excluded and flagged for the kernel owner.
- One compute KernelSpec per non-empty sub-region (preserves per-region CTA multiplicity), each in its own WorkUnitSpec; IN/OUT DFBs sized to the max present region (superset of the legacy per-region CB pairs). Per-core RTA loop copied verbatim (buffer-addr RTAs → TensorBindings).
- .hpp → create_program_artifacts. No device-op edit (already in variant + selector).
- untilize's OWN legacy `untilize_wh.cpp` + mixed `reader_unary_interleaved_wh_multicore.cpp` are now unreferenced (dead JIT files; left in place, out of scope to delete).
- Self-audit clean: no ProgramDescriptor/CBDescriptor/KernelDescriptor/TensorAccessorArgs/ta:: artifacts; reader(3 CTA+3 RTA)/writer(4 CTA+7 RTA)/compute(3 CTA) names match uwu kernel tokens. **Now python-routable** → added `test_quasar_untilize_multi_core_block` (wide + tall-wide shapes; tall shapes hit the cliff sub-regions). 4 of 8 untilize factories done.

---
## ✅ Pre-existing bug — FIXED by Pass 4 (kept for history) — `UntilizeMultiCoreBlockProgramFactory`
Surfaced when a wide test shape routed `untilize` to the block factory: brisc JIT-compile of
`untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` fails with
`'args'/'tensor'/'dfb' has not been declared` / `get_arg` → `experimental::get_arg`.

Root cause: that writer is **shared by two factories** —
1. uwu's `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp` (Metal 2.0), and
2. untilize's `untilize_multi_core_block_program_factory.cpp` (**still legacy `create_descriptor`**).

Prior uwu work m2-ified the shared kernel **in place (no fork)** — it now uses `dfb::`/`tensor::`/`args::`
named tokens that only resolve when the launching factory supplies Metal 2.0 bindings (which generate the
header opening those namespaces). untilize's legacy block factory supplies none → bare names don't compile.
This is exactly the recipe's "shared top-level kernel entry point → must fork before porting" rule, violated
upstream. Latent until a shape selects the block factory; the resnet demo will hit it if block is chosen.

**Fix options (owner / a later pass):** (a) fork the writer — uwu keeps the m2 version, restore a legacy
`writer_unary_stick_layout_wh_multicore.cpp` for untilize's block factory; or (b) port untilize's block factory
to Metal 2.0 (preferred — also closes one of the 7 remaining factories). Out of scope for this pass (touches
neither factory I ported). Tracked so it isn't mistaken for a parallelize_column regression.
