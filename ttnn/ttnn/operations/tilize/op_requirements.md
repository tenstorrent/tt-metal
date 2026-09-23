# Operation Requirements: tilize

## Definition
- **Formula**: `tilize(x)[i, j] == x[i, j]` for every element of the data region. The output is `Layout::TILE` (32×32 or `tile=` geometry, faces TL/TR/BL/BR). With a pad argument, every pad position holds exactly `pad_value`, and the logical shape never grows. Values are bit-identical unless `dtype=` casts, which must be value-preserving.
- **PyTorch Reference**:
  ```python
  import torch.nn.functional as F

  def tilize_reference(x, *, pad_value=None, padded_shape=None, tile_h=32):
      """Logical view: identity. Padded view: F.pad to the padded shape with pad_value (rank >= 2)."""
      if pad_value is None and padded_shape is None:
          return x.clone()
      if padded_shape is None:  # auto: tile-round the last two dims
          padded_shape = [*x.shape[:-2], -(-x.shape[-2] // tile_h) * tile_h, -(-x.shape[-1] // 32) * 32]
      pads = []
      for d, p in reversed(list(zip(x.shape, padded_shape))):
          pads += [0, p - d]
      return F.pad(x, pads, value=0 if pad_value is None else pad_value)
  ```
- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:
  ```python
  tilize(
      input_tensor: ttnn.Tensor,                      # ROW_MAJOR (or TILE, to re-tile)
      memory_config: ttnn.MemoryConfig | None = None, # output placement (default: input's)
      *,
      dtype: ttnn.DataType | None = None,             # output dtype (default: input's; fp8_e4m3 -> float32)
      low_l1: bool = False,                           # per-core L1 independent of tensor dims
      output_padded_shape: list[int] | ttnn.Shape | None = None,
      pad_value: float | int | None = None,
      tile: ttnn.Tile | None = None,                  # output tile geometry (default 32x32)
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Perf-focus contract** (`feature_spec.LOOSE_CASES[0]`, `# attention: PERF FOCUS`): [1,1,16384,64], bf16 → bf16, ROW_MAJOR in, `TensorMemoryLayout::INTERLEAVED` DRAM in and out, rank 4, `tile_grid = tall_narrow`, no pad, 32×32 tiles, `low_l1=False`, `fp32_dest_acc_en=False`, fast tilize. `measured_ns_wormhole_b0` = 25998, `measured_ns_blackhole` = 11852. **Every axis value and knob of this contract is already in Phase 0 SUPPORTED**, so the first two generality refinements are ordered by difficulty alone. Every perf refinement below targets this shape and this exact config.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]; **output_dtype**: [bfloat16]
- **SUPPORTED layout**: ROW_MAJOR input (`in_tile_height = "none"`) → TILE output, `tile_height = 32`
- **SUPPORTED placement**: `shard_api = "none"`, `out_scheme = "interleaved"`, `buffer = "dram_to_dram"`, `orientation = "none"`
- **SUPPORTED shape-derived axes**: `rank = 4`; `alignment = tile_aligned`; `pad_mode = pad_value = "none"`; `tile_grid ∈ {single_tile, small, tall_narrow}`; `low_l1 = False`
- **Cores**: multi-core row split. `split_work_to_cores(runtime grid, R, row_wise=True)` reaches `min(R, N)` Tensix cores (64/64 on the perf-focus shape).
- **Compute config**: `fp32_dest_acc_en=False`, `dst_full_sync_en=False` (half-sync), fast tilize, one helper call per kernel
- **Block knobs**: `block_width` = `balanced_width(C, CB_BUDGET_BYTES[low_l1])`; `rows_per_quantum` from `QUANTUM_MIN_TILES` (verifier); `DEPTH_IN = DEPTH_OUT = 2`; `READ_AHEAD` = 1 and the split reader, parked as live knobs
- **Golden baseline**: 30 supported_pass / 1746 xfail_expected / 2310 invalid_skipped / 0 loud (per verifier CLI). `test_regression.py` has 10 tracked failures (fp32 / uint16 / int32), closed by Refinement 7.

### [x] Refinement 1 — Sharded and L1 placement (legacy 2-D, ND, crossovers, DRAM-sharded) + rank widening

**Goal**: add to SUPPORTED:
- `shard_api`: `legacy_2d`, `nd`
- `out_scheme`: `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `nd`
- `orientation`: `ROW_MAJOR`, `COL_MAJOR`
- `buffer`: `dram_to_l1`, `l1_to_l1`, `l1_to_dram`
- `rank`: 2, 3, 5, 6

Build the two deferred placement regimes of op_design.md → Regimes, with the selection function exactly as pinned there:
- **`sharded_resident`**: an L1-sharded side with `resident_ok`. Core assignment = that side's shard grid (the output grid wins when both are sharded). One block = the whole resident shard: `block_height = shard_h / tile_h`, `block_width = shard_w / 32`. The valid extents of partial final shards are RT args; CB quanta stay nominal.
  - A resident **input** shard backs `cb_input_sticks` via `ttnn.cb_descriptor_from_sharded_tensor`, and the reader only publishes pages.
  - A resident **output** shard backs `cb_output_tiles`: compute packs straight into it and the writer issues no NoC writes.
  - With the same spec on both sides the op moves zero bytes.
- **`sharded_accessor`**: every other sharded case (interleaved ↔ sharded crossovers, cross-spec height-in / width-out, DRAM-sharded output, ND specs without a tile-aligned 2-D equivalent). The existing reader/writer address remote pages through `TensorAccessor` over the sharded side's core assignment.

The L1-interleaved `buffer` values and ranks 2/3/5/6 need no kernel change per the design (`TensorAccessor` addresses interleaved L1 and DRAM identically; leading dims fold into R). Land them here with a golden run as evidence.

**Verifier notes**:
- **Ordering.** Hardest refinement first (difficulty tier 1: block-sharded, COL_MAJOR, cross-spec and ragged shard grids). No other refinement blocks it. The padding, tile-geometry, 2-D-split and numeric refinements then extend these regimes rather than the other way round.
- **Blocking-model class.** Every shard scheme cuts only *independent* axes (HEIGHT cuts `tile_row`, WIDTH cuts `tile_col`, BLOCK cuts both), so there is no cross-core combine and no semaphores. It stands alone because of placement breadth, not topology.
- **Native path is the contract.** A core's *own* local shard must be consumed through the CB placement, **never re-read through a `TensorAccessor` as if it were interleaved**. A golden pass by accessor on a resident shard does not count as implementing the value; the verifier checks the dataflow, not the test colour.
- **Resident input layout matches the CB.** A ROW_MAJOR shard row has stride `shard_w * elem_bytes` = `block_width * tile_col_bytes`, which is exactly the tilize input CB layout, so the resident input needs no copy.
- **Shard-block expression.** Do not re-chunk the per-core loop below the shard block without an L1 reason. `QUANTUM_MIN_TILES` / `rows_per_quantum` apply to the *streamed* side only.
- **Runtime tagging is ready.** `validate()` already tags legacy vs ND from `created_with_nd_shard_spec` (Phase 0 review fix; `test_tilize_registry.py`). The harness's `axes.py:_spec_of` still mis-tags legacy inputs as `nd` (reported, not an op bug), so expect `merge_axes` mismatch warnings until the harness is fixed.
- **Program cache.** Shard addresses ride on the CB descriptors or RT args, never on CT args (`PROGRAM_CACHE_CASES`: `height_sharded_same_spec`, `interleaved_to_height_sharded`).
- **Cells that stay xfail.** Cells that also need `pad_mode` (`padded_to_height_sharded`) or `tile_grid = short_wide` (`short_wide_width_sharded`) remain xfail until Refinements 4 and 5. XPASS-strict will promote them there.
- **Reference timings.** The sharded LOOSE_CASES (not flagged) give device-ns references to report against: HEIGHT-sharded in → DRAM 16852 ns, DRAM → HEIGHT out 12142 ns, same spec 1891 ns, BLOCK COL_MAJOR 1832 ns (WH).

**Done when**: every golden cell whose only unsupported axes are the ones above passes, with zero loud categories. The `sharded_legacy_2d` / `sharded_nd` groups and the translated sharded / L1 / rank tests pass. The sharded LOOSE_CASES report device-ns and core counts, and the same-spec case shows no NoC traffic on the resident sides.
**Outcome**: landed. Every named axis value is in SUPPORTED. `test_golden.py` has 32 pass / 0 fail / 0 XPASS; 24 of those are new, including every `sharded_legacy_2d` / `sharded_nd` / buffer / rank cell at bf16 → bf16. `short_wide` / `square_large` are admitted only where an L1 shard fixes the core assignment; EXCLUSIONS refuse them on every row-split path until Refinement 5, and that is what lets `short_wide_width_sharded` and the four sharded LOOSE_CASES run now. Translated sharded / ND / L1 / rank tests: 191 pass, 1 fail. The failure, `test_tilize_program_cache_addr_change[sharded_width_l1]`, passes alone; in module order an earlier COL_MAJOR 1×4 WIDTH case already built the byte-identical program, so its `entries == 1` sees a correct cache hit. Sharded LOOSE_CASES, WH device-kernel ns: HEIGHT in → DRAM 16502 on 64 Tensix cores (ref 16852); DRAM → HEIGHT out 12269 on 64 (ref 12142); same spec 1914 on 64 (ref 1891); BLOCK COL_MAJOR 1971 on 16 (ref 1832). The same-spec case moves no NoC bytes: NCRISC 252 ns (page publish only), BRISC waits on compute. What is left: the same-spec / BLOCK cases are compute + launch bound (~1.7 µs for 16 tiles per core); DRAM → HEIGHT out is read-bound (NCRISC ~10.7 µs of 12.3).

### [x] Refinement 2 — Tile geometry: tiny output tiles + retile of a TILE input

**Goal**:
- add `tile_height` 16, 8, 4, 2, 1 to SUPPORTED. Both CBs carry `TileDescriptor(tile_h, 32)`. The output is allocated through a `TensorSpec` carrying `tile` (not `allocate_tensor_on_device`, which defaults to 32×32). The helper leaves the fast path and uses standard `tilize_block`. The reader already groups `tile_h` sticks per tile-row.
- add `in_tile_height` 32, 16, 8, 4, 2, 1: a `Layout::TILE` input re-tiled in the same dispatch, via the `retile_l1_facewalk` regime:
  - the reader reads whole input tiles (one page-sized NoC read each) into the reader-private `cb_retile_staging` (ledger → deferred rows);
  - it face-walks them into the stick layout of `cb_input_sticks` on the same RISC-V;
  - `block_height` along R is rounded to `row_align = max(1, in_tile_h / tile_h)`, so an input tile-row never straddles two cores;
  - compute and writer are unchanged.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Ordering.** Difficulty tier 3 (layouts), second after sharding.
- **Bundling.** Retile is a new *reader* block operation (a scheme-change of `load_block`, with no cross-core topology). It is bundled with tiny tiles because 4 of the 6 retile cases produce tiny tiles. Land tiny tiles first inside this refinement.
- **Rejected designs.** `retile_compute_untilize` and `retile_dram_facerow_reads` are rejected in op_design.md; do not build them.
- **Prompt MUSTs.** Both tile axes MUST be built and verified on this box (Wormhole), and the retile MUST NOT materialize a ROW_MAJOR tensor.
- **Tiny-tile L1.** `in_tile_bytes = tile_h · 32 · elem_bytes` shrinks with `tile_h`, so `block_width` and `rows_per_quantum` re-derive automatically. Check that the fast-tilize cap and `QUANTUM_MIN_TILES` still produce sensible quanta at `tile_h` = 1: a "tile" is then one stick.
- **Deferred to Refinement 7.** `bfloat8_b` / `bfloat4_b` × tiny tiles. If the packer cannot do it, it goes to EXCLUSIONS then, not INVALID.
- **INVALID follow-up.** `fp8_e4m3 × in_tile_height ≠ none` is structurally impossible (fp8 is ROW_MAJOR-only). It is recommended for INVALID (verification_report.md); it is not refinement work.

**Done when**: the `tile_geometry_tiny` and `tile_geometry_retile` cells pass bit-exact at bf16, including `retile_1_to_32` (the `in_tile_height: 1 × tile_height: 32` required cross). `PROGRAM_CACHE_CASES` `tiny_tile_16` and `retile_32_to_16` pass. Zero loud categories.
**Outcome**: landed. Every named axis value is in SUPPORTED: `tile_height` 32/16/8/4/2/1 on every placement, and `in_tile_height` none/32/16/8/4/2/1.
- **Golden.** All 3 `tile_geometry_tiny` and all 6 `tile_geometry_retile` cells pass bit-exact at bf16, including `retile_1_to_32`, as do `PROGRAM_CACHE_CASES` `tiny_tile_16` / `retile_32_to_16`. Slice: 14 passed, 0 failed, 0 XPASS.
- **Tiny tiles.** Host only: a `TensorSpec`-carried output tile, and a quantum floor counted in full-tile equivalents.
- **Retile.** `read_retile` (`retile_l1_facewalk`) reads whole tiles into a prefetching staging ring, or reads a resident TILE shard in place. It face-walks them into `cb_input_sticks` with NoC loopback reads (`RETILE_FACEWALK_NOC`), which measured 2.4× faster than the RISC-V copy the design named. Retile 32→16 on [1,1,16384,64] (64 Tensix cores, WH) went from 93.5 to 38.3 µs; tiny 16 / 8 take 24.0 / 24.9 µs; the perf-focus path is unchanged at 25.5 µs.
- **What is left.** Retile is face-walk-issue bound (NCRISC 31 µs vs 17 µs for the reads alone). 1→32 is bound by its 64-byte input page reads; tile_h = 1 is bound by its 64-byte output page writes.

### [x] Refinement 3 — Speed up the perf-flagged profile (block-quantum / depth / NoC co-tune)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags [1,1,16384,64] (bf16 → bf16, DRAM interleaved, 64 Tensix cores on WH) as the mandatory perf target. `measured_ns_wormhole_b0` = 25998, `measured_ns_blackhole` = 11852; the current op is at ~25.4–26.1 µs on WH.
- **Measured signature.** About 165 GB/s combined DRAM traffic, versus about 198 GB/s for the same kernel with 1 KiB stick segments ([1,1,16384,512]). Reads-only ≈ writes-only ≈ 15.6 µs, and they add up.
- **Diagnosis.** The shape is bound by small (128-byte, one-page) stick reads and by read/write serialization, not by compute.
- **Levers (T1/T2 knob co-tunes on the surface the planner and verifier exposed).** Pick from `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `noc_placement`, `split_reader`):
  - co-tune `QUANTUM_MIN_TILES` × `DEPTH_IN` / `DEPTH_OUT` × `READ_AHEAD` (the quantum was a single tile-row until the Phase 0 review; 8 tiles measured best so far);
  - NoC / stream placement of the read and write streams, including splitting one stream across both NoCs;
  - write-side batching.
- **Headroom first.** Establish the roofline with `/perf-ceiling-dm` before tuning.
- No SUPPORTED change.

**Done when**: measured device-ns improves on [1,1,16384,64] at this exact config, with the core count reported (64 on WH). The golden suite stays green, and there is no regression across the config-spanning guard set: one representative per distinct kernel path × layout × placement that exists at that point, i.e. narrow- and wide-row interleaved DRAM, L1 interleaved, sharded resident, sharded accessor, tiny tile and retile.
**Outcome**: measured null, levers parked. [1,1,16384,64] runs at 25.2 → 25.0 µs on 64 Tensix cores (WH), within noise; the 7-path guard set is unchanged within noise.
- **Bottleneck.** Aggregate DRAM throughput for this transaction mix (128-byte stick reads plus 2 KiB tile writes, about 165 GB/s including fixed costs). Evidence: 32 Tensix cores take 30.0 µs, only 17 % slower. Reads only (16.1 µs) plus writes only (19.2 µs) minus the no-transfer floor (8.1 µs) ≈ the full 25.4 µs.
- **Levers.** Each named lever was built, verified bit-exact, measured, and parked at a default that leaves the default path's kernel code and CB sizes unchanged:
  - quantum × depth × read-ahead windows, with write-ahead as the write-side batching twin;
  - eager publish;
  - splitting either stream across both NoCs;
  - bank-stride read addressing.
- **Results.** Every lever is flat or slower on the perf shape. The one apparent 2 % gain (1-row quanta, 4-deep windows) did not reproduce and regressed [1,1,16384,32] by 9 %. NoC1 reads cost +30–40 %. Cheaper read issue made reads-only slower, because it congests the DRAM banks.
- **Next.** Fewer, larger DRAM reads (Refinement 6's bank-coalesced reads: sticks `p`, `p + 12`, `p + 24` are contiguous in one bank). The parked bank-stride addressing already computes them. Not done here because it is Refinement 6's scope.

### [x] Refinement 4 — Padding: auto / explicit pad, all fill signs, non-aligned H / W, rank 0 / 1

**Goal**: add to SUPPORTED:
- `pad_mode`: `auto`, `explicit`
- `pad_value`: `zero`, `positive`, `negative`
- `alignment`: `w_non_aligned`, `h_non_aligned`, `hw_non_aligned`
- `rank`: 0, 1

**In-kernel.** `load_block` gains the per-image stick map `r → (image = r / h_tiles_img, h = (r % h_tiles_img) · tile_h + s)`. It reads only `h < H`, `image < num_images` and the valid W extent. It fills the W tail, the H tail, whole pad sticks and whole pad tiles / tile-rows / images (`output_padded_shape` beyond the tile-round) with `dataflow_kernel_lib::fill_l1_range<elem_bytes>` (`l1_helpers.hpp:89-90`). The fill value is packed per input dtype as an RT arg.

**Host.** `R`, `C` come from the padded shape. The output is allocated with the padded shape while the logical shape stays the input's. Ranks 0 and 1 synthesize the tile dims (`[] → [32, 32]`, `[W] → [32, round_up(W, 32)]`).

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Ordering.** Difficulty tier 3 (non-alignment fill in the reader).
- **Extends earlier regimes.** Padding must also land on the Refinement 1 sharded regimes (`padded_to_height_sharded`, the `pad_mode × legacy_2d` required cross) and on the Refinement 2 tile heights (alignment is measured against `tile_h`).
- **No wrapper ops.** Native kernel fill only. `ttnn.pad`, `ttnn.fill_implicit_tile_padding` and `ttnn.to_layout` wrappers are forbidden by the prompt (MUST NOT), so there is no partial-tick wrapper path here.
- **Fold across images.** The leading-dim fold breaks at image boundaries when `H % tile_h ≠ 0`; `square_large_from_leading_dims` [8,1,249,2048] reaches R = 64 only through the per-image `ceil`.
- **Cells that pass later.** `short_wide_single_stick`, `short_wide_w_tail` and `padded_low_l1` also need Refinement 5. `test_pad_value_extremes` (fp32) needs Refinement 7. XPASS-strict promotes them.

**Done when**: the `padding_auto`, `padding_explicit` and `padding_crossed` cells (other than those gated on Refinement 5), `rank1` and `rank0_scalar` pass. The pad region holds exactly the fill, and `to_torch(out)` still equals `x` at the logical shape. `PROGRAM_CACHE_CASES` `auto_hw_tails_negative_fill` passes. Zero loud categories.
**Outcome**: landed. Every named axis value is in SUPPORTED.
- **Golden.** Every `padding_auto` / `padding_explicit` cell, `padded_to_height_sharded`, `padded_l1_to_l1`, `rank1`, `rank0_scalar` and `test_program_cache_reuse[auto_hw_tails_negative_fill]` pass bit-exact at bf16: 15 of 19 cells, 0 failed, 0 XPASS. The other 4 (`padded_low_l1`, `short_wide_single_stick`, `short_wide_w_tail`, `square_large_from_leading_dims`) xfail on Refinement 5 axes. The translated `tilize_with_val_padding` family passes, including the sharded cases whose output `MemoryConfig` has no shard spec (now derived from the input's).
- **Mechanism.** The output is allocated at the padded shape and returned as a zero-copy logical view. The stick reader walks the padded grid through `PadMap` and fills with `PadFill`: CPU stores for short ranges, NoC loopback copies from a 1 KiB pre-filled source for long ones.
- **Refused.**
  - Retile × a pad that has something to fill is an EXCLUSION: the face walk has no fill; it would have to clamp staging reads and walk to the input's tile-rows / tile-columns, then fill after the walk lands.
  - Inner leading-dim growth raises `NotImplementedError`: TTNN's logical view cannot express it.
- **Perf.** No unpadded path regressed (guard set within noise). On the padded path the remaining cost is the per-stick W-tail band store on narrow tensors: [1,1,16370,50] is 36.6 µs vs 25.3 µs aligned, and 27.6 µs with the band stubbed. It is persisted across CB-ring passes, so tall tensors pay it once per CB row: [1,1,65520,50] is +9 % over aligned.

### [x] Refinement 5 — 2-D grid split (short_wide, square_large) + low_l1

**Goal**: add `tile_grid` `short_wide`, `square_large` and `low_l1` `True` to SUPPORTED.
- **2-D split.** Implement the `grid_2d_split` assignment rule pinned in op_design.md → Regimes: choose `(g_r, g_c)` minimizing the busiest core's tiles, with tie-breaks for wider stick segments and then fewer cores. `NUM_COL_GROUPS` is replaced by that rule, and every column start is a multiple of `col_align_tiles`. The kernels already take `(row_start, core_row_tiles, col_start, core_col_tiles)`.
- **low_l1.** `CB_BUDGET_BYTES[True]` = 65536 bytes is already a bounded, dimension-independent budget. Add the value and prove the A/B (`low_l1=True` vs `False`) is bit-identical.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- **Blocking-model class.** Knob-turn: both work axes are independent, so there is no cross-core combine.
- **Ordering.** Tier 4 (simple cross-core split), after padding, because `short_wide_single_stick` / `short_wide_w_tail` / `padded_low_l1` need the fill.
- **Must fill the grid.** Reported core counts on `short_wide_canonical` [1,1,32,2048] and `square_large` [1,1,2048,2048] must be ≈ the full runtime grid. Declaring `short_wide` while collapsing onto 1–2 cores is the SUPPORTED drift the golden suite cannot catch.
- **Also apply the rule on `tall_narrow` when `R < N`.** On Blackhole (110–130 cores), `tall_narrow_grid_scale` (R = 64) idles cores under the row split.
- **Transaction-size lamp.** Measure a `core_col_tiles` floor (e.g. 8 tiles = 512-byte segments) against maximum participation (op_design.md perf lamps).
- **Re-measure the quantum knob.** `QUANTUM_MIN_TILES` interacts here: narrow column groups make `block_width` small, which raises `rows_per_quantum`, capped by `max_positions // DEPTH_IN`. Re-measure on short_wide, where a core may own a single tile-row.
- **Reference timings.** The short_wide LOOSE_CASES are device-ns references: [1,1,32,8192] 7142 ns, [1,1,32,2048] 3486 ns (WH).

**Done when**: the `work_geometry` cells pass, including `short_wide_l1_forcing` and `low_l1_forcing_width` with no OOM at either setting. Every `low_l1` scenario passes the bit-identical A/B. `PROGRAM_CACHE_CASES` `short_wide_canonical`, `tall_narrow_grid_scale` and `square_large` pass. Core counts are reported. Zero loud categories.
**Outcome**: landed. `tile_grid` `short_wide` / `square_large` run on every placement (the row-split EXCLUSIONS are gone) and `low_l1` is `[False, True]`.
- **Golden.** `test_golden.py` has 74 pass / 0 fail / 0 XPASS: every `work_geometry` and `low_l1` cell, with no OOM at either setting on `short_wide_l1_forcing` / `low_l1_forcing_width`, the bit-identical `low_l1` A/B, and `PROGRAM_CACHE_CASES` `short_wide_canonical` / `tall_narrow_grid_scale` / `square_large`.
- **Rule.** `grid_2d_split` is host-only, with `ROW_COST_TILES = 1.5` added to the pinned tile-count cost: the pure count regressed [4,3,256,96] by 22 % and [1,1,2080,2048] by 2×.
- **Core counts (WH).** short_wide_canonical 64 of 64 Tensix cores (was 1): 14.3 → 3.7 µs (ref 3.49). [1,1,32,8192] 64 cores: 30.8 → 7.2 µs (ref 7.14). square_large 64 of 64 (row split = the 2-D optimum on 64 cores; 64 × 2 on a 130-core grid). tall_narrow_grid_scale 64 of 64.
- **Lamps.** The column-group floor lamp favors maximum participation, so it is parked at 1. Gated column pipelining (`PIPELINE_MIN_POSITIONS` 2, ≥ 2 KiB segments) takes square_large from 92.3 to 87.0 µs.
- **What is left.** short_wide is one tile-row per core: 32 reads of 64–256 bytes, then one tilize, then 1–4 writes, fully serialized. Launch and fill latency are most of its ~3.7 µs.

### [x] Refinement 6 — Speed up the perf-flagged profile (bank-coalesced stick reads)

**Type**: perf

**Goal**: the same flagged target, [1,1,16384,64] bf16 → bf16 DRAM interleaved, toward `measured_ns_wormhole_b0` = 25998 / `measured_ns_blackhole` = 11852 and beyond.
- **The one T3 lever for this phase: fewer, larger DRAM read transactions on narrow sticks.** At W = 64 bf16 every stick is its own 128-byte DRAM page, so the row split issues 16384 × 128-byte reads.
- **Why coalescing is possible.** On `TensorMemoryLayout::INTERLEAVED` DRAM, stick page `p` lives in bank `p mod num_banks` at offset `(p div num_banks) · aligned_page_size`. The sticks one core needs from one bank are therefore contiguous in that bank. One NoC read of `k · stick_page_bytes` fetches `k` sticks. They are then placed into the tilize stick layout, either by an L1-local scatter on the reader RISC-V or by choosing the per-core row assignment and landing layout so compute consumes them directly.
- Measured ~198 GB/s at 1 KiB segments vs ~165 GB/s here bounds the expected gain.
- Pattern references: `ttnn/ttnn/operations/examples/master.md` (`split_reader`: transaction size vs issue rate; `double_buffer`; `noc_placement`).
- Validate by building the variant and measuring device-ns, not by inferring from an ablation.
- No SUPPORTED change.

**Done when**: measured device-ns improves on [1,1,16384,64] at this exact config, with the core count reported and the output bit-exact. The golden suite stays green, with no regression across the config-spanning guard set (one representative per kernel path × layout × placement: narrow- and wide-row interleaved DRAM, L1 interleaved, sharded resident, sharded accessor, tiny tile, retile, 2-D split, `low_l1`).
**Outcome**: landed. [1,1,16384,64] takes 25.7–26.0 → 23.3–23.5 µs on 64 of 64 Tensix cores (WH B0, −9 %, medians of 3, bit-exact).
- **Mechanism.** The `bank_coalesced` load_block issues one NoC read per DRAM bank per run of tile-rows (~5 sticks each) instead of one 128-byte read per stick, then a NoC-loopback scatter into the tilize stick layout. The quantum on this path is 2 tile-rows.
- **Other shapes.** [1,1,16384,32] 18.2 → 13.3 µs (−27 %) and [1,1,32768,64] 51.5 → 46.3 µs (−10 %). The path is gated to sticks of at most 256 bytes, because 2 KiB sticks regressed +7.6 %. The rest of the guard set is within noise.
- **Bottleneck now.** Aggregate DRAM throughput for the read + write mix. The no-transfer floor is 2.3 µs (was 8.1). Reads only take 12.0 µs, flat in the per-bank read size. Writes only take 17.8 µs, and the same on 32 Tensix cores as on 64. The scatter costs ~22 cycles per stick but only ~1.3 µs of the wall.
- **Writer twin.** Measured and not built: one 4 KiB bank-contiguous write per tile-row made writes-only slower, 17.0 → 28.3 µs.
- **Next.** Try raising DRAM write efficiency, e.g. read/write phase grouping to cut DRAM bus turnarounds. I did not try it here because it is outside this refinement's read-coalescing scope.

### [ ] Refinement 7 — Numerical formats: fp32 / fp8 / integer inputs, block-float and integer outputs

**Goal**: add to SUPPORTED:
- `dtype`: `float32`, `fp8_e4m3`, `uint32`, `int32`, `uint16`, `uint8`
- `output_dtype`: `float32`, `bfloat8_b`, `bfloat4_b`, `uint32`, `int32`, `uint16`, `uint8`

Also:
- The value-preserving cast happens at pack: the `cb_output_tiles` format is the output dtype.
- fp32 → fp32 and 32-bit integers use `Fp32Mode::Lossless` + `fp32_dest_acc_en=True` + `UnpackToDestFp32` on `cb_input_sticks`. The fast path truncates fp32 → tf32 (`tilize_helpers.hpp:104`, static-asserted at `tilize_helpers.inl:107-115`).
- Expose `compute_kernel_config` if the skill's contract calls for it.
- Correct the intermediate-CB formats.
- Cells that fail out of the box go to EXCLUSIONS, not their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Ordering.** Cheapest tier, deliberately last. It extends every regime built by Refinements 1, 2, 4 and 5 (sharded, retile, padding, 2-D split), so it lands once those structures are fixed.
- **Keep the perf-focus path.** The perf-focus bf16 → bf16 path must stay on `fp32_dest_acc_en=False` + fast tilize. The flagged shape must not regress under the new config plumbing; re-measure it.
- **Byte-level fixes.** Integer widths change `col_align_tiles`: uint8 gives 32-byte tile-columns, so 2-tile alignment on Blackhole's 64-byte DRAM alignment. The pad fill needs a signed → unsigned bit_cast for negative integer fills (extends Refinement 4's fill).
- **fp8_e4m3.** Constructible only on Blackhole (`INVALID_FOR_ARCH` on WH). Declare it in SUPPORTED unconditionally, as feature_spec says the op does, and verify it on a Blackhole run.
- **Ledger audit 3.** A `Float32` page is only legal with `fp32_dest_acc_en=True`, and a 16-bit page under fp32 DEST is a correctness bug. Update `l1_ledger.md` page formats. At fp32 → fp32 `per_col_tile_bytes` doubles, so the `block_width` cap drops to 32 tiles.
- **Tracked failures this closes.** `test_regression.py::test_integer_passthrough` (uint16 / int32), `test_extreme_magnitudes` (fp32) and `test_pad_value_extremes` (fp32 + pad).
- **Reference timing.** The fp32 LOOSE_CASE [1,1,8192,32] (WH 15064 ns) is the device-ns reference for the fp32 datapath.

**Done when**: every golden cell whose only unsupported axes are `dtype` / `output_dtype` passes at its golden tolerance (`exact` for fp32 → fp32 and same-width integers, per `helpers._transition_tolerance` for block-float casts), or is listed in EXCLUSIONS with its failure category. The 10 `test_regression.py` failures pass. Zero loud categories.

### [ ] Refinement 8 — Speed up the perf-flagged profile (post-generality re-tune)

**Type**: perf

**Goal**: the same flagged target, [1,1,16384,64] bf16 → bf16 DRAM interleaved. After all generality refinements have reshaped the kernels (regime selection, the pad-aware reader, the numeric config plumbing), re-measure the flagged shape and the guard set. Recover any regression those refinements introduced on the flagged path, then take the next lever the roofline leaves (`/perf-ceiling-dm`) from `ttnn/ttnn/operations/examples/master.md`. Candidates: revisit the parked split reader once the write path is shorter; the tiny-work grid-participation lamp ([1,1,128,64] on 4 cores is at ~3.1 µs vs a 2.29 µs reference). If the roofline shows no headroom left, close the phase with that measurement. No SUPPORTED change.

**Done when**: measured device-ns on [1,1,16384,64] is at or below the best recorded after Refinement 6 and improves where headroom exists. The golden suite stays green, with no regression across the config-spanning guard set (one representative per kernel path × layout × placement × dtype family).
