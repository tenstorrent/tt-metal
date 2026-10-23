# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-23
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - One `ttnn.generic_op` dispatch per call.
  - Regime R1 (`row_local`): tile-rows split across up to the full 120-core grid.
  - Regime R2 (`w_split_root_combine`): W split across row-major W-groups, combined by a unicast gather to the
    group root, a root `reduce<AccumulateViaAdd, ReduceWithinTile::Skip>` with a fused mean / +ε / rsqrt, and a
    `ttnn.Mcast` multicast back to the group.
  - Native TILE and ROW_MAJOR layouts (in-kernel tilize / untilize).
  - Non-aligned W via a partial reduce scaler on the last tile, plus zero-fill of the RM tail.
  - Single-sourced blocking plan (`_select_blocking` → `BlockingPlan`).
- **SUPPORTED at Phase 0**:
  - dtype = [float32, bfloat16]; fp32_dest_acc_en = [True]; layout = [TILE, ROW_MAJOR]
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned]; rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]; gamma_dtype = [float32, bfloat16, "none"]; gamma_layout = [TILE, ROW_MAJOR, "none"]
  - memory_layout = [INTERLEAVED]
  - EXCLUSIONS = [{float32, fp32_dest_acc_en=False}] (mandated)
- **Accuracy achieved** (`test_rms_norm_precision_baseline.py`, 4 shapes × 2 dtypes × 2 layouts, HiFi4, fp32 DEST):
  - bf16: PCC = 0.9999985, max_abs_err ≤ 0.042, rel_rms_err ≤ 0.0020, within 1 bf16 ULP (p50 = 0).
  - fp32: PCC = 0.9999997, max_abs_err ≤ 0.031, rel_rms_err ≤ 0.0019, ≈ 1.5e4 fp32 ULP p50, −0.13 % median bias.
    This is TF32-class: the FPU operand registers truncate fp32 operands. It is not a scale bug (confirmed by CPU
    emulation).
- **Golden suite at Phase 0**: 737 `supported_pass` / 0 `supported_fail` / 0 `xpass_drift` / 0 `xfail_wrong_mode`;
  6180 `xfail_expected`, 33900 `invalid_skipped` (per `verifier_report.json`).
- **Issues encountered** (verifier fixes):
  - `INPUT_TAGGERS` registered `tag_gamma_dtype` / `tag_gamma_layout`, so the golden cartesian never iterated
    gamma formats. All 1680 cartesian gamma cells were skipped as INVALID. Fixed by registering only the
    shape-derived taggers; `supported_pass` went from 177 to 737, and every newly exercised gamma cell passes.
  - The L1 solve (`_footprint`) and the allocation (`_build_cbs`) restated every CB size, so the x-depth knob was
    read from two different sources. Fixed with one `_cb_specs()` definition used by both.
  - `epsilon` was not validated (design contract: finite, ≥ 0). It now raises `ValueError`.
- **Tests added**: `test_rms_norm.py` (acceptance, planner), `test_rms_norm_debug.py` (implementer),
  `test_rms_norm_precision_baseline.py` (verifier).

## Refinement 1 — Numerical configurability: bfloat8_b (input and gamma) + fp32_dest_acc_en=False
- Date: 2026-10-23
- What was done:
  - **Op file:** `SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]` += `bfloat8_b`;
    `SUPPORTED["fp32_dest_acc_en"]` += `False`.
  - **EXCLUSIONS:** keeps `{float32, False}`. Adds `{bfloat8_b, w_non_aligned}` and `{bfloat8_b, h_non_aligned}`,
    mirroring the author-scoped INVALID entries in `feature_spec.py`, so a caller that bypasses the harness's INVALID
    skip (for example `test_translated.py`) gets `ExcludedCell`.
  - **Intermediate CB formats follow DEST width.** The `_FMT_F32` role became `_FMT_INTERM`, and `F32_TILE_BYTES`
    became `Geometry.interm_tile_bytes`, both inside `_cb_specs`, so the L1 solve and the allocation move together.
    Pages are `Float32` (4096 B) at `fp32_dest_acc_en=True` and `Float16_b` (2048 B) at `False`. This covers
    `cb_sumsq`, `cb_rowsum_partial`, `cb_rowsum_gather`, `cb_inv_rms_root`, `cb_inv_rms` and `cb_x_gamma`. The
    caller's `compute_kernel_config` is still passed through unchanged.
  - **bf8b TILE gamma:** the writer reads the whole 1088 B page per tile, via the new compile-time arg
    `gamma_is_block_float`. The bf16 / fp32 TILE gamma keeps its two face-row reads. A bf8b page's shared exponents
    precede the mantissas, so the face-row byte offsets do not apply. The Row broadcast still consumes row 0 only.
  - **Block-float element size:** `element_size()` is undefined for block-float dtypes, so `_elem_bytes()` returns
    0 bytes for them. That value only feeds the ROW_MAJOR stick paths and the face-row gamma reads, and a bf8b tensor
    (TILE only) never takes either.
  - **Compute kernel and reader unchanged.** The helpers carry the data-format reconfig, so the pass condition held.
  - **Reused:** R1/R2 selection, chunk-batched reads/writes, grid fill, every helper call, `_cb_specs` as the single
    CB source. **Added:** one format role, one Geometry field pair, one writer compile-time branch, two EXCLUSIONS.
  - **Selector shift:** 16-bit intermediates halve `cb_x_gamma`, so `_select_blocking` picks coarser chunks at
    `fp32_dest_acc_en=False`.
    - `(1,1,8192,4096)` bf16 goes from 2 chunks per row (`k=2, c=64`) to 1 (`k=1, c=128`).
    - The regime-pinned acceptance tests run at the default fp32 DEST, so they still pin their regime.
    - At 16-bit DEST, chunked-row R1 is now pinned by `(1,1,4096,8192)`: bf16 `k=5`, bf8b `k=2`.
    - The R2 pins (`32×16384`, `32×16300`, `64×12288`, `32×7168`) select the same `gw` at both DEST widths.
- Accuracy achieved (HiFi4 unless noted; reference computed in fp32 from the device-visible quantized input):
  - **bf16 + `fp32_dest_acc_en=False`** (shapes `[1×1×32×64, 1×1×128×512, 1×1×2048×1000, 1×1×32×16384]`, TILE + RM):
    PCC ≥ 0.999991, rel_rms ≤ 0.0054, ULP p99 = 2, max ≤ 5. Ratio median 0.9988 – 1.0038; at fp32 DEST it is
    0.9990 – 0.9993.
  - **bf8b** (aligned shapes, TILE): PCC ≥ 0.99995 at fp32 DEST and ≥ 0.99994 at 16-bit DEST; rel_rms ≤ 0.0098 and
    ≤ 0.0129. Ratio median 1.0023 – 1.0028 (fp32 DEST) and 1.0045 – 1.0083 (16-bit DEST).
  - **16-bit-DEST bias:** the extra ratio offset is per-row 16-bit rounding of the x² accumulation, S/W, +ε and
    rsqrt. A CPU emulation brackets it: round-to-nearest at every stage gives 1.0000 – 1.0005, truncation at every
    stage gives 1.0008 – 1.0196.
    - The fp32-DEST rows share every code path except the CB formats, and they keep the 5e-3 scale-bug guard.
    - The baseline's guard is therefore keyed on the precision corner: 1e-2 for bf16 and 1.5e-2 for bf8b at
      16-bit DEST.
  - **`_perf_case` corner** (bf16, HiFi2, `fp32_dest_acc_en=False`, bf16 TILE gamma): PCC ≥ 0.999985 on
    `32×1024`, `32×7168`, `640×6144` and `8192×1024`, against the 0.9995 gate.
- Golden test progress:
  - **Targeted slice** (every INTERLEAVED target cell of 4 cartesian shapes — `1x1x32x8192`, `4x8x47x256`,
    `2x4x128x512`, `2x1x100x47` — plus all 104 interleaved loose cells that are target cells, including the 12
    `_perf_case` profiles): 192/192 runnable cells pass; 584 skipped, all INVALID.
  - **`test_translated.py`:** 52 passed; the 54 xfails are all sharded placements (Refinement 2).
  - The full suite is the harness's run of record.
- Issues encountered:
  - `Tensor.element_size()` raises for bfloat8_b ("datum for bfp2, bfp4, bfp8 is invalid"); fixed with
    `_elem_bytes()`.
  - The baseline's ratio-median guard (5e-3) tripped on the bf8b rows at 16-bit DEST (1.0083 and 1.0070). This is
    the 16-bit-DEST precision class, not a scale bug (see above), so the guard was keyed on the precision corner.
- Tests added:
  - `test_rms_norm_numeric_formats.py` (91 cases):
    - bf16 at 16-bit DEST across R1 and R2, both layouts, and none / TILE / RM gamma;
    - bf8b input at both DEST widths with none / bf8b TILE / bf16 RM / fp32 TILE gamma;
    - bf8b TILE gamma under bf16 input, including non-aligned shapes;
    - the `_perf_case` precision corner;
    - the chunked-row R1 pin at 16-bit DEST;
    - the ExcludedCell checks.
  - `test_rms_norm_precision_baseline.py` gains `test_rms_norm_precision_baseline_numeric_formats` (bf16 at 16-bit
    DEST, and bf8b at both DEST widths). The existing rows are unchanged.

## Refinement 2 — Sharded placements: HEIGHT / WIDTH / BLOCK, native zero-copy
- Date: 2026-10-23
- What was done:
  - **Op file.** `SUPPORTED["memory_layout"]` gains `HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED`.
    `_validate_placement()` enforces the placement contract:
    - a sharded input requires an L1-sharded output with the same shard spec;
    - `memory_config=None` now inherits a sharded input's spec, so it is written in place (interleaved inputs still
      default to DRAM);
    - interleaved ↔ sharded mismatches raise `UnsupportedAxisValue`.
    `EXCLUSIONS` mirrors the author-scoped INVALID `{ROW_MAJOR, *_SHARDED, gamma TILE}`: a sharded RM slice is
    element-granular, and the TILE-gamma load is tile-granular.
  - **Mapping onto the built regimes.** `_shard_placement()` resolves the shard spec into a `ShardPlacement`:
    shard (sr, sc) maps to core `cores[sr·cols + sc]`, with the orientation order checked against the Mcast order.
    - HEIGHT → R1 (`gw = 1`).
    - WIDTH / BLOCK → R2. The W-groups are the shard-grid rows of cores, and the Mcast uses the shard orientation's
      core order. A WIDTH group may span several grid rows (for example 32 cores on an 8×4 grid).
    - The interleaved and sharded per-core arguments are now built from one `CoreWork` record each
      (`_interleaved_work` / `_sharded_work`).
  - **Zero-copy.**
    - TILE: `cb_x_tiles` and `cb_out_tiles` are backed on the input and output shards
      (`ttnn.cb_descriptor_from_sharded_tensor`, page overridden to the CB unit). The reader publishes the shard with
      one reserve/push (no NoC), and the writer only retires chunks.
    - RM shards that are whole tile-rows of whole tiles: `cb_x_rm_sticks` (tilize source) and `cb_out_rm_sticks`
      (untilize destination) are backed on the shards.
    - Any other RM shard: the stick stride is not a tilize stride and the shard is not a whole tile-row tall, so
      sticks are restrided by local L1 NoC copies. No `TensorAccessor` and no DRAM are involved.
    - Kernel CT mode: 0 = interleaved, 1 = RM restride, 2 = zero-copy.
  - **Blocking.** `_fit_block()` is the chunk / lookahead / block-rows fit now shared by `_select_blocking`
    (interleaved) and `_select_sharded_blocking`. The shard is the block: the selector picks the coarsest block that
    fits, down from the whole shard.
    - Zero-copy needs `k·c == n`, and a short last block runs at its actual row count (CT `ragged_last_block`).
      Without that, a prime 23-tile-row shard collapsed to `b = 1`; BLOCK 7168×2048 went from 74.7 µs to 51.6 µs.
    - Interleaved keeps nominal blocks and is byte-identical in behavior.
  - **Streamed gamma** (`plan.gamma_streamed`, sharded R1 only, used only when no resident fit exists):
    `cb_gamma_tiles` is `2·c`, and the writer loads gamma per (block, chunk) two ahead of the output. This is needed
    for HEIGHT shards with W = 11008, whose zero-copy shards leave 52 KB of CB L1.
  - **RM gamma** is read from the DRAM-alignment unit below the slice start (`cb_gamma_rm_stick` + 1 alignment
    unit). Face rows that are not 16 B aligned (an fp32 RM shard with a bf16 gamma) are copied by the RISC, at
    2 × 16 elements per tile.
  - **`_footprint`** skips shard-backed CBs: the shard bytes are already excluded by `_l1_budget_bytes`, so they are
    not counted twice. The CB-region addresses of `cb_rowsum_gather` / `cb_inv_rms` are unchanged (M1).
  - **Reused:** R1 / R2 kernels and every compute helper call, the combine (gather + semaphore + `ttnn.Mcast`),
    `_cb_specs` as the single CB source, and the RM tilize / untilize staging.
    **Added:** the shard placement, CB backing in `_cb_specs`, the x / out mode, short last blocks, streamed gamma,
    and element-granular RM slices (`slice_valid_elems`, `x_col0_bytes` / `out_col0_bytes`, `gamma_col0_elem`).
- Accuracy achieved: all at bf16 / HiFi2 / `fp32_dest_acc_en=False` / bf16 gamma.
  - The 9 sharded `_perf_case` geometries: rel_rms ≤ 0.0070, max_abs ≤ 0.197; the soft PCC 0.9995 gate passes on
    every profile.
  - HEIGHT 256×512 TILE, WIDTH 416×1184 RM and BLOCK 333×1000 RM: rel_rms ≤ 0.0071.
  - `test_rms_norm_sharded.py` gates every case at the dtype PCC (0.995 bf16, 0.999 fp32, 0.99 bf8b) and
    rel_rms < 0.05.
- Device-kernel ns (baseline for Refinement 3, BH p150b, against the reference at 1350 MHz):

  | Placement | Profile | ns | Reference ns |
  |---|---|---|---|
  | WIDTH | 32×1024 | 4051 | 4110 |
  | WIDTH | 32×2304 | 4617 | 4617 |
  | WIDTH | 32×5120 | 5433 | 5267 |
  | WIDTH | 32×7168 | 5918 | 5481 |
  | BLOCK | 640×6144 | 13826 | 19039 |
  | BLOCK | 640×7168 | 17114 | 21669 |
  | BLOCK | 7168×1024 | 43424 | 18871 |
  | BLOCK | 7168×2048 | 51570 | 25947 |
  | BLOCK | 8192×1024 | 60432 | 25640 |

  The tall BLOCK profiles are compute-bound per tile (≈ 470 ns per tile with one 23-row block per core, including
  the root's `gw`-way AccumulateViaAdd). Perf lamps L4 / L5 / L8 in Refinement 3 target that.
- Golden test progress (targeted slices; the harness's full run is the run of record):
  - sharded loose cells (resilience, pad-poison, perf, `_SHARDED`) plus the translated sharded tests: 337 passed,
    0 failed, 3 skipped (input-shard OOM in the harness), 2 xfailed (bf8b `w_non_aligned` EXCLUSION);
  - cartesian sharded cells for 1x1x32x8192, 4x8x47x256, 2x1x100x47, 128x8192 and the first ~527 cells in id order:
    911 passed, 0 failed, the rest INVALID / expected xfail;
  - interleaved loose cells plus the non-sharded translated tests: 159 passed, 0 failed.
- Issues encountered:
  1. The refactored chunk-candidate loop never terminated for `n = 1` (`ceil(1/k) = 1` never drops below the
     floor). It hung RM WIDTH shards with 8-element slices, which surfaced as a pytest timeout and also stalled the
     precompile collect pass. Fixed by bounding `k ≤ n`.
  2. A HEIGHT shard with W = 11008 could not keep gamma resident; streamed gamma fixes it.
  3. A prime shard height collapsed the block to 1; short last blocks fix it.
- Tests added: `test_rms_norm_sharded.py` (114 cases). It covers:
  - TILE auto geometry over 8 adversarial shapes × 3 placements;
  - poisoned padding;
  - fp32 / bf8b / no-gamma formats;
  - the 9 pinned perf geometries;
  - RM auto restride (6 shapes × 3 placements × gamma / no gamma);
  - fp32 RM with bf16 / fp32 gamma (misaligned gamma face rows);
  - RM zero-copy pinned geometries, with an assertion that the zero-copy path is the one selected;
  - streamed gamma (W = 11008 / 8192 HEIGHT);
  - the placement-mismatch, RM-sharded-TILE-gamma and default-`memory_config` contract cases.
