# Verification Report: conv2d_3x3

Board: Blackhole p150 (11x10 compute grid). Regime under test: R0 interleaved-slab (DRAM interleaved,
ROW_MAJOR bf16 activation). Artifacts: `golden_results/verifier_report.json` (CLI output, compacted:
the raw report is 1.6 MB, over the repo's 500 KB file limit, so bulk categories are reduced to per-memory_layout
counts), `golden_results/golden_results.txt`.

## Code Review

### Fixed

| # | Finding | Fix |
|---|---------|-----|
| 1 | **SUPPORTED under-claimed** (drift the gate hid): the kernel already takes `stride` and `dilation` as CT args (gather row map `h_in = ho*s + kh*d - p`, band stride `min(d*W, band_len)`, per-tap reads at d>1), and every weight CB format comes from the weight tensor's dtype with the matmul's `INPUT_AND_OUTPUT` reconfig. Probe: the gate lifted on the 80 interleaved RM bf16 golden cells; **80/80 pass**, including both dispatches and the cache-hit check (stride 2, d = 2..18, bf8_b weights incl. the mixed bf16 x bf8_b cell, `dilated + strided`). | `weight_dtype += bfloat8_b`, `stride_mode += strided`, `dilation_mode += dilated`. Golden supported_pass 12 -> 64. |
| 2 | `cb_bias` pages are sized from the **weight** tile size (`_cb(CB_BIAS, bias.dtype, g.weight_tile_bytes, ...)`). With bf8_b weights now supported, a bf16 bias next to bf8_b weights would declare a bf16 CB with 1088 B pages. | `_check_contract` raises `ValueError` when `bias.dtype != weight_tensor.dtype` (prepare_conv2d_bias always emits `weight_dtype`, so this enforces the documented contract). New test `test_bias_dtype_contract_violation`. |
| 3 | `l1_ledger.md` assumed a 1300 KB budget; the device budget is 1367 KB. Its d18 worked-value row was still marked "(refinement)". | Ledger corrected; worked values verified against `select_blocking` on device (table below). |
| 4 | No precision baseline. | `tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_precision_baseline.py` (4 shapes x {bf16, bf8_b} weights). |

### Not promoted: `channel_alignment = c_in_unaligned`

The same probe passes all 16 `c_in_unaligned` interleaved cells (C_in in {8, 16, 48}, incl. stride 2). It
is **not** promoted. The kernel copies whole slab pages (`C_in_p * 2` B). For RM interleaved,
`C_in_p = roundup(C_in, dram_alignment / 2)`, so lanes `C_in..C_in_p` are DRAM page-alignment padding, which
is not data. The design's own risk table ("Garbage x zero weight = NaN") requires zero-filling those lanes,
and the reader does not. The golden suite cannot see the gap. The host write path stores full aligned pages
(`tt_metal/impl/buffers/dispatch.cpp:108-109`, `page_size_to_write = aligned_page_size`). A NaN-poison probe
(`probes/probe_003.py`, run with `c_in_unaligned` temporarily in SUPPORTED) tested it directly: a NaN tensor with the same 64 B page stride was freed, and the
C_in = 8 input then landed at the same address. The output still had 0 NaNs, because `from_torch`
rewrote the padding. An input produced by an upstream device op writes only `C_in * 2` B per page, so the lanes keep
stale bytes and `0 x Inf/NaN` poisons the output. This is filed in Refinement 1 with an explicit lane-zeroing
requirement. The K-pad zero-fill (each kh block padded to 32 columns) is already implemented. On BH no
golden INPUT exercises it. On WH, C_in = 8 does.

### Reviewed, no change needed

- **Helpers.** Compute uses `matmul_block` with tilize as the `PreKBlockFn` (`ShortAfterPreKBlock`, fp32
  packer-L1 accumulation, `TileRowMajor`) and `add_bias_bcast_rows<RowBroadcast>`. `pack_reconfig_l1_acc(0)`
  is issued before every later pack, as the lessons-file trap requires. Raw dataflow is used for the
  gather (no helper can express a tap map with zero-fill; `read_sticks_for_tilize` was rejected in the
  design) and for tile-block reads and writes (no helper exists).
- **CB sync.**
  - `cb_im2col_rm`: push `bk` per tile-row; the tilize waits `bk` per tile-row (`WaitBlock`).
  - `cb_im2col_tiles`: the tilize pushes `bm*bk` per K-block and the matmul pops `bm*bk` per K-block.
  - `cb_weights`: push and pop `bk*bn` per K-block.
  - `cb_output_tiles`: push and pop `sbh*bn` per row group.
  - `cb_bias`: push `bn` once, wait once, pop once at kernel end.
  - Ragged tail: nominal counts everywhere.
- **API.** `TensorAccessor` throughout, `void kernel_main()`, `api/dataflow/dataflow_api.h`, trid fencing
  via `local_copy_helpers_dataflow.hpp`.
- **Broadcast.** Bias is a row broadcast of tile row 0, the minimal form. No redundant fills.
- **Writer deadlock-freedom (> 1 M-block per core).** Weights for block j are reserved before any output
  of block j is drained. Prefetch of block j+1 happens only into slots that
  `cb_pages_reservable_at_back` reports free. This matches the design's stall-shadow argument.

### Prompt rules (`eval/prompts/conv2d_3x3.txt` `## Rules`)

| Rule (applies to current SUPPORTED) | Status |
|---|---|
| MUST NOT use ttnn.conv2d / host im2col + ttnn.matmul | ✓ own kernels |
| validate MUST raise ValueError for rank/shape, `H_out/W_out < 1`, `weight.shape[2] != K_p` | ✓ (`_check_contract`, tests `test_input_shape_contract_violation`, `test_weight_k_contract_violation`) |
| prepare_conv2d_weights row order (kh, kw, c), C_in padded first, each kh block padded | ✓ (`test_prepared_weight_layout`, golden `test_single_tap_shift` 9/9) |
| weight logical shape `[1, 1, K_p, C_out]` | ✓ |
| no host work in `conv2d_3x3` | ✓ (no torch / from_torch; blocking is `lru_cache`d; descriptor build is host metadata only) |
| every buffer address a runtime arg | ✓ (golden second dispatch adds 0 cache entries on all 64 cells) |
| one native dispatch per invocation | ✓ (one `ttnn.generic_op`) |
| Phase 0 weight reads: prefer each core reading its own kh blocks (soft) | followed; multicast is filed as Refinement 3 (perf) |
| relu / sharded / BLOCK rules | not yet applicable (values not in SUPPORTED); re-armed in Refinements 1, 2, 4 |

### Design conformance (op_design.md)

- **Algorithm and pipeline topology** match. The slab gather (reader, NCRISC) feeds tilize + K-blocked
  matmul + bias (compute), and the writer (BRISC) handles the weight stream and the output.
- **Work distribution.** M is split row-major over the live grid on every call
  (`num_m_groups = min(tensor_m_tiles, grid_cores)`), one M-block per core on every supported INPUT
  (`bm = core_m_tiles`). The few-core shapes (8x16 images: 4 cores) are limited by M itself. The design
  names this (lamps P3/P4, regime R6, traffic rank 4 for the N split), so it is not a contract violation.
- **Blocking-model fidelity.**
  - `select_blocking` is the single source of `bm`/`bk`/`bn`, the depths and the subblock shape.
    CB page counts, CT args and RT args all derive from `Conv3x3Blocking`. No collapsed knob.
  - No CB unconditionally scales with a whole-op dimension. The slab is the block footprint, and the
    weights CB is `bk x bn`. `bn = tensor_n_tiles` is the N knob at its designed value, bounded by
    TARGET `C_out <= 512` (16 tiles).
  - The depth constants (`IM2COL_RM_DEPTH_ROWS`, `WEIGHT_DEPTH`, `OUT_DEPTH_ROW_GROUPS`) are defined once
    and read by both the L1 solve and the blocking record.
  - The slab element width is restated in the kernel as `ELEM_BYTES = 2`. That is a format constant
    (the slab is bf16 by design), not a block knob.
- **Expression.**
  - Writer: one read barrier per weight K-block, one flush per output row group.
  - Compute: whole-block helpers.
  - Reader: one fence per band, then one barrier and one push per gathered tile-row. That per-tile-row
    handshake is the planned streaming window of `cb_im2col_rm` (M streams at depth 2, lamp P2).
    `bm = 1` on every supported INPUT, and P2 measured flat in the implementer's run (225 -> 218 us at
    best), so this is not an expression failure.
- **Axis accounting** in the ledger matches the code (spans vs streams per CB).

### Advisories (not refinements)

- The descriptor hard-codes `fp32_dest_acc_en=True` and ignores `dst_full_sync_en`. This is equivalent
  under the gate (TARGET is `[True]` only). `DEST_TILES_FP32_HALF_SYNC = 4` assumes half-sync, so honoring
  `dst_full_sync_en` would need the subblock cap derived from it.
- `L1_TENSOR_HEADROOM_BYTES = 128 KB` is a heuristic for other L1-resident buffers. For the sharded regimes
  the budget must subtract the actual input and output shard bytes (Refinements 2 and 4).
- The K-block default (`bk` = one kh block, so 3 K-blocks) is a reasoned departure from "coarsest block
  that fits". Lamp P1 was never measured. It matters where `tensor_kkh_tiles` is tiny (stem.conv1:
  1 tile per K-block). It is folded into Refinement 6.
- Deep-K `8x16x2048 -> 256` solves to `bk = 24` of 192 (24 K-blocks) with bf16 weights and `bk = 32` with
  bf8_b. It is a 4-core, compute-bound shape and not a Panoptic layer (ASPP is BLOCK_SHARDED).

## Registry Conformance

- `INPUT_TAGGERS` (5 taggers, all `(inputs, axes)`), `SUPPORTED` (every gated axis: dtype, weight_dtype,
  layout, memory_layout, fp32_dest_acc_en, bias_mode, activation + every tagger key), `EXCLUSIONS = []`, and
  `validate()` (contract `ValueError` -> SUPPORTED `UnsupportedAxisValue` -> EXCLUSIONS `ExcludedCell`) are
  present and correctly wired. `conv2d_3x3` calls `validate()` first. The op file declares no `INVALID`.
- `prepare_conv2d_weights` / `prepare_conv2d_bias` refuse on every axis derivable from their arguments,
  per the prompt. The bias check, which cannot see C_in, refuses only when no C_in alignment could make
  the cell supported.
- Auto-fix applied: `weight_dtype`, `stride_mode`, `dilation_mode` widened (Code Review #1).
- **INVALID audit** (`eval/golden_tests/conv2d_3x3/feature_spec.py`): one entry,
  `{dtype: bfloat8_b, layout: ROW_MAJOR}`. It is the canonical entry, both axes describe the activation,
  and the impossibility is structural (a block-quantized format has no row-major form). There are no
  cross-tensor entries, no "not built yet" entries, and the op is not norm-like (no weight
  canonicalization cells). Clean.

## L1 Ledger Audit

- **Currency.** All 7 CBs the descriptor declares have rows. Size expressions match `create_program_descriptor`
  exactly (`cb_bias` only when `has_bias`). Worked values reproduced on device:

  | Shape | cores | bm | bk / kkh | slab px | total (bf16 w) | total (bf8_b w) |
  |-------|-------|----|----------|---------|----------------|-----------------|
  | 16x34x32 -> 32 p0 | 14 | 1 | 3 / 3 | 102 | 46 KB | 39 KB |
  | 32x64x128 -> 128 p1 | 64 | 1 | 12 / 12 | 102 | 329 KB | 235 KB |
  | 32x64x128 -> 128 p1 s2 | 16 | 1 | 12 / 12 | 193 | 352 KB | 258 KB |
  | 32x64x128 -> 128 p18 d18 | 64 | 1 | 12 / 12 | 204 | 355 KB | 261 KB |
  | 8x16x2048 -> 256 p1 | 4 | 1 | 24 / 192 (bf8_b: 32) | 66 | 1256 KB | 1072 KB |

  Budget: `get_max_worker_l1_unreserved_size()` 1530976 B - 128 KB = 1399904 B.
- **Capacity vs live set.**
  - Over: none. Each 2x gap (`cb_im2col_rm`, `cb_weights`, `cb_output_tiles`) is a stated double buffer.
    `cb_im2col_tiles` (`bm*bk`) is the matmul's whole-block wait. `cb_matmul_partials` is exactly
    `bm*bn` (mechanism cap #2).
  - Under: none. The slab spans M, kh, kw and c, and its capacity scales with all of them.
- **Page format vs DEST.** The only Float32 page is `cb_matmul_partials`, with `fp32_dest_acc_en=True`.
  That is correct width, not over. No 16-bit accumulator page.
- **Disjoint lifetimes.** Every row carries a sharing justification. The one candidate alias
  (`cb_im2col_tiles` with `cb_output_tiles`) is rejected for cross-block pipelining. No defect.
- **Bounds.** Every non-block capacity symbol has a bound and its predicate
  (`c_in_p <= 2048`, `bn <= 16`, `band_len_px` closed form). The total is closed-form.
- **Data-movement budget.** Present and consistent with the code. Input ~1 + halo crossings, weights
  `num_m_groups` crossings (each core streams all `K_p x C_out_p`), output 1. The cheapest-traffic split
  (M + weight broadcast, R1) is a `deferred` regime row with its delta, and the reason is prompt-directed.
  It is filed as Refinement 3 (perf).
- **Block-size defaults.** Interleaved: full grid first, then the coarsest `bm` (`= core_m_tiles`) ✓.
  `bk` departs from the coarsest choice by design (see Advisories, P1). The deep-K shape goes below the
  tap floor only because one kh block of weights (2 x 1.5 MB double-buffered) cannot fit. The inventory
  table was settled before the solve.
- **Per-core footprint.** `F = slab(bm; s, d, W) * C_in_p*2 + 2*bk*T_bf16 + bm*bk*T_in0 + 2*bk*bn*T_w +
  bn*T_w + bm*bn*T_fp32 + 2*sbh*bn*T_out`.
  - `bm` scales the slab (affine), im2col tiles and partials.
  - `bk` scales the im2col RM and tile CBs and the weights.
  - `bn` scales the weights, bias, partials and output.
- **Findings filed.** No standalone entries. The sharded refinements (2, 4) must update the ledger for
  shard-backed `cb_output_tiles` and shard bytes in the budget. Refinement 3 changes only the
  `cb_weights` producer.

## Precision Baseline

`test_conv2d_3x3_precision_baseline.py` (HiFi4, fp32 DEST, with bias, bf16 activation, fp32 torch
reference on the same bf16 inputs). ULP is bf16 ULP against the bf16-rounded reference. The ratio is
`actual / expected` over |ref| > 1e-3 x max.

| Shape | Weights | PCC | Max abs | Mean abs | Rel RMS | ULP median / p99 | Ratio median [p5, p95] |
|-------|---------|-----|---------|----------|---------|------------------|------------------------|
| 8x8x32 -> 32 p1 | bf16 | 0.9999986 | 0.145 | 0.0185 | 0.00169 | 0 / 6 | 1.00014 [0.9966, 1.0034] |
| 8x8x32 -> 32 p1 | bf8_b | 0.9999687 | 0.510 | 0.0996 | 0.00793 | 1 / 104 | 1.00011 [0.9548, 1.0458] |
| 16x16x64 -> 128 p1 | bf16 | 0.9999985 | 0.280 | 0.0268 | 0.00171 | 0 / 4 | 1.00001 [0.9967, 1.0034] |
| 16x16x64 -> 128 p1 | bf8_b | 0.9999691 | 0.739 | 0.142 | 0.00786 | 1 / 109 | 1.00002 [0.9530, 1.0463] |
| 32x64x128 -> 128 p6 d6 | bf16 | 0.9999986 | 0.471 | 0.0360 | 0.00170 | 0 / 4 | 1.00004 [0.9967, 1.0034] |
| 32x64x128 -> 128 p6 d6 | bf8_b | 0.9999694 | 1.231 | 0.191 | 0.00782 | 1 / 91 | 1.00004 [0.9532, 1.0466] |
| 8x16x2048 -> 256 p1 (K = 18432) | bf16 | 0.9999985 | 2.040 | 0.150 | 0.00171 | 0 / 3 | 1.00002 [0.9967, 1.0034] |
| 8x16x2048 -> 256 p1 (K = 18432) | bf8_b | 0.9999698 | 4.284 | 0.784 | 0.00777 | 1 / 97 | 1.00004 [0.9546, 1.0468] |

**Assessment.**
- bf16 weights: the error is the bf16 output rounding. 85% of outputs are bit-exact against the
  bf16-rounded reference, the median ULP is 0, and relative RMS is ~0.0017 independent of K. That means
  fp32 accumulation holds across 18432-deep K with no growth.
- bf8_b weights: the error is the weight quantization (relative RMS ~0.0078, flat in K).
- Every ratio median is 1.0000 with a symmetric spread, so there is no scale or structural bug.
- Max ULP (~33k) is an artifact of the ULP metric on sign flips of near-zero outputs. Use the median and
  p99.

**Recommended tolerances** (current golden gates are 0.997/0.04 for bf16 and 0.99/0.12 for bf8_b, with
large margin): bf16 weights PCC >= 0.9999 and rel-RMS <= 0.005; bf8_b weights PCC >= 0.9995 and
rel-RMS <= 0.02.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/conv2d_3x3/` -> 88 passed / 1976 (0 failed, 0 errors,
489 skipped, 0 hangs); `python3 -m eval.verify_supported ... ttnn.operations.conv2d_3x3`:

- supported_pass: **64** (was 12 before the drift fix)
- xfail_expected: 1399. Every cell maps to a refinement:
  - 416 INTERLEAVED -> Refinement 1
  - 494 HEIGHT_SHARDED (14 LOOSE) -> Refinement 2
  - 489 BLOCK_SHARDED (9 LOOSE) -> Refinement 4
- invalid_skipped: 480
- no_axes_found: 33. These are `test_regression.py` (non-registry): 24 pass; the 9 C_in = 8 tap-shift
  cases skip until `c_in_unaligned` lands.
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

TARGET - SUPPORTED after this pass: `dtype: bfloat8_b`, `layout: TILE`, `memory_layout: HEIGHT_SHARDED,
BLOCK_SHARDED`, `activation: relu`, `channel_alignment: c_in_unaligned`. All are queued in `op_requirements.md`.
None are covered by INVALID. No omissions.

## Measured Performance (implementer's Phase 0 numbers, Blackhole p150b, HiFi4, bf16, with bias)

| Shape | cores | kernel | binding stage |
|-------|-------|--------|---------------|
| 32x64x128 -> 128 | 64 | 97.7 us | weight stream (18.4 MB at ~190 GB/s); 24.8 us with weights stubbed |
| 8x16x2048 -> 256 | 4 | 468 us | compute + weights on 4 cores |
| 128x256x64 -> 64 | 103 | 225 us | per-pixel DRAM slab fill (DRAM-transaction-bound) |

## Recommendations

- **Queue order** (op_requirements.md):
  1. Refinement 1: TILE + bf8_b activation + ReLU + unaligned C_in, interleaved.
  2. Refinement 2: HEIGHT_SHARDED.
  3. Refinement 3, perf: weight multicast on the HEIGHT Panoptic layers.
  4. Refinement 4: BLOCK_SHARDED, K-split + reduce-scatter.
  5. Refinements 5 and 6, perf: BLOCK layers; stem.conv1.
- No LOOSE case carries an `attention:` perf flag, so the perf targets are free-selected. All are
  Panoptic acceptance layers in their real config.
- **Weight traffic is the dominant cost to expect on the HEIGHT layers.** Every core streams the full
  weight tensor. `semantic decoder.res2.fuse_conv.0` is 705 KB bf8_b x 103 cores = 72 MB per call. At
  the measured ~190 GB/s aggregate that is roughly 380 us of weight reads, more than the compute.
  Refinement 3 exists for this. Refinements 1 and 2 must not foreclose it: the `cb_weights` quantum
  (`bk*bn`) and depth are the multicast receiver contract.
- **One-dispatch mandate.** No refinement may fall back to a `ttnn.to_layout` / `ttnn.tilize` /
  `ttnn.to_memory_config` wrapper, not even as a partial tick. TILE input, sharded I/O and lane zeroing
  are in-kernel only.
- **L1 headroom on the large HEIGHT layers.**
  - stem.conv3 (38 tile-rows per core, C_out = 128) is expected to need `bm < core_m_tiles` once its
    shards occupy L1. That is the first shape where regime R5 (weights resident across M-blocks) pays.
  - `semantic decoder.res2.fuse_conv.0` needs `bk` cut to the 9-tile tap floor (a 480 KB slab at
    C_in_p = 288). The solve already handles both cases; Refinement 2 must subtract the shard bytes from
    the budget.
