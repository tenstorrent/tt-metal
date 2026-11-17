# Changelog: conv2d_3x3

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime R0 (interleaved-slab), built as designed:
  - M is split row-major over the live grid (11x10 on the measured p150b), one M-block per core on every acceptance shape.
  - K is walked per kh block (`bk = tensor_kkh_tiles`; the deep-K case solves down to 24 of 192 tiles with bf16 weights).
  - Per-core weight stream, fp32 packer-L1 accumulation, row-broadcast bias.
  - CB set, sizes and data-movement budget match `l1_ledger.md`.
- **SUPPORTED at Phase 0**:
  - dtype=[bfloat16], weight_dtype=[bfloat16, bfloat8_b], layout=[ROW_MAJOR], memory_layout=[INTERLEAVED], fp32_dest_acc_en=[True].
  - bias_mode=[with_bias, no_bias], activation=[none], channel_alignment=[tile_aligned], hw_alignment=[hw_aligned].
  - stride_mode=[unit, strided], dilation_mode=[unit, dilated], padding_mode=[valid, same].
- **Accuracy achieved** (HiFi4, fp32 DEST, with bias; 4 shapes via `test_conv2d_3x3_precision_baseline.py`):
  - bf16 weights: PCC >= 0.9999985, max_abs_err <= 2.04 (K = 18432), rel_rms_err ~0.0017, median bf16 ULP 0.
  - bf8_b weights: PCC >= 0.99996, rel_rms_err ~0.0078.
  - The got/true ratio median is 1.0000 on every shape.
- **Golden suite at Phase 0**: 64 / 64 supported cells passing (1399 xfail_expected, 480 invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode) per `golden_results/verifier_report.json`.
- **Issues encountered**:
  - Verifier drift fix: SUPPORTED under-claimed `weight_dtype=bfloat8_b`, `stride_mode=strided` and `dilation_mode=dilated`. The kernel already handled all three, and a gate-lifted probe passed 80/80 interleaved RM bf16 cells. They are promoted, taking supported_pass from 12 to 64.
  - Verifier contract fix: `validate()` now rejects a bias whose dtype differs from the weight's, because `cb_bias` pages are sized from the weight tile.
  - `c_in_unaligned` also passed the probe but stays unsupported. The reader copies DRAM page-alignment lanes without zeroing them (Refinement 1).
- **Tests added**: test_conv2d_3x3.py (+ `test_bias_dtype_contract_violation`), test_conv2d_3x3_precision_baseline.py, test_conv2d_3x3_perf.py

### Implementer notes (departures from the design text, advisory)
- The slab fill's page runs use `num_contiguous_pages` only when `contiguous_page_stride() == 1`. On an interleaved accessor a "run" is pages `p, p + num_banks, ...` (same bank, not consecutive pixels), so a contiguous slab destination can take only one page per read there.
- Self-aimed gather and zero reads use `noc_async_read(get_noc_addr(l1), ...)`, with the local_copy helpers' trid calls for fencing. Outstanding reads per trid are throttled below `NOC_MAX_TRANSACTION_ID_COUNT`.

### Measured (Blackhole p150b, HiFi4, bf16, with bias; DEVICE KERNEL DURATION)

| Shape | cores | kernel | weights stubbed | binding stage |
|-------|-------|--------|-----------------|---------------|
| 32x64x128 -> 128 | 64 | 97.7 us | 24.8 us | weight stream: 18.4 MB at ~190 GB/s (DRAM peak) |
| 16x16x64 -> 128 | 8 | 16.2 us | 10.8 us | mixed |
| 8x16x512 -> 256 | 4 | 126 us | 68.8 us | weights + compute (4 cores) |
| 8x16x2048 -> 256 | 4 | 468 us | 258 us | compute (4608 tile-MACs/core at HiFi4) + weights |
| 64x128x128 -> 128 | 86 | 134 us | 80 us | slab fill + weights |
| 128x256x64 -> 64 | 103 | 225 us | 212 us | slab fill: 835 x 128 B DRAM reads/core (halo 2.6x) |

Slab-fill classification: sourcing the same reads from local L1 takes 128x256x64 from 212 to 88 us
(weights stubbed). The fill is therefore DRAM-transaction-bound in aggregate, not RISC-issue-bound.

Lamps measured (kept at design defaults):
- P9 bank stagger (`SLAB_READ_STAGGER_PX_PER_CORE`, parked at 0): flat on every shape.
- P2 `im2col_rm_depth_rows = block_m_tiles`: flat (225 -> 218 us at best).
- P3 `core_m_tiles x2`: mixed.
  - Faster: 32x64x128 97 -> 66 us, 64x128x128 134 -> 119 us.
  - Slower: 128x256x64 225 -> 242 us, and the 4-core shapes.
  - Weight-bound shapes want R1 (weight multicast, Refinement 3), not fewer cores.

## Refinement 1 — TILE / bf8_b activation, fused ReLU, unaligned C_in (interleaved path)
- **Date**: 2026-11-17
- **What was done**:
  - **TILE input (regime R3).** The reader plans each M-block's slab as the three kh bands rounded out to 32-pixel input tile-rows; a band that starts inside the previous band's tile-rows shares them. It streams those tile-rows in slab order into the new `cb_input_tiles`, `ub` tile-rows per push with one trid fence per push. Compute untilizes them (`untilize` helper, one tile-row of `Ct` tiles per block) into `cb_halo_slab`, which is now a compute->reader CB. The reader waits per band and pops the whole slab after the last K-block. Gather, tilize and matmul are shared with ROW_MAJOR; the only layout difference in the gather is the per-kh slab base. The full-grid M split and blocking are unchanged.
  - **bf8_b activation.** `cb_input_tiles`, `cb_im2col_tiles` and `cb_output_tiles` take the activation dtype. The untilized slab is bf16. Im2col re-quantization is exact because tap boundaries are multiples of `C_in_p` (32-aligned on TILE).
  - **ReLU.**
    - With bias: `pack_relu_config(zero)` around `add_bias_bcast_rows`.
    - Without bias: `LastBlockTarget::OutWithRelu`. The tilize `PreKBlockFn` clears PACK_RELU before tilizing and re-arms it on the last K-block, because the helper sets it before the hook runs.
    - PACK_RELU is cleared after every M-block's epilogue. Packer L1-acc is cleared before the untilize.
  - **Unaligned C_in.** The gather copies only the `C_in` real channels of a tap; lanes `C_in..C_in_p` never reach an im2col row. With one K-block per kh (the default), every K-block has the same column layout, so `cb_im2col_rm` is zeroed once at kernel start (`noc.async_write_zeros`), and lanes and kh-block K padding stay zero. With sub-kh K-blocks, per-row zero-fills cover them. Cost: unaligned-C_in shapes use per-tap reads instead of the combined d=1 read. New contract: `C_in % 8 == 0` (per-tap L1 copies are whole 16 B words; TARGET's C_in values are 8, 16, 48 and multiples of 32).
  - **L1 solve.** `untilize_batch_rows` (`ub`) is a new knob. It defaults to the tile-rows one band touches and is the first extent the solve shrinks.
  - **SUPPORTED.**
    - dtype += bfloat8_b; layout += TILE; activation += relu; channel_alignment += c_in_unaligned.
    - Also channel_alignment += c_out_unaligned, both_unaligned. Done-when requires the C_in = 8 tap tests, which use C_out = 8. A gate-lifted probe (C_out 8 / 40 / 48 across RM bf16, TILE bf16, TILE bf8_b, bias, ReLU) passed with no kernel change, because prepared weight and bias padding columns are zero.
- **Accuracy achieved** (HiFi4, fp32 DEST, with bias; `test_conv2d_3x3_precision_baseline.py`, 4 shapes up to K = 18432):
  - TILE bf16 is bit-identical to ROW_MAJOR bf16 (PCC 0.9999985, rel-RMS 0.0017, median ULP 0).
  - bf8_b activation x bf16 weights: PCC >= 0.99994, rel-RMS <= 0.0112.
  - bf8_b x bf8_b: PCC >= 0.99991, rel-RMS <= 0.0135 (gate 0.99 / 0.12).
- **Golden test progress**:
  - INTERLEAVED `test_op`: 480 / 480 pass (all 416 cells this refinement targets plus the 64 prior), 160 INVALID skipped.
  - `test_regression.py`: 33 / 33 pass, including the 9 C_in = 8 tap shifts.
  - HEIGHT / BLOCK cells stay xfail (Refinements 2 and 4).
- **Measured** (Blackhole, DEVICE KERNEL DURATION, bf16 weights, with bias; RM bf16 / TILE bf16 / TILE bf8_b, in us):
  - 32x64x128->128: 97.6 / 97.7 / 97.6
  - 16x16x64->128: 16.4 / 16.3 / 16.1
  - 8x16x512->256: 126.0 / 126.0 / 126.1
  - 8x16x2048->256: 468.7 / 462.8 / 461.1
  - 64x128x128->128: 135.2 / 134.1 / 135.3
  - 128x256x64->64: 225.7 / 111.5 / 102.1. The ROW_MAJOR slab fill is DRAM-transaction-bound (one 128 B page per pixel). TILE reads 2 KB / 1088 B tile pages despite 1.38x tile-row read amplification.
- **Issues encountered**:
  - First TILE solve cost a 12% regression on 8x16x2048 TILE bf16 (526 us). The double-buffered `ub = 3` input window (768 KB at `Ct = 64`) pushed `bk` from 24 down to 3 (192 K-blocks). Making `ub` the first extent the solve shrinks gives `bk = 16` and 463 us.
  - The NaN-lane regressions (TILE input built with `pad_value=nan`, and a ROW_MAJOR input written by a device untilize) cannot fail on Blackhole. With lane exclusion disabled they still pass: the matmul FPU returns 0 for 0 x NaN, 0 x Inf and 0 x 3e38 (plain `ttnn.matmul` behaves the same). The lane exclusion stays as specified; it is defense in depth for an FPU with IEEE products.
  - A bf8_b input with NaN padding at C_in = 8 is already garbage on the host (`from_torch` packs the NaN into the exponent group shared with the 8 real channels; round-trip rel-RMS 0.9999). That is an input-construction limit, not a kernel bug, so the bf8_b NaN test uses C_in = 16.
  - bf8_b-activation outputs show a got/true median ratio of 1.0033 (bf16: 1.0000). The gather/untilize/matmul path is unbiased: a bf16 conv on bf8_b-representable input gives 1.0000. A device typecast bf16->bf8_b shows the same +0.29%, while fp32->bf8_b gives 0.99997. So this is double rounding in the device bf8_b pack, well inside tolerance. Not chased further.
- **Tests added**:
  - `test_conv2d_3x3_layouts.py`:
    - Layout matrix: 7 shapes x {RM bf16, TILE bf16, TILE bf8_b} x {bias, bias+ReLU, ReLU}.
    - NaN padding lanes: TILE bf16 C_in = 8, TILE bf8_b C_in = 16, device-written RM C_in = 8.
    - Multi-M-block ReLU with a pinned one-tile-row blocking. It fails at PCC 0.77 / 0.79 when the post-epilogue PACK_RELU clear is removed.
  - Precision baseline and perf harness parametrized over the activation format.

## Refinement 2 — HEIGHT_SHARDED input and output (regime R2), all layouts
- **Date**: 2026-11-17
- **What was done**: regime R2 as a placement change on the existing R0/R3 pipeline. No new kernel, CB or compute phase.
  - **Grid and work split.** The grid is the input shard grid, in the buffer's shard order (`corerange_to_cores(grid, n, row_wise = orientation == ROW_MAJOR)`). Work index i runs on shard core i and owns output tile-rows `[i*core_m_tiles, ...)`. `core_m_tiles = ceil(tensor_m_tiles / n_shard_cores)` is the same `m_split` as R0, and equals the output shard height `roundup32(ceil(M / n)) / 32`. Cores past the last output shard run nothing.
  - **Output, native placement.** The output is allocated L1 HEIGHT_SHARDED on the input's grid and orientation, shard `[core_m_tiles*32, roundup32(C_out)]`. `cb_output_tiles` is backed on it (`cb_descriptor_from_sharded_tensor`, core range = working cores), so compute packs in place. The writer gets an `output_sharded` CT flag: it streams bias and weights only, and stores nothing. Nothing pops the output CB, so the M-blocks of a core must tile the shard exactly: on a sharded output the solve takes `bm` from the divisors of `core_m_tiles`.
  - **Input.** The reader's slab fill (ROW_MAJOR) and footprint stream (TILE) share one `read_pages` helper. It issues one NoC read per contiguous page run (`num_contiguous_pages` when `contiguous_page_stride() == 1`), from whichever core owns the pages, so a band costs one read per shard it touches. The TILE stream now reads a batch's consecutive tile-rows as runs instead of one read per tile. Interleaved runs stay one page.
  - **L1 budget.** The budget subtracts `RESIDENT_SHARD_SETS = 2` x (input shard + output shard), on top of the 128 KB headroom: this call's pair plus the previous call's. Every enqueue, including a program-cache hit, re-checks the CB region against the lowest live L1 buffer, and the golden harness's second dispatch keeps the first dispatch's shards alive. The shard-backed output CB is not counted in the footprint.
  - **Solve.** After `bm` shrinks, the solve now re-tries the coarsest `bk` above the tap floor at each `bm`, instead of keeping `bk` at the floor. On stem.conv3 this took bm 19 from bk 2 to bk 6: 669 -> 331 us. No interleaved INPUTS shape shrinks `bm`, so no interleaved blocking changed.
  - **Contract.** A HEIGHT_SHARDED input must be in L1 with full-width shards (`ValueError` otherwise). SUPPORTED `memory_layout` += HEIGHT_SHARDED.
- **Accuracy achieved**: same gather/tilize/matmul arithmetic as interleaved. Smoke probe with bf8_b weights, bias, ReLU: RM bf16 PCC 0.99993-0.99996, TILE bf16 0.99993-0.99996, TILE bf8_b 0.99988 on 16x16x64->128, 32x32x8->32 s2, 32x64x64->64 d8. All golden cells pass at their tolerances: bf16 (0.997, 0.04), bf8_b (0.99, 0.12).
- **Golden test progress**:
  - HEIGHT_SHARDED `test_op`: 480 / 480 pass, 160 INVALID skipped.
  - HEIGHT LOOSE_CASES: 14 / 14 pass at the pinned 110-core geometry (stem.conv1 RM bf16 C_in 8 s2; 13 TILE bf8_b layers), both dispatches, program-cache hit.
  - INTERLEAVED `test_op`: 480 / 480 (no regression). `test_regression.py`: 33 / 33.
  - BLOCK_SHARDED stays xfail (Refinement 4).
- **Measured** (Blackhole p150b, DEVICE KERNEL DURATION, Panoptic config: bf8_b weights, bias, ReLU, HiFi4, pinned shards):

  | Layer | cores | bm / bk (of kh block) | device time |
  |-------|-------|-----------------------|-------------|
  | stem.conv1 (RM bf16) | 108 | 38 / 1 (1) | 214 us |
  | stem.conv3 | 108 | 19 / 6 (6) | 331 us (669 us at bk 2 before the solve fix) |
  | res2.x.conv2 | 103 | 10 / 6 (6) | 76 us |
  | res3.0.conv2 (s2) | 86 | 3 / 12 (12) | 83 us |
  | res3.x.conv2 | 86 | 3 / 12 (12) | 83 us |
  | semantic decoder.res2.fuse_conv.0 | 103 | 5 / 9 (27) | 868 us |

  - **fuse_conv.0 is weight-bound.** `bm = 5` of 10 streams its 705 KB weight twice per core, about 145 MB of DRAM reads. Pinned blockings, one dispatch each:
    - bm 10 / bk 9: 506 us. It needs 1136 KB of CBs. Only 1134 KB is left once two shard pairs are live, so it fails the cache-hit re-dispatch.
    - bm 10 / bk 3 (below the tap floor): 751 us.
    - bm 5 / bk 9 (chosen): 868 us.
  - **Next.** Refinement 3's weight multicast removes the doubled DRAM stream. Re-measure the bm 10 vs 5 choice after it lands.
  - **Interleaved path unchanged:** 32x64x128 RM 97.1 us, 128x256x64 TILE bf8_b 103.3 us, TILE bf16 112.1 us, 8x16x2048 TILE bf16 464.0 us. Refinement 1 measured 97.6 / 102.1 / 111.5 / 462.8 us, so within noise.
- **Issues encountered**:
  - No kernel failures: the first HEIGHT probe passed on all three layouts.
  - Tooling: `run_safe_pytest.sh --profile -k "a and b"` loses the `-k` quoting through the tracy wrapper, so pass node ids. One profile run lost its trace when tracy-capture exited with code 3; the identical re-run was clean.
- **Tests added**:
  - `test_conv2d_3x3_height_sharded.py` (29 cases):
    - Layout matrix: RM bf16 / TILE bf16 / TILE bf8_b x 4 shapes, with halo several shards away, RM shards not row-aligned, stride 2, d = 8, C_in = 8 / 48.
    - Output placement asserts.
    - COL_MAJOR orientation on a 3x4 grid at (1, 1), 11 shards on 12 cores, plus a cache-hit re-dispatch. It fails at PCC 0.088 if the shard order ignores the orientation.
    - Multi-M-block shard with ReLU (one-tile-row blocks in a 4-tile-row shard).
  - `test_conv2d_3x3_perf.py`: `test_conv2d_3x3_height_sharded_perf` over 6 Panoptic HEIGHT layers at their pinned geometry.
