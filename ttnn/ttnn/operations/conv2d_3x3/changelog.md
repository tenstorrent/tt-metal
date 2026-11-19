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

## Refinement 3 — Weight multicast for the HEIGHT-sharded Panoptic layers
- **Date**: 2026-11-17
- **What was done**: built regime R1 (weight read-once + multicast) as an overlay on every regime (R0 / R2 / R3), with the multicast helper (`ttnn.Mcast` + `attach()` on the host, `McastArgs` in the writer).
  - **Topology.** One channel, `weight_mcast`, on the writer's NoC 1. Participants are the cores that run the most M-blocks (all but a ragged last core). The fixed sender is work core 0 (`McastExplicitFixedSenderConfig`).
  - **Writer.** Every weight K-block is one round. The sender reserves its `cb_weights` slot, reads the `bk*bn` tiles from DRAM into it and multicasts them to the same slot on every receiver. A receiver reserves its slot, calls `receive()` (ack, then wait for data-ready) and pushes `bk*bn` exactly as the R0 writer does. Participants push the same quanta in the same order, so their write pointers agree every round.
  - **Fallback.** A core outside the channel keeps R0's own DRAM stream. That is the ragged last core, or every core when the channel would have fewer than `WEIGHT_MCAST_MIN_CORES` (2) participants or more than 3 rectangles. A ragged core inside the channel would owe rounds it never consumes.
  - **Unchanged.** No CB, block extent, buffer depth or compute phase changed. The new L1 is two semaphores per participant. Reader and compute are unchanged.
  - **Knob.** `WEIGHT_MCAST_MIN_CORES` is the single on/off control; a large value restores the byte-identical R0 weight stream. It is on by default because it won on every measured shape.
  - **Reused / added.** Reused: R0's `cb_weights` quantum, depth and push protocol, and the descriptor's core list. Added: the channel and the sender/receiver branch in `issue_weight_k_block`.
- **Accuracy achieved**: the weights are bit-identical copies, so accuracy is unchanged from Refinement 2. The golden tolerances hold: bf16 (0.997, 0.04), bf8_b (0.99, 0.12). Perf harness PCC >= 0.99 on all 14 HEIGHT layers.
- **Golden test progress**:
  - LOOSE_CASES: 14 / 14 HEIGHT pass, 9 BLOCK xfail.
  - `test_op` slice (shapes 32x64x128 incl. s2 / d6 / d18, 8x16x2048, 32x32x8, 16x16x64; every layout, placement and dtype): 336 pass, 168 xfail (BLOCK), 168 INVALID skipped.
  - Unit tests: `test_conv2d_3x3.py` + layouts + precision baseline 112 / 112. HEIGHT 31 / 31; the 29 pre-existing HEIGHT cases plus 32x64x128 RM interleaved are also clean under `--dev` (watcher + NoC sanitizer).
- **Measured** (Blackhole p150b, DEVICE KERNEL DURATION, one fresh-cache run per variant; R0 = `WEIGHT_MCAST_MIN_CORES` parked off, R1 = channel on):

  | Layer (HEIGHT, Panoptic config) | cores | R0 | R1 |
  |---------------------------------|-------|----|----|
  | semantic decoder.res2.fuse_conv.0 | 103 | 864.0 us | 466.5 us |
  | semantic res2.fuse_conv.1 | 103 | 745.7 us | 396.8 us |
  | semantic decoder.res3.fuse_conv.0 | 86 | 391.5 us | 145.2 us |
  | semantic decoder.res3.fuse_conv.1 | 86 | 312.1 us | 110.0 us |
  | instance decoder.res3.fuse_conv.0 | 86 | 198.1 us | 91.7 us |
  | instance decoder.res2.fuse_conv.0 | 103 | 188.3 us | 154.2 us |
  | instance res2.fuse_conv.1 | 103 | 144.9 us | 134.0 us |
  | res3.0.conv2 (s2) | 86 | 78.8 us | 40.0 us |
  | res3.x.conv2 | 86 | 79.1 us | 40.7 us |
  | res2.x.conv2 | 103 | 75.8 us | 76.6 us |
  | center_head.1 | 103 | 79.4 us | 79.4 us |
  | stem.conv1 (RM bf16) | 108 | 214.4 us | 214.3 us |
  | stem.conv2 | 108 | 258.7 us | 258.7 us |
  | stem.conv3 | 108 | 330.6 us | 330.8 us |

  The 14 layers sum to 3961 -> 2539 us (1.56x). The flat layers were not weight-bound.

  - **Interleaved perf matrix** (RM bf16 / TILE bf16 / TILE bf8_b; Refinement 1 numbers -> R1):
    - 32x64x128: 97.6 / 97.7 / 97.6 -> 25.9 / 25.3 / 23.8 us
    - 16x16x64: 16.4 / 16.3 / 16.1 -> 10.7 / 10.2 / 10.3 us
    - 8x16x512: 126.0 / 126.0 / 126.1 -> 79.7 / 79.6 / 79.9 us
    - 8x16x2048 (4 cores): 468.7 / 462.8 / 461.1 -> 282.7 / 289.3 / 288.9 us
    - 64x128x128: 135.2 / 134.1 / 135.3 -> 80.0 / 60.4 / 52.6 us
    - 128x256x64: 225.7 / 111.5 / 102.1 -> 209.6 / 102.9 / 91.4 us
    - Nothing regressed, including the few-core deep-K shape the verifier flagged: a single sender multicasting to 3 cores beats 4 cores each reading 9.4 MB.
  - **Ablation (all weight movement stubbed, only reserve/push kept):**
    - fuse_conv.0: 465.9 vs 466.5 us. res3.0 / res3.x: 40.0 / 41.5 vs 40.0 / 40.7 us. The weight path is fully hidden on the HEIGHT targets.
    - Still exposed: 8x16x2048 TILE bf16 268.5 vs 289.3 us (21 us over 36 rounds); 32x64x128 RM 25.2 vs 25.9 us, TILE bf8_b 20.7 vs 23.8 us; 128x256x64 TILE bf8_b 91.0 vs 91.4 us.
  - **bm re-check** (Refinement 2 asked to re-measure bm 10 vs 5 once R1 landed; extra L1 from budgeting one shard pair instead of two):
    - fuse_conv.0: bm 10 / bk 9 / ub 2 497 us vs bm 5 / bk 9 / ub 7 466 us.
    - fuse_conv.1: bm 10 / bk 12 / ub 2 414 us vs bm 5 / bk 12 / ub 7 397 us.
    - stem.conv3: bm 38 / bk 3 581 us vs bm 19 / bk 6 331 us.
    - With weights multicast, the coarser bm no longer pays. The two-pair budget stays.
- **Bottleneck now**: fuse_conv.0 / fuse_conv.1 are compute-thread-bound. Per core: 6480 / 5760 HiFi4 tile-MACs (bf8_b x bf8_b) plus the tilize and untilize of the slab, about 460 / 400 us. The stem layers and res2.x are bound by the slab fill / gather / untilize path (Refinement 6's lamps), not weights.
- **Next lever, not taken**: injector double-buffering. The sender would read round r+1 while round r multicasts, with depth 3 or a sender-only staging CB and per-round read trids. It would recover at most the 21 us on the 4-core deep-K shape and 1-3 us on the small interleaved shapes. The HEIGHT targets have 0 us of weight time left to recover, so it was not built.
- **Issues encountered**: none on device; the first build passed every test. Note: an ablation run where tests fail on PCC leaves their L1 shards alive (exception frames). Later tests then hit a CB / L1-buffer clash at enqueue (`program.cpp:2691`). Profile each failing variant on its own.
- **Tests added**:
  - `test_conv2d_3x3_height_sharded.py::test_conv2d_3x3_height_sharded_weight_mcast_ragged_core` (RM bf16, TILE bf8_b). One-tile-row M-blocks in 6-tile-row shards over 64 tile-rows: 10 cores take 18 multicast rounds and the ragged last core runs 4 M-blocks. It asserts the channel excludes that core.
  - `test_conv2d_3x3_perf.py`: `test_conv2d_3x3_height_sharded_perf` covers all 14 HEIGHT LOOSE layers (8 added).

## Refinement 4 — BLOCK_SHARDED: K split across core columns + row reduce-scatter (regime R4)
- Date: 2026-11-17
- What was done: built regime R4 as designed (scheme-change: dependent-axis split + cross-core combine), on the existing R2/R3 pipeline.
  - **Placement and work split.** The grid is the input shard grid. `block_grid_shape` reads (rows, cols) from the bounding box and the orientation: ROW_MAJOR runs height shards down y, COL_MAJOR down x. Core (r, c) owns output tile-rows `[r*core_m_tiles, ...)` (`m_split` over the grid rows) and channel slice c. Output: L1 BLOCK_SHARDED on the same grid, shard `[core_m_tiles*32, out_n_tiles*32]` with `out_n_tiles = ceil(tensor_n_tiles / k_cols)`. `cb_output_tiles` is backed on it. Columns past the last C_out tile own no output (empty shards) and only send. A single shard column is plain R2 on that column (`k_cols = 1`, R1 multicast still applies).
  - **Weights.** `prepare_conv2d_weights` lays rows out as `(slice, kh, kw, c)`, one `3*roundup32(3*C_slice_p)` K-block per slice. `padded_k` and the BLOCK contract share `channel_slices()`. The writer offsets its weight reads by `col * tensor_k_tiles`.
  - **Reader.** Reads only its column's slice of each band: row `v` of the slice is page `v * row_page_stride + col * pages_per_slice_row`. One owner shard stores its rows of the slice back to back, so a band costs one read per owner-shard run (pixels for RM, tile-rows for TILE). Gather, untilize and tilize are unchanged; a short last slice (C_in not a multiple of the shard width) gets its own reader kernel with the real channel count, so its lanes stay excluded.
  - **Partials.** Under the split, the matmul packs fp32 partials `[bm, tensor_n_tiles]` with `caller_owns_pack_target`: compute reserves and pushes `cb_matmul_partials` once per M-block, and the WRITER is its only consumer. The helper's per-K-block drain would have made compute a second consumer. PACK_RELU is never armed for the matmul under the split.
  - **Row reduce-scatter**, one round per M-block, two monotonic semaphores per core (`combine_credit`, `combine_arrival`):
    - An owning column reserves its whole `cb_combine_slots` (`k_cols` slots of `[bm, out_n_tiles]` fp32) and credits every row peer.
    - Each core waits for its partials and one credit per owning column. It writes N-slice y into slot `my_col` of column y, fences, and increments y's arrival count.
    - After `k_cols` arrivals the owner pushes the slots.
    - Compute sums them with `reduce<SUM, REDUCE_COL, AccumulateViaAdd, ReduceWithinTile::Skip>`: a `[k_cols, bm*ns]` block, pairwise FPU adds into fp32 DEST, odd counts seeded by a copy.
    - With bias: the sum goes to `cb_combined` (fp32), then `add_bias_bcast_rows` with the core's bias N-slice. ReLU is armed only around that last pack. Without bias, the reduce packs straight into the output shard.
  - **Two-NoC scatter** (`COMBINE_SPLIT_NOCS`, on). The writer sends its own and lower columns on NoC1; the reader sends the higher columns on NoC0, handed over by two core-local counters (`send_go`, `send_done`). The partials CB keeps a single consumer (the writer).
  - **No per-column weight multicast.** R1's single channel is not attached under the split, because each column multiplies a different weight slice.
  - **Contract.** BLOCK input must be in L1, one rectangle, with exactly one grid column per channel slice (`ValueError` otherwise). SUPPORTED `memory_layout` += BLOCK_SHARDED.
  - **Reused / added.** Reused: R2's shard-backed output, sharded slab fill, gather, tilize, matmul and bias-add helpers, and the L1 solve (it never shrinks `bm` on a BLOCK LOOSE layer). Added: the channel-slice row reads, `cb_combine_slots` / `cb_combined`, the writer's combine round and the reader's NoC0 half (one shared scatter helper, `conv2d_3x3_combine_dataflow.hpp`), and the reduce + bias epilogue.
- Accuracy achieved (HiFi4 unless noted, fp32 DEST, bias + ReLU):
  - RM bf16, 16x16x64->128, 4 slices, bf16 weights: PCC 0.999984, rel-RMS 0.0021.
  - TILE bf16, 8x16x256->256, 8 slices: PCC 0.999983, rel-RMS 0.0021. The non-split paths measure 0.0017; the difference is the tf32 operand read of the fp32 partials in the cross-core add.
  - Panoptic bf8_b x bf8_b: res4.x PCC 0.99965 / rel-RMS 0.0155; res4.x at LoFi 0.99950 / 0.0243 (its gate is 0.996 / 0.08); aspp.convs.1 rel-RMS 0.0155.
  - Golden tolerances (0.997, 0.04) / (0.99, 0.12) hold on every BLOCK cell.
- Golden test progress:
  - BLOCK_SHARDED `test_op`: 480 / 480 pass, 160 INVALID skipped.
  - BLOCK LOOSE_CASES: 9 / 9 pass at the pinned grids: res4.0, res4.{1..5} and its LoFi twin, res5.0/1/2 (d 2/4/8), aspp.convs.1/2/3 (C_in up to 2048, d up to 18). Both dispatches, cache hit; also clean under `--dev` (watcher + NoC sanitizer).
  - No regression: HEIGHT LOOSE 14 / 14, the interleaved + HEIGHT d = 18 cells 48 / 48, unit suites + `test_regression.py` 176 / 176.
- Measured (Blackhole p150b, DEVICE KERNEL DURATION, Panoptic config, one fresh-cache run per variant):

  | Layer (BLOCK) | cores | all-on-writer scatter | split NoC scatter (shipped) | weights stubbed | scatter writes stubbed |
  |---------------|-------|-----------------------|-----------------------------|-----------------|------------------------|
  | res4.0.conv2 (s 2) | 64 | 88.7 us | 81.3 us | 88.9 us | 68.3 us |
  | res4.{1..5}.conv2 | 64 | 86.8 us | 78.7 us | 86.5 us | 66.1 us |
  | res4.x @ LoFi | 64 | 67.9 us | 60.2 us | 68.3 us | 48.0 us |
  | res5.0.conv2 (d 2) | 80 | 214.3 us | 210.5 us | 204.2 us | 208.0 us |
  | res5.1.conv2 (d 4) | 80 | 281.2 us | 266.3 us | 281.6 us | 244.5 us |
  | res5.2.conv2 (d 8) | 80 | 280.3 us | 265.9 us | 280.3 us | 244.5 us |
  | aspp.convs.1 | 88 | 400.5 us | 400.8 us | 400.1 us | 396.4 us |
  | aspp.convs.2 | 88 | 401.8 us | 401.5 us | 401.7 us | 398.4 us |
  | aspp.convs.3 (no bias) | 88 | 207.2 us | 205.2 us | 194.6 us | 201.3 us |

  The two ablations were measured on the all-on-writer build.
  - **Weights.** The per-core weight stream is hidden: stubbing it moved at most 13 us (aspp.convs.3), so a per-column weight multicast buys almost nothing on these layers.
  - **Scatter.** The scatter was the exposed cost: 20.7 us on res4.x. Rotating each sender's destination order was flat (88.4 us), so it was not destination incast. Sending only the leftward half cut it to 12.1 us for 56% of the bytes, so time scales with the bytes through one core's NoC interface (in plus out), not with ring distance. Splitting the scatter over both NoCs drops the per-interface peak from 16 to 9 slice units and measured 20.7 -> 12.6 us.
  - **Guard set unchanged** (vs Refinement 3): 32x64x128 RM 26.5 us, TILE bf8_b 23.8 us; 8x16x2048 TILE bf16 291.5 us; stem.conv1 214.5 us; res3.x 40.7 us; fuse_conv.0 466.6 us.
- **Bottleneck now**:
  - res4.x: matmul ~25 us at HiFi4 (19 us more than LoFi), the remaining scatter ~12.6 us, and ~40 us of slab fill, untilize, gather and tilize per core.
  - res5.x / aspp: the per-core K-slice matmul plus the gather path. res5.1/5.2 run `bk` 3 of 6 for L1 (the d = 4 / 8 slab), hence the 266 vs 210 us against res5.0.
- **Next levers** (Refinement 5 scope, not taken here):
  - Pipeline the scatter with the last K-block: an fp32 out-CB pushed per row group, which needs the helper's Out target and a second partials CB.
  - Send bf16 partials, halving the scatter bytes at a precision cost the design does not allow by default.
  - The R4b activation all-gather for res4.x.
- Issues encountered: none on device; every kernel built and passed on its first full run after one compile fix (`ReduceFp32Mode` is global, not in `compute_kernel_lib`).
- Tests added:
  - `test_conv2d_3x3_block_sharded.py` (42 cases):
    - Layout matrix: RM bf16 / TILE bf16 / TILE bf8_b x 5 shapes. Covers halo 8 shards away (d 8), stride 2, a single shard column, columns that own no output, a short last channel slice, a short last N-slice, and 8 slices.
    - ReLU-after-combine: slices whose partials cancel. With ReLU armed on the partials, the `no_bias` cases fail PCC.
    - COL_MAJOR orientation on an offset grid, with a cache-hit re-dispatch.
    - Multi-M-block: 4 combine rounds per row.
    - The writer-only scatter knob setting.
    - Grid-column contract error.
  - `test_conv2d_3x3_perf.py::test_conv2d_3x3_block_sharded_perf`: the 9 BLOCK LOOSE layers. `CONV2D_3X3_PERF_ABLATION=1` skips the PCC check for ablation profiling.

## Refinement 5 — Speed up the BLOCK-sharded res4 layer
- Date: 2026-11-17
- What was done: two perf levers, chosen from a per-RISC zone breakdown and ablations on `res4.{1..5}.conv2 @ LoFi`. No SUPPORTED change.
  - **Diagnosis.** Zones per RISC, LoFi, p50 core, 60.1 us:
    - the reader's gather ran 1.4 -> 38.2 us and waited only 2.6 us of it for CB space;
    - compute's tilize waited on it (unpack: 30 of 40 us of the matmul phase);
    - the weight stream (24 us) was hidden;
    - the combine phase added ~14 us.

    The gather issued one 192 B L1 read per output pixel per kh, 768 per core. The heading's suggested levers did not aim at that bound:
    - weights were measured hidden;
    - R4b would multiply the binding gather and tilize work by `k_cols` = 8;
    - the combine's FPU add is ~1.4 us.
  - **Lever 1: tap-major im2col** (`TAP_MAJOR_IM2COL`, gated by `tap_major_im2col()`: stride 1, `C_in_p % 32 == 0`, no padded lanes in any slice, `bk` whole taps).
    - The reader lays each tile-row out as one `[32 pixels, C_in_p]` block per tap. At stride 1, the pixels of one output image row read consecutive slab pixels, so a tap costs one read per image-row run plus zero reads for padding. On res4.x that is about 100 reads per core instead of 768.
    - Compute tilizes each block `C_in_p / 32` wide, which yields the same tiles in the same `[m][k]` order. The output is bit-identical to the old path (probe: res4.x LoFi / HiFi4, 32x64x128 interleaved; max diff 0).
    - Measured: 60.1 -> 41.7 us (LoFi).
  - **Lever 2: per-column weight multicast** (the heading's named lever).
    - Once the gather was fast, the reader waited 13.9 us for CB space, and the 64-core per-core weight stream became the bound: 5 MB from DRAM in ~21 us. Stubbing it gave 41.7 -> 36.1 us.
    - The R1 channel is now attached under the K split with one group per grid column (`ttnn.Mcast(..., group = participating rows, McastFixedSenderConfig(), ColumnMajor)`; `RowMajor` on a COL_MAJOR grid). The column's first participating row reads its K-slice and multicasts it down the column. A ragged last row with fewer M-blocks streams its own slice.
    - The writer kernel is unchanged: its sender / receiver / fallback branch already serves any group layout. The channel's semaphores follow the four combine semaphores.
    - Measured: 41.7 -> 36.0 us, equal to the weights-stubbed number.
  - **Reused / added.**
    - Reused: the R1 writer branch and `cb_weights` protocol, the gather's slab addressing and zero-fill, and the tilize helper.
    - Added: one reader branch, a tilize-width CT arg, the host predicate, and the column grouping.
- Accuracy achieved:
  - Bit-identical to the pre-refinement output.
  - res4.x LoFi: PCC 0.99954, rel-RMS 0.0186 (gate 0.996 / 0.08).
  - res4.x HiFi4: PCC 0.99969, rel-RMS 0.0103.
  - Golden tolerances (0.997, 0.04) / (0.99, 0.12) hold on every cell run.
- Golden test progress:
  - Slice of all 23 LOOSE cases plus every `test_op` cell of 32x64x128 (p1 and d18), 24x40x32, 16x34x32 and 8x8x32: 383 passed, 120 INVALID skipped, 0 failed.
  - Unit dir: 229 / 229 (3 added).
  - BLOCK unit file plus res4.x / res2.x perf layers clean under `--dev`: 48 / 48.
- Measured (Blackhole p150b, DEVICE KERNEL DURATION, one fresh-cache run per variant; "off" = both levers parked, same session):

  | Shape | off | on |
  |-------|-----|----|
  | res4.{1..5}.conv2 @ LoFi | 60.3 us | 35.2 us |
  | res4.{1..5}.conv2 (HiFi4) | 78.8 us | 54.1 us |
  | res4.0.conv2 (s 2, gather gated off) | 81.5 us | 81.5 us |
  | res5.0.conv2 (d 2) | 211.3 us | 141.5 us |
  | res5.1 / res5.2 (bk splits a tap) | 266.3 / 265.8 us | 266.6 / 265.5 us |
  | aspp.convs.1 / 2 (short last slice) | 400.3 / 401.0 us | 400.0 / 400.4 us |
  | aspp.convs.3 (column multicast only) | 205.3 us | 187.4 us |
  | HEIGHT stem.conv2 / stem.conv3 | 264.4 / 330.9 us | 126.0 / 201.9 us |
  | HEIGHT res2.x / res3.x / center_head.1 | 76.5 / 40.8 / 79.1 us | 38.1 / 33.3 / 45.5 us |
  | HEIGHT sem res2.fuse_conv.0 / inst res2.fuse_conv.1 | 466.6 / 132.7 us | 392.3 / 98.1 us |
  | HEIGHT stem.conv1 (RM, C_in 8, s 2: gated off) | 214.4 us | 214.5 us |
  | interleaved 128x256x64 RM / TILE bf16 / TILE bf8_b | 210.0 / 102.5 / 93.8 us | 185.7 / 64.1 / 52.4 us |
  | interleaved 64x128x128 RM / TILE bf16 / TILE bf8_b | 80.4 / 60.8 / 52.3 us | 76.1 / 52.4 / 44.1 us |
  | interleaved 32x64x128 RM / TILE bf8_b | 25.8 / 24.2 us | 24.4 / 23.7 us |
  | interleaved 8x16x2048 TILE bf16 (few-core deep K) | 291.6 us | 291.6 us |

  None of the 41 perf-harness shapes regressed (worst +0.5%, noise).
- **Bottleneck now** (res4.x LoFi, 35 us; zones plus a scatter-writes-stubbed ablation):
  - 3.4 us slab untilize.
  - ~18 us of compute-bound matmul phase (tilize ~8 us + matmul ~10 us; the reader waits 7.7 us for CB space).
  - ~14 us of combine. The partial scatter's bytes are 11.6 us of it: stubbed, 35.2 -> 23.6 us (HiFi4 54.1 -> 42.4 us). The direct all-to-all puts 16 slices (512 KB) per direction on the middle link of each 8-core row.
- **Next levers, not taken** (each restructures the combine protocol; the heading scopes one ⭐⭐⭐ lever per phase):
  - Pipeline the scatter under the last K-block by pushing the partials per row group.
  - Use a reduce-then-forward topology (e.g. recursive halving nearest-first: ≤128 KB per link per step).
  - bf16 partials would halve the bytes, but that is a precision change.
  - Gather gates: res5.1/5.2 need a tap-aligned `bk` preference in `select_blocking` (bk 3 splits a 2-tile tap); aspp needs per-slice-width compute kernels (its short last slice forces the row layout on the shared compute kernel).
- Issues encountered: none on device; both levers passed every test on their first run.
- Tests added: `test_conv2d_3x3_block_sharded.py::test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_row` (RM bf16, TILE bf16, TILE bf8_b).
  - One-tile-row M-blocks over 3 grid rows of 7-tile-row shards, 20 output tile-rows: rows 0-1 take 21 multicast rounds per column, interleaved with 7 combine rounds.
  - The ragged last row (6 M-blocks) must stay out of its column's channel.
  - W_out = 40, so tap-major runs straddle image rows.

## Refinement 6 — Speed up stem.conv1
- Date: 2026-11-17
- What was done: stem.conv1 (HEIGHT RM bf16, C_in 8, stride 2, 108 working cores, bf8_b weights, bias + ReLU, HiFi4) went from 214.5 to 24.7 us. No SUPPORTED change.
  - **Diagnosis** (DEVICE KERNEL DURATION + zones + ablations, Blackhole p150b).
    - Stubbing the gather reads: 214 -> 128 us. Matmul `SKIP_COMPUTE` on top: no change, so math was not the bound.
    - With reads stubbed, zones showed ~21-25 us per K-block of reader row loop and single-tile tilize, ping-ponging through the 2-deep `cb_im2col_rm`.
    - The cost was per pixel: 3648 x 48 B gather reads per core (~57 ns each), a per-row address loop, and 114 one-tile tilizes, all paid 3 times per M-block (`tensor_kkh_tiles = 1`).
  - **Face-direct im2col** (`FACE_DIRECT_IM2COL`; gated on `stride * C_in_p == 16`, bf16 RM activation, no padded lanes, one tap per face row unless d = 1).
    - One output pixel steps the slab by exactly one 16-element tile-face row, so a face's 16 im2col rows are one contiguous 512 B slab run.
    - The reader writes the bf16 in0 tiles itself, 4 reads per tile, and compute drops the tilize: no PreKBlockFn, Reverse-order startup.
    - A face holding tap 2 also reads the next pixel into its K-padding lanes, against zero weight rows. That is real input read only inside the image; outside, those lanes are zero-filled. The band reaches one tap further (`band_last_tap`) so that pixel is in the slab.
    - A face made only of K padding (C_in 16) is zeroed once, because every K-block starts at the CB base.
    - The output matches the old path on all but < 0.001% of elements, which differ by at most 0.0039 (accumulation order).
  - **P1: whole-K K-block** (`FACE_DIRECT_WHOLE_K_BLOCK`). `bk = tensor_k_tiles` is the first L1-solve candidate on face-direct shapes: one matmul init, no L1-acc passes. Under a K split face-direct also requires one K-block, because only the tilize ordered the L1-accumulating packs there.
  - **Interior fast path.** One in-image test per 16-pixel row group, then fixed-offset reads; edge groups take the per-run/per-tap path. Top-padding rows zero whole faces.
  - **P7: stateful reads** for the 512 B interior faces (`set_read_state` once, `read_with_state` per face).
  - **Split slab fill by NoC** (`SLAB_SPLIT_NOCS`, HEIGHT RM).
    - High-index cores' footprints lie 1-3 shards ahead. A NoC0 read response from a higher-index core wraps the ring, so band waits reached 20-25 us there against ~1.5 us on low cores.
    - The reader reads runs owned by its own and lower shards. The writer reads higher shards' runs on NoC1 into the same slab: `slab_fwd_done` writer -> reader, `slab_free` reader -> writer per M-block.
    - The writer issues these reads before its bias prefetch.
  - **Split gather** (`GATHER_SPLIT_RISCS`). The gather is RISC-issue-bound (its trid wait was 0), so the writer gathers every odd row group on NoC1. The reader reserves the block, raises `gather_go`, gathers the even groups and pushes in order, waiting on `gather_done` for the writer's groups.
  - **In0 streaming** (`IN0_STREAM_TILE_ROWS = 2`, whole subblocks, ragged last group allowed).
    - The reader pushes in0 per row group, and compute runs one `matmul_block` call per group with the weights kept fronted (`WaitAndRetainOnLastBlock`, popped once).
    - The bias add runs once per `BIAS_STREAM_GROUPS = 5` groups, so only the last chunk trails the gather.
  - **Reused / added.**
    - Reused: the slab/band plan (now the shared `conv2d_3x3_slab_dataflow.hpp`), the R1 weight channel, `matmul_block` and `add_bias_bcast_rows`, the shard-backed output.
    - Added: `conv2d_3x3_gather_dataflow.hpp` (tagged reads + face gather, shared by reader and writer), four core-local semaphores on HEIGHT RM, and the knobs above.
- Measured (DEVICE KERNEL DURATION, stem.conv1, one fresh-cache run per step):

  | Step | us |
  |------|----|
  | before (Refinement 5 code) | 214.5 |
  | face-direct im2col + whole-K block | 99.5 |
  | + interior fast path | 63.7 |
  | + stateful face reads | 60.6 |
  | + slab fill split by NoC | 53.2 |
  | + gather split reader / writer (halves) | 36.3 |
  | + in0 streaming, alternating groups | 27.1 |
  | + forward slab reads before the bias prefetch | 26.2 |
  | + bias add per 5 groups | 24.7 |

  Measured and not kept:
  - Bias add per group: 29.1 us (compute fell behind). Per 3 groups: 24.8 us; per 10 groups: 25.5 us.
  - `InitMode::None` + no reconfig between consecutive matmul calls: 24.4 us, inside noise, so the simpler single instantiation stays.
  - 4- and 6-tile-row groups: 27.5 / 27.7 us.
- Guard set, same-session baseline (Refinement 5 commit) -> now, us:
  - aspp.convs.1 400.0 -> 404.1, aspp.convs.2 400.2 -> 401.9, aspp.convs.3 187.2 -> 186.5.
  - res2.x 37.8 -> 38.0, center_head.1 45.4 -> 44.5, instance res2.fuse_conv.1 98.8 -> 100.1.
  - 32x64x128 TILE bf8_b 24.3 -> 24.2.
  - The other 33 perf-harness shapes are within ±2% of Refinement 5's numbers (32x64x128 RM 24.1, 8x16x2048 TILE bf16 290.6, res4.x LoFi 35.3, stem.conv2 / conv3 127.0 / 203.1 us).
  - A first build that moved the reader's zero-fill address into the shared read struct cost aspp.convs.1 +1.8%, reproducible over 3 runs. The row-gather paths keep their kernel-local zero address.
- **Bottleneck now**:
  - Compute: with the face reads stubbed the op still takes 24.0 us. That is the HiFi4 matmul (228 tile-MACs, ~11 us), the bias add (3.3 us), and a ~0.36 us fixed cost per streamed matmul call (19 calls).
  - Dataflow ends at ~22 us: ~4 us of slab fill exposed (`slab_fwd_done` wait), then ~17 us of gather on two RISCs at ~84 cycles per 512 B face read (cmd-buffer poll + 3 register writes + loop).
- **Next, not taken**:
  - A progressive per-subblock in0 wait inside one `matmul_block` call (helper change) would stream without the per-call cost.
  - Folding the bias into the matmul would drop the 3.3 us bias pass.
  - Both halves bind within ~2 us of each other, so either alone moves the wall by at most that.
- Accuracy achieved: stem.conv1 Panoptic config, C_in 3 padded to 8: PCC > 0.999999, rel-RMS 0.0079, max abs 0.236 vs the fp32 torch reference with bf16 weights (bf8_b weight quantization dominates); same as the old path. Golden tolerances hold on every cell run.
- Golden test progress:
  - Target slice (every sharded ROW_MAJOR cell of 32x32x8 s2, 16x16x8, 16x16x16, HEIGHT RM d18 / 24x40 / 32x32 d2 s2, plus stem.conv1 LOOSE): 73 passed, 72 INVALID skipped.
  - Cross-path slice (all 23 LOOSE cases plus every `test_op` cell of 32x64x128 p1 and 8x8x32): 167 passed, 48 skipped, 0 failed.
- Issues encountered: none on device. One compile failure (division by a zero CT shard-row count in code compiled on every path) and test-side dataclass construction were fixed before the first device run.
- Tests added:
  - `test_conv2d_3x3_height_sharded.py::test_conv2d_3x3_height_sharded_face_direct` (9 cases): C_in 8 s2 and C_in 16 s1 (K-padding-only faces), with an odd row-group count and a shorter last group, bias+ReLU / ReLU / bias. Each case asserts the face-direct, whole-K, streamed blocking.
  - `::test_conv2d_3x3_height_sharded_face_direct_multi_m_block`: two M-blocks per shard, exercising the slab handshake across blocks.
  - Disabling the writer's higher-shard reads fails these tests.

## Perf 1
- Date: 2026-11-18
- Scope: perf tournament round 1 on the six `[perf]` LOOSE cases (`feature_spec.py`): Panoptic config at LoFi, fp32 DEST, bf8_b activation and weights, bias + ReLU, pinned shard grids. No SUPPORTED change. Board: Blackhole p150b, AICLK 1350 MHz. All numbers are DEVICE KERNEL DURATION from one fresh-cache run per variant, unless marked as a median.
- Instrumentation (permanent): `MaybeDeviceZoneScope` zones in all three kernels. Waits are split from work where a wait can be hoisted (`*_reserve`, `*_wait_*`, `compute_combine_wait`). Tools:
  - focus harness: `test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus`;
  - zone report: `perf_experiments/zone_report.py`;
  - ablation hooks: `perf_experiments/ablation/` (a patch against `acc3d2e23e2`).
  - Under the profiler, the reader zones cost ~5% on the old per-pixel row gather (res5.1 211 vs 222 us); the compute zones are free. The before/after table below is unzoned (zones stripped for the run).

### Measured breakdown (before)
- All six cases ran the per-pixel row gather. Tap-major was gated off in each:
  - res5.x: bk 3 split a 2-tile tap;
  - aspp: the short last channel slice;
  - sem res2.fuse_conv.1: bk 12 split an 8-tile tap;
  - res4.0: stride 2.
- Zones (p50 core, zoned):
  - res5.1: `reader_gather` 188 of 222 us; compute's tilize waited on it.
  - aspp1: gather 198 us p50, 276 us on the short-slice column, which the whole row's combine waited on (`compute_combine_wait` 76 us).
  - sem_res2: gather 143 us and weight-multicast wait 149 us overlap.
  - res4.0: gather 38 us, combine ~14 us, untilize 9 us.
- Cumulative ablation peel (zoned us; full > gather > matmul > weights > scatter > fill):

  | case | full | −gather | −matmul | −weights | −scatter | −fill |
  |---|---|---|---|---|---|---|
  | res5.1 | 221.9 | 69.1 | 67.0 | 66.7 | 46.3 | 46.1 |
  | aspp1 | 297.7 | 96.6 | 83.3 | 82.3 | 62.1 | 58.5 |
  | sem_res2 | 191.4 | 125.0 | 92.3 | 77.7 | 77.0 | 69.2 |
  | res4.0 | 63.4 | 39.0 | 38.9 | 38.8 | 27.1 | 26.8 |

  - Weights were hidden once the gather was gone (`gather,weights` equals `gather` on all six), so no weight idea was floated.
  - With the gather stubbed, `SKIP_COMPUTE` moved res5.1 by only 2 us: the K-blocked walk there was bound by the fp32 packer-L1-accumulation passes (6 x 112 tiles), not by math.
  - The R4 scatter was a serial tail: 20 us on res5.x and aspp, 12 us on res4.0.
- Ranked: (1) row gather, 66-201 us exposed; (2) R4 combine tail, 12-20 us; (3) K-blocked L1-acc pack passes; (4) the stride-2 gather (res4.0).

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Precision was identical in every option: the outputs are bit-identical to the old path unless noted.

| idea | verdict | focus result (us, isolated) | domain |
|---|---|---|---|
| `tap_aligned_blocking`: L1 solve keeps only tap-aligned bk when tap-major is otherwise eligible | WIN | res5.1 211.3 -> 89.2, res5.2 210.2 -> 89.7, sem_res2 189.5 -> 136.9 | everywhere tap-major can run; 8x16x512 interleaved flat; bm re-sweep and single-buffered weights measured slower except res5 (option B, not taken) |
| `short_slice_tap_major`: zero a short channel slice's padded lanes once, never read them | WIN | aspp1 293.9 -> 109.6 (A) / 101.6 (C: + 32-pixel zero block for out-of-image runs), aspp2 291.5 -> 109.4 / 101.0 | every BLOCK short slice, TILE and RM. Exception (inexpressible): TILE short slice whose real channels are not whole tiles |
| `split_row_gather`: the writer gathers the units the reader hands it (dynamic queue 2, ring depth 8, `noinline` row gather) | WIN | res4.0 63.3 -> 47.4 (median of 3) | every row gather with > 1 tile-row per K-block: res3.0 -4.7%, interleaved C_in 48 -14.6%, HEIGHT TILE C_in 48 -36%; flat elsewhere |
| `nslice_outer_ksplit`: K split walks owner N-slices outermost (whole-K DEST accumulation, slice y's send overlaps slice y+1's matmul) | WIN with exceptions | res5.1 222.0 -> 47.6 (it also frees the L1 that lets bk 6 run tap-major), res4.0 63.4 -> 56.0, res4.x LoFi 35.9 -> 25.5 | every K split whose resident `[bm, K]` in0 fits. Exceptions: aspp1/2 (inexpressible: in0 1.57 MB > budget), aspp.convs.3 HiFi4 +6% (measured regression while its gather was the slow row gather) |
| `l1acc_pack_hoist`: issue the pack format / L1-acc reconfig once per K-block in `matmul_block` | WIN in isolation, not graduated | res5 walk bench 135.2 -> 118.8 us (1.14x); sem_res2 walk flat (FPU-bound) | the res5 walk it speeds up no longer exists after `nslice_outer_ksplit`; on every remaining focus walk (bk >= 4) it is flat; it is a `matmul_block_helpers.inl` edit, outside this op |

Not taken: `split_row_gather`'s static ownership modes, cross-K-block hand-outs (+1-5%) and weights-up-front variants; `short_slice_tap_major` option A, which C supersedes; `nslice_outer_ksplit`'s unpinned bm.

### Graduated (one path each, old code deleted unless noted)
1. **Tap-aligned bk** (`select_blocking`): when `tap_major_capable(g)`, `bk_above_floor` keeps only multiples of the tap width. No carve-out.
2. **Short-slice tap-major** with the zero block (option C).
   - `tap_major_capable` admits a short slice whose lanes the reader keeps at zero: TILE by zeroing the padding tile-column slots of the `cb_input_tiles` ring once, RM by zeroing the slab once and filling real channels only.
   - Every tap-major out-of-image run reads `CB_ZERO_SRC` (32 zero pixels, one read). This replaces the 512 B `MEM_ZEROS` chunk loop on the tap-major path.
   - Carve-out (inexpressible): a TILE short slice whose real channels are not whole tiles keeps the row gather.
3. **N-slice-outer K split** (R4).
   - Under a K split, `select_blocking` solves the K-blocked walk as before, then takes N-slice-outer at that bm with the coarsest bk that fits.
   - Changes:
     - in0 is resident as `[bm, K]`;
     - the reader gathers tile-row outer, K-block inner;
     - one `matmul_block` per owner slice, Out target, no L1 acc, alternate order +1, -1, +2, -2, ..., own slice last;
     - the reader sends the +d slices on NoC0 from `CB_SLICE_R`, the writer the -d slices and its own slice on NoC1;
     - weights stream one slice per multicast round, two rounds ahead, depth 4, with the sender prefetching round r+1 under its own trid;
     - `combine_credit` / `combine_arrival` keep their meaning, and `send_go` / `send_done` are unused on this path.
   - Carve-out, written as `nslice = solve(...) or legacy`: the K-blocked walk and its post-walk scatter stay for K splits whose resident in0 does not fit L1 (aspp1/2 at bm 8).
   - The aspp.convs.3 HiFi4 regression measured in isolation does not survive the short-slice graduation: aspp.convs.3 HiFi4 goes 192.1 -> 106.9 us whole-op (guard set below).
4. **Split row gather**.
   - In both gather orders: per K-block hand-outs in K-block-outer order (only that band is known to be in), M-block-wide hand-outs in the N-slice-outer order (all bands waited first).
   - The writer plans the block's slab bands itself (shared `tile_slab_plan` / `slab_bands`) and serves hand-outs.
   - It runs one kernel per channel-slice width, because lane handling is compile-time.
   - Adds 6 semaphores; `cb_im2col_rm` holds `min(8, bm * num_k_blocks)` units on this path, counted in the L1 solve.
   - Carve-out (nothing to split): with one tile-row per K-block the plain path runs.
- Refactor that rides along: both per-tile-row gather bodies moved into `conv2d_3x3_gather_dataflow.hpp`.
  - `RowGather::tile_row` is `noinline`: inlined, its per-pixel loop measured slower.
  - `TapMajorGather::tile_row` is inline: measured 1-2 us faster than `noinline` on res5 / aspp / sem_res2.
  - The TILE slab plan is shared reader/writer (`TileSlabPlan`).

### Whole-op result (unzoned, us; ttnn.conv2d target from feature_spec)

| case | before | after | speedup | ttnn target |
|---|---|---|---|---|
| res5.1.conv2 @ LoFi | 211.1 | 45.9 | 4.60x | 75.5 |
| res5.2.conv2 @ LoFi | 210.3 | 46.1 | 4.56x | 79.8 |
| aspp.convs.1 @ LoFi | 294.0 | 103.2 | 2.85x | 159.3 |
| aspp.convs.2 @ LoFi | 291.3 | 102.5 | 2.84x | 220.5 |
| semantic res2.fuse_conv.1 @ LoFi | 189.9 | 135.2 | 1.40x | 145.6 |
| res4.0.conv2 @ LoFi | 62.5 | 44.2 | 1.41x | 39.8 |

Five of six now beat ttnn.conv2d on the same input; res4.0 is 11% behind.

### Guard set (whole perf harness, 47 shapes, zoned on both sides, same session; baseline = `acc3d2e23e2`)
- BLOCK:
  - res4.0 HiFi4 81.9 -> 63.2, res4.x HiFi4 54.8 -> 43.1, res4.x LoFi 35.9 -> 25.2;
  - res5.0 / 5.1 / 5.2 HiFi4 142.2 / 274.7 / 273.5 -> 115.5 / 116.5 / 117.2;
  - aspp.convs.1 / 2 / 3 HiFi4 411.4 / 405.9 / 192.1 -> 228.7 / 228.7 / 106.9.
- HEIGHT:
  - sem res2.fuse_conv.1 HiFi4 398.0 -> 345.1;
  - the other 13 layers move -3.8% to +0.2% (stem.conv1 25.7 -> 25.1, stem.conv3 202.3 -> 199.5, sem res2.fuse_conv.0 391.6 -> 392.5).
- Interleaved: the 18 shapes move -3.2% to +3.9%. The largest, H16_W16_Ci64 RM bf16 (10.2 -> 10.6), re-measured 3x on each side: median 10.2 vs 10.5 us, with a 0.5 us swing between baseline runs. That is inside the noise band, so not a material regression and no carve-out.
- Golden: `eval/golden_tests/conv2d_3x3/` all green — `test_op` 1584 passed (INVALID skipped), `test_op_loose` + `test_regression` 62 passed. Unit dir 198 / 198. The BLOCK unit file plus the six focus cases are clean under `--dev`: 51 / 51.

### Bottleneck now (zoned, p50 core)
- **res5.x (43 us):** compute.
  - The slice walk is 28.5 us against a ~23 us LoFi matmul floor (2016 tile-MACs).
  - Plus untilize 4, tilize 7 and combine 3.6 us.
- **aspp1/2 (101 us):** the K-blocked walk (nslice does not fit).
  - Walk 69 us, tilize 20 us of it.
  - Combine 23 us: arrivals wait 13 us on the slowest column, and the writer's send is 21 us at max.
  - Lever: make N-slice-outer fit aspp, e.g. stream in0 per K chunk with the slice loop inside.
- **sem_res2 (131 us):** compute.
  - Walk 110 us over 2 M-blocks: matmul roofline ~66 us, plus tilize 26 and the L1-acc packs.
  - Untilize 17 us.
- **res4.0 (43.5 us):**
  - untilize 9.4 us;
  - split gather ~20 us, which compute's tilize waits on;
  - slice walk 15 us: eight 72-tile-MAC matmuls paced by the weight multicast rounds; its 72-tile weight slice would fit fully resident;
  - combine 3 us.

### Helper bypasses
No kernel_lib compute helper was bypassed: the graduated compute paths use `tilize`, `matmul_block`, `reduce` and `add_bias_bcast_rows`. Two new dataflow constructs are raw NoC / CB / semaphore code because no dataflow helper covers them:

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (CB / dataflow_buffer family) | capability | A producer RISC cannot reserve CB slots on behalf of another RISC. The split row gather needs the writer to fill slots the reader reserved and pushes in order, so the slots are index-addressed (`base + (s % depth) * slot`) with a mailbox / `row_go` / `row_done` handshake. A "shared-producer reserve with per-slot completion" primitive would replace it. | n/a (no helper) | res4.0 47.4 us vs 63.3 us single-producer | conv2d_3x3_reader.cpp:452 (`top_up_writer`), conv2d_3x3_writer.cpp:519 (`serve_until`) |
| no helper covers this (cross-core partial combine family) | capability | There is no helper for a point-to-point fp32 block send with a remote arrival count (write a slice into slot `my_col` of an owner, fence, bump the owner's semaphore). It is the per-slice form of R4's scatter, which was already raw. | n/a (no helper) | res5.1 45.9 us (scatter hidden behind the slice matmuls) | conv2d_3x3_combine_dataflow.hpp:50 (`send_slice`) |

Helper-library feedback that was measured but not taken (not a bypass): `matmul_block` re-issues `pack_reconfig_data_format` and `llk_pack_reconfig_l1_acc` before every output subblock although both are constant within a K-block. Hoisting them to once per K-block measured 1.14x on a pack-bound K-blocked fp32 L1-acc walk (bk <= 3, 1x4 subblocks), bit-identical, flat when FPU-bound (`perf_experiments/l1acc_pack_hoist/`).

## Perf 2
- Date: 2026-11-18
- Scope: perf tournament round 2 on the same six `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights, pinned shard grids; bias + ReLU). No SUPPORTED change. Base `c33f4181192` (end of Perf 1). Board: Blackhole p150b, AICLK 1350 MHz. DEVICE KERNEL DURATION from one fresh-cache run per variant, unless marked as a median.
- Instrumentation: the Perf 1 zones were kept unchanged, and every new path carries zones (`writer_slab_dead_wait`, `writer_fill`, plus the existing ones).
- Ablation hooks for this tree: `perf_experiments/ablation_r2/ablation_hooks.patch`, against `c33f4181192`. Its tokens are `gather, fill, weights, scatter, untilize, tilize, matmul`; `untilize` and `tilize` keep their CB traffic.
- The NoC-estimator CLI is not built on this box, so the ceiling gate used peak arithmetic: BH NoC 64 B/cycle; LoFi matmul 16 cycles per tile-MAC.

### Measured breakdown (before, zoned, us)
Cumulative peels in both orders. Data movement first: full > gather > fill > weights > scatter > untilize > tilize > matmul (the last column is everything stubbed).

| case | full | −gather | −fill | −weights | −scatter | −untilize | −tilize | −matmul |
|---|---|---|---|---|---|---|---|---|
| res5.1 | 45.9 | 46.2 | 45.5 | 42.2 | 40.5 | 39.5 | 36.8 | 15.8 |
| aspp1 | 103.7 | 96.2 | 93.7 | 93.7 | 73.8 | 67.7 | 56.8 | 38.7 |
| sem_res2 | 137.3 | 129.0 | 121.9 | 121.6 | 121.4 | 111.4 | 95.3 | 51.7 |
| res4.0 | 44.8 | 26.9 | 26.5 | 26.0 | 23.1 | 15.2 | 14.4 | 8.6 |

Compute first: full > matmul > tilize > untilize > gather.
- res5.1: 45.9, 41.3, 42.0, 41.8, 41.1
- aspp1: 103.7, 89.3, 83.2, 83.9, 84.5
- sem_res2: 137.3, 105.5, 102.1, 96.7, 95.8
- res4.0: 44.8, 43.7, 42.8, 40.4, 17.6

Ranked, with the ceiling gate:
1. **res4.0 is gather-bound.** The stride-2 split row gather costs 17.9 us. 147 KB of local copies would take ~3 us at NoC rate, so it is issue-bound at ~67 cycles per 192 B read. The untilize comes next: 7.9 us payload, on the slow 1-tile-wide helper path.
2. **aspp: the K-blocked walk.**
   - The fp32 L1-accumulation packs make a 38.7 us floor even with the matmul stubbed.
   - The post-walk scatter tail is 19.9 us. NIU bytes allow ~7.5 us.
   - The LoFi FPU floor is 41 us; the walk takes 69.
3. **sem_res2: compute chain of 121 us** with no data movement.
   - Matmul 43.6 us over the pack floor; tilize 16; untilize 10.
   - Gather pacing the tilize: 8.3 us. Fill pacing the untilize: 7.1 us. The fill cost sits on the last-column cores, whose NoC0 reads of the next shard wrap the ring (fill 25.9 vs 13 us).
   - The weight multicast alone is ~87-96 us of DM, hidden but close behind.
4. **res5.x: co-bound.** The gather / untilize prelude, the weight rounds (~3.4 us each) and the slice walk (28.4 vs a 23.9 us floor) each remove ~0 alone. Not targeted.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Every option listed is bit-identical to the baseline output.

| idea | verdict | focus result (us, isolated) | domain |
|---|---|---|---|
| `row_gather_fixed_state`: row gather walks output-row runs; in-image window found once per run, then `set_read_state` / `read_with_state` per pixel; out-of-image runs one zero fill | WIN | res4.0 44.7 -> 31.7 | every row gather, no exception. res4.0 HiFi4 62.7 -> 50.1, res3.0 39.3 -> 34.3, HEIGHT RM C_in 48 28.7 -> 16.0, dilated/padded-lane cases -5 to -10% (the single zero fill), tap-major shapes untouched |
| `nslice_smaller_bm` (b): bm 8 N-slice-outer with `cb_combine_slots` in `cb_halo_slab`'s L1 and `cb_combined` in `cb_zero_src`'s, single-buffered slice CBs | WIN | aspp1 103.8 -> 76.9, aspp2 102.9 -> 76.7 | K splits with one M-block per core. Exception (inexpressible): more than one M-block per core. Option (a), N-slice-outer at bm 4, measured +20-22% (weights streamed twice): not taken |
| `coarse_kblock_l1`: fewer K-blocks on sem_res2 by reshaping quanta (per-tap im2col unit, weights in N-chunks) | NULL | bk 24: flat to +1.5%; side option bm 10: -3% | per-K-block cost is hidden under a matmul at ~20 cycles/tile-MAC (walk with all DM/tilize stubbed: 95.3 -> 92.4 us at bk 24). The bm 10 option needed a `matmul_block` bypass and a gated dual path for -3% on one shape: not taken |
| `deep_im2col_ring`: `cb_im2col_rm` grows to one K-block (bm tile-rows) in L1 left after the solve | WIN | sem_res2 135.3 -> 128.5 | every K-blocked walk with the non-split gather, no exception. aspp -2.9% (depth 3), res2 -8.3%, center_head1 -7.9%, interleaved 128x256x64 bf8b -6.9%, rest flat |
| `tile_fill_split_nocs`: HEIGHT TILE fill split by owner direction; writer reads higher shards' tile-rows on NoC1 and serves hand-overs while waiting | WIN with carve-out | sem_res2 137.6 -> 129.8 | HEIGHT TILE layers. Carve-out (measured): the multicast sender and own-stream cores keep the reader-only fill; splitting them was 139.2 vs 129.8 us. In isolation res3_0_conv2 read +7.2% |
| `narrow_untilize`: 1-tile-wide untilize, 4 tile-rows per DEST section, section packed as one contiguous L1 stream | WIN | res4.0 44.4 -> 42.3; untilize zone 9.5 -> 3.0 | every 1-tile-wide slab (compile-time), no exception. res4.x LoFi -7%, res4.x HiFi4 -3.8%. Wider-slice CB batching measured null |

Not floated (cap of 6):
- **K-blocked weight-multicast pipelining:** weights are hidden on every focus case (solo -0.3 us).
- **A res5 fusion of prelude + weights + walk:** co-bound, so it needs three levers at once.

### Graduated (one path each, old code deleted unless noted)
1. **Fixed-state row gather** (`4a9d314a84e`). `RowGather::tile_row` is rewritten; edge pixels, dilated rows and padded lanes keep the per-tap body inside it. No predicate.
2. **Deep im2col ring** (`df0440b790a`).
   - `_im2col_ring_target` = max(2, bm) on the K-blocked walk with the non-split gather.
   - `select_blocking` grows the ring only into L1 left at the solved (bm, bk, ub), so it never costs a coarser block.
   - Host-only.
3. **Narrow untilize** (`1e430406181`).
   - Raw LLK in `conv2d_3x3_narrow_untilize.hpp`, taken when `untilize_width_tiles == 1`.
   - Wider slabs keep `compute_kernel_lib::untilize`, which already runs `fast_untilize` there. That is the same helper the bypass replaces, not a dual path of this pattern.
   - The experiment's RowMop / Strided / diagnostic modes and wide batching were dropped.
4. **Shared-L1 N-slice walk** (`23f8e1d27bc`).
   - `select_blocking` runs N-slice-outer with the slab / combine-slot and zero-block / combined regions shared wherever every core runs one M-block. Slice CBs are depth 1.
   - The writer's credits wait for the reader's `slab_dead` (send_go's semaphore id, unused on this walk).
   - **Not a carve-out:** the experiment's "only when plain N-slice does not fit" ordering. Shared everywhere was measured on the BLOCK harness: 8 shapes flat, res4.x LoFi median of 3 23.6 vs 23.3 us (noise). So the shared layout is the one-M-block path.
   - Plain N-slice remains only for more than one M-block per core (inexpressible). The K-blocked walk remains only where neither fits.
5. **Split TILE fill** (`520d5b2a2ea`).
   - On by structure for HEIGHT TILE: sharded output, TILE input, no K split.
   - Carve-out (measured): the multicast sender and the own-stream cores keep the reader-only fill.
   - Measured in combination on the final tree (2 runs each side): sem_res2 128.3 -> 121.9; center_head1 -7.4%; inst_res3_fuse_conv0 -3.8%; sem_res2_fuse_conv0 / sem_res2_fuse_conv1 HiFi4 -2.0%; inst_res2_fuse_conv0/1 -1.7%.
   - res3_0_conv2, once the row gather was fixed: 34.45 -> 34.95 (+1.4%, noise). It gets no carve-out.

### Whole-op result (unzoned, us; same session; ttnn.conv2d target from feature_spec)

| case | before (`c33f4181192`) | after (`520d5b2a2ea`) | speedup | ttnn target |
|---|---|---|---|---|
| res5.1.conv2 @ LoFi | 45.6 | 45.8 | flat | 75.5 |
| res5.2.conv2 @ LoFi | 45.7 | 46.4 | flat (+1.5%) | 79.8 |
| aspp.convs.1 @ LoFi | 102.9 | 76.2 | 1.35x | 159.3 |
| aspp.convs.2 @ LoFi | 102.4 | 75.8 | 1.35x | 220.5 |
| semantic res2.fuse_conv.1 @ LoFi | 135.4 | 121.2 | 1.12x | 145.6 |
| res4.0.conv2 @ LoFi | 44.3 | 29.7 | 1.49x | 39.8 |

All six now beat ttnn.conv2d on the same input. res4.0 was the last one behind (44.3 vs 39.8 us).

### Guard set (whole perf harness, 47 shapes, unzoned, same session, before vs after)
- **BLOCK:**
  - res4.0 HiFi4 62.5 -> 47.0;
  - res4.x HiFi4 42.9 -> 41.1;
  - res4.x LoFi 24.5 -> 23.2;
  - aspp.convs.1 / 2 HiFi4 228.4 / 228.1 -> 201.6 / 201.7;
  - res5.0 / 5.1 / 5.2 and aspp.convs.3 flat (within ±0.4%).
- **HEIGHT:**
  - center_head1 43.7 -> 37.8;
  - res3_0 38.9 -> 34.6;
  - res2 36.6 -> 33.7;
  - sem_res2_fuse_conv0 390.0 -> 373.4;
  - sem_res2_fuse_conv1 HiFi4 343.5 -> 330.4;
  - inst fuse convs -4.5 to -4.8%;
  - stem_conv3 199.0 -> 194.8;
  - the rest flat (res3_conv2 +0.9%, stem_conv1 -0.8%).
- **Interleaved:**
  - 128x256x64 tile bf8b 50.8 -> 47.7;
  - the other 17 move -1.8% to +5.3%. The +5.3% outlier, H16_W16_Ci64 tile bf16 at ~10 us, took no graduated path; median of 3 per side is 9.84 vs 9.88 (rm 9.79 vs 9.90, bf8b 10.11 vs 9.99). Noise: no regression, no carve-out.
- **Golden:** `eval/golden_tests/conv2d_3x3/` and the unit dir: 1747 passed, 480 skipped (INVALID), 0 failed, on the final tree.
- **`--dev`:** the HEIGHT unit file plus the six focus cases are clean (47 / 47).

### Bottleneck now (zoned, p50 core)
- **res5.x (45.7 us): unchanged and co-bound.**
  - untilize 2.8, gather 7.2 (tilize 7.5 waits on it), slice walk 29 us, combine 3.6.
  - The weight rounds (27 us over 8) keep pace with the walk.
- **aspp1/2 (77 us): compute-bound.**
  - untilize 9.5, tilize 16 (paced by the 15.7 us gather), walk 44.8 (92% of the 41 us LoFi floor), combine 2.
  - Lever: start the gather/tilize before all three bands are untilized, or overlap them with the first slices.
- **sem_res2 (121.8 us):**
  - walk 103 us, including tilize 16.6; the matmul runs at ~20 cycles/tile-MAC against 16. Untilize 10.3, bias 3.
  - Lever: matmul efficiency at 1x4 subblocks.
- **res4.0 (29.2 us):**
  - tilize 12.1, paced by the split gather (reader 12.7 / writer 10.2 us); walk 9.1; untilize 2.6; combine 3.3.
  - Lever: the gather's per-unit overhead and the all-band wait.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::untilize<1, ...>` | capability | At block width 1 BH cannot take `fast_untilize` (it needs >= 2 tiles), so the helper runs one `pack_untilize_block<1,1>` per tile-row: one DEST section and one CB sync per row, and the untilize MOP closes every 32-datum row and reprograms its L1 address. There is no multi-row-per-section or contiguous-stream mode for 1-wide rows. Two traps: `pack_untilize_block` with `block_rt_dim > 1` re-reads the same input tile; and a pack that leaves PAC ch0 Y non-zero silently corrupts the next BH tilize (PCC 0.9999, only a bit-exact check sees it). | 44400 (untilize zone 9.5 us; 157 ns/tile isolated) | 42297 (3.0 us; 43.5 ns/tile) | conv2d_3x3_compute.cpp:268, conv2d_3x3_narrow_untilize.hpp:114 |
| `dataflow_kernel_lib` `ReceiverPipe::receive` (mcast) | capability | It blocks on the data-ready semaphore, with no poll / try-receive / idle-callback form, so a writer cannot serve other work (the split fill's hand-overs) while a round is in flight. The bypass re-implements the fixed-sender receive from `detail::positional_mcast_semaphore` / `mcast_wire` internals, which couples the op to the helper's private wire layout. | 138553 (split fill, blocking receive) | 129791 (polled) | conv2d_3x3_writer.cpp:314 |
| `local_copy` trid barrier (`async_read_barrier_with_trid`) | capability | There is no non-blocking "is this transaction id flushed" query, so the bias prefetch cannot serve hand-overs while its DRAM reads land. Raw `ncrisc_noc_read_with_transaction_id_flushed`. | res3_conv2 +3.8% vs base (blocking bias read) | flat vs base (polled) | conv2d_3x3_writer.cpp:356 |
| no helper covers this (CB / dataflow_buffer family) | capability | Same gap as the Perf 1 split row gather: one RISC cannot reserve CB slots on behalf of another, so the writer fills the `cb_input_tiles` slot the reader reserved, index-addressed (`base + (g % depth) * batch`), with a `fill_go` / `fill_done` handshake. | n/a (no helper) | sem_res2 121.9 vs 128.3 | conv2d_3x3_writer.cpp:243 (`fill_one`) |

Helper feedback that was measured but not taken (not a bypass):
- **`compute_kernel_lib::matmul_block`** waits for and pops one contiguous bk x bn weight block per K-block. It cannot consume a weight CB that holds fewer than two whole K-blocks (no per-N-chunk wait/pop), and that blocked the bm 10 option on sem_res2 (`perf_experiments/coarse_kblock_l1/`).
- **`dataflow_kernel_lib::set_read_state` / `read_with_state`** take the size as a template argument, so the row gather instantiates two sizes (full and kernel-row tail). This is an ergonomic note; there is no bypass.

## Perf 3
- Date: 2026-11-18
- Scope: perf tournament round 3 on the same six `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights, pinned shard grids; bias + ReLU). No SUPPORTED change. Base `0995ef092d3` (end of Perf 2). Board: Blackhole p150b, AICLK 1350 MHz. DEVICE KERNEL DURATION from one fresh-cache run per variant, unless marked as a median.
- Instrumentation: the Perf 1/2 zones are kept, and every new path carries zones (`reader_center`, plus the existing `reader_gather` / `compute_tilize` around the window and center-slot paths).
- Ablation hooks for this tree: `perf_experiments/ablation_r3/ablation_hooks.patch` (against `0995ef092d3`). Same tokens as round 2; `fill` now also stubs the writer's split-fill reads.
- Ceiling gate: the NoC-estimator CLI is still not built on this box, so peak arithmetic (LoFi 16 cycles per tile-MAC; BH NoC 64 B/cycle; 8 DRAM banks).

### Measured breakdown (before, zoned, us)
DM first: full > gather > fill > weights > scatter > untilize > tilize > matmul.

| case | full | −gather | −fill | −weights | −scatter | −untilize | −tilize | −matmul |
|---|---|---|---|---|---|---|---|---|
| res5.1 | 45.9 | 45.1 | 44.2 | 42.5 | 40.5 | 39.4 | 36.7 | 15.8 |
| aspp1 | 77.0 | 73.3 | 77.4 | 71.1 | 70.5 | 64.9 | 55.3 | 14.0 |
| sem_res2 | 121.9 | 121.7 | 121.8 | 121.5 | 121.5 | 112.0 | 93.9 | 52.4 |
| res4.0 | 29.3 | 19.7 | 19.7 | 20.4 | 16.5 | 15.1 | 14.5 | 10.0 |

Compute first: full > matmul > tilize > untilize > gather > fill > weights.
- res5.1: 45.9, 42.6, 42.2, 42.7, 41.9, 42.6, **24.6**
- aspp1: 77.0, 65.4, 66.7, 64.7, 65.6, 69.1, **19.1**
- sem_res2: 121.9, 98.2, 97.1, 94.8, 94.4, 94.2, **52.5**
- res4.0: 29.3, 28.5, 27.8, 27.5, **16.8**, 17.0, 15.7

Ranked:
1. **The weights are a hidden co-bottleneck** (new this round). With compute stubbed, res5 stays at ~42 us, aspp at ~65 us and sem_res2 at ~94 us. Only stubbing the weights drops them to 24.6 / 19.1 / 52.5. The solo removals of rounds 1-2 read them as hidden.
2. **sem_res2: compute chain of 121 us.** The walk without the tilize runs ~88 us against the 68 us LoFi floor. The tilize payload is 18.1 us and the untilize 9.5.
3. **res4.0: gather-bound.** The stride-2 per-pixel gather costs 9.6 of 29.3 us.
4. **aspp: compute ~70 us, co-bound with weights ~65 us.** Gather pacing costs 3.7 us.
5. **res5: compute ~44 us vs weights ~42 us.** Both bound it.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Every option listed is bit-identical to the baseline output.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `stride2_strided_tilize`: at stride 2 a tile-row's im2col rows are overlapping 3-pixel windows of one input run; one read per (tile-row, kernel row), then a fast tilize initialized at 2c and read 3c wide (2-pixel row stride); in-image runs tilized straight from the slab on the N-slice walk | WIN | res4.0 29.5 -> 21.54 (median of 3); 22.30 without the slab-direct raw LLK | stride 2, dilation 1, whole-tile pixel slices, one K-block per kernel row. BLOCK stride 2: res4.0 HiFi4 -17%, short-slice -60%, 2x8-grid 128x256x64 -66%; HEIGHT res3.0 flat. Inexpressible: W_out % 32 != 0 when the shared slot is not zero on both sides, dilation > 1 at stride 2, non-tile slices |
| `center_tap_direct_tiles`: with stride 1, p == d and W % 32 == 0 the kw = 1 taps are whole input tile-rows; the reader copies them from the shard and compute skips their tilize (`In0SourceFn` for one-tap K-blocks, reserved-and-pushed slots for one-row K-blocks) | WIN | sem_res2 121.9 -> 115.4 (5 runs 1.053-1.057x) | sharded TILE input. HEIGHT -1..-9%. Measured regression: interleaved TILE (-7% bf8_b, -15% bf16). Inexpressible: ROW_MAJOR, stride != 1, p != d, W % 32 != 0, the K split's K-blocked walk. L1-gated: it only takes L1 left after the solve |
| `slab_reuse_across_mblocks`: untilize each input tile-row once per core (core-wide slab) instead of per band per M-block | WIN, not graduated | sem_res2 121.9 -> 118.0 (median of 3) | needs +112 KB on sem_res2 against 129 KB of slack; center-tap needs 42.5 KB of the same slack. They cannot both fit on any shape where slab reuse applies (sem_res2 LoFi and HiFi4 only), and center-tap measured larger |
| `weight_bank_spread`: slice-contiguous weight pages so a round spreads over the 8 DRAM banks | NULL | whole op flat (aspp1 76.67 -> 76.40, res5.1 45.57 -> 45.40, median of 3) | the weight chain is bound by all 11 column senders sitting on grid row 0, not by banks. Staggered senders (`McastSenderPlacement.Staggered`, host only) halve aspp's weights-only chain (65.5 -> 38.3); the chunk layout adds 4-5 us more. Hidden behind compute today |
| `matmul_subblock_shape`: run the K-blocked walk at the LoFi floor via subblock orientation / height | NULL | sem_res2 121.7 -> 122.2 (caller-owned partials, flat) | 1x4 and 4x1 both run 19.2 cycles/tile-MAC isolated; every 4-tile subblock is within 2%; mixed heights lose (a mid-K-block init drains the pipe). Walk budget per core: 68.3 us floor, +6.4 LLK per-tile overhead, +7.2 K-blocking, +22.0 per-K-block tilize |
| `nslice_stream_first_slice`: interleave slice-0's matmul with the in0 tilize so the walk starts under the gather | NULL on aspp/res5, superseded on res4.0 | aspp1 +4.6% (4-row groups), res5 +2% | the N-slice ring is 2 units, less than one tile-row, and aspp's L1 is full, so streaming stalls the gather. It won only on res4.0 HiFi4 (-8.3%) through the split row gather's ring, which `stride2_strided_tilize` removes. Side finding: a 9-unit N-slice ring with no stream gave res5.1 -2.0%, res5.2 -2.5% (single runs) |

Not floated (cap of 6):
- **K-blocked weight prefetch:** sem_res2's sender reads, then multicasts, at ~5 us per round with no lookahead. It is hidden (94 vs 121 us), so it is an enabler only.
- **BLOCK fill split:** fill removal measured ~0 on every focus case.
- **Combine fusion:** about 2-3 us, fp32-unpack bound.

### Graduated
1. **Strided-window im2col** (`b37bd7f9b27`).
   - `strided_window_capable` / `_strided_window` in the descriptor turn the split row gather off wherever the window applies. The per-pixel row gather stays for shapes outside the domain (16-channel RM BLOCK slices, C_in 48, the inexpressible stride-2 shapes).
   - Raw LLK on the unpack thread (`window_tilize`, `window_unpack_chunk`): `ckl::tilize` ties the row stride to the block width and reads at the CB read pointer.
   - The slab-direct units read their address from a 16 B word at the end of each unit. The reader keeps the slab alive until compute has popped every unit (one `cb_reserve_back` before `slab_dead`).
   - No carve-out.
2. **Center-tap direct in0** (`47d638a7180`).
   - Mode 1 (one tap per K-block): `cb_center_tiles` through `matmul_block`'s `In0SourceFn`.
   - Mode 2 (one kernel row per K-block, bm >= 2): the reader writes the slots of a second in0 K-block; compute's tilize reserves and pushes them unpacked.
   - Fixed while merging: two `-Werror=div-by-zero` builds on W < 32 and C_in = 8 shapes. The copy lambda is compiled on every path, and the subagent's unit runs never built one of those shapes.
   - Menu-only knobs (single-buffered slots, buffer-in-solve) deleted.
3. **Carve-out (`3d8bc6218cb`, measured): no center taps under a K split.**
   - The N-slice walk was in the optimizer's domain as flat. Measured on the merged tree, median of 3, on vs off: res5.1 46.6 vs 45.9 (+1.5%) and aspp1 78.5 vs 76.7 (+2.3%), both with non-overlapping ranges; res5.2 -0.6%, aspp2 +1.0%. No N-slice case got faster.
   - Its tilize is paced by the gather, so the copy only adds reads. The N-slice center code (resident-in0 slots, the slot-free polling) is deleted rather than gated.
   - The K split's K-blocked walk was already inexpressible.

### Whole-op result (unzoned, us; two alternating base/final pairs, mean; ttnn.conv2d target from feature_spec)

| case | before (`0995ef092d3`) | after (`3d8bc6218cb`) | speedup | ttnn target |
|---|---|---|---|---|
| res5.1.conv2 @ LoFi | 46.0 | 45.8 | flat | 75.5 |
| res5.2.conv2 @ LoFi | 46.5 | 46.5 | flat | 79.8 |
| aspp.convs.1 @ LoFi | 76.3 | 76.5 | flat | 159.3 |
| aspp.convs.2 @ LoFi | 75.9 | 76.9 | flat (one 77.8 run; program path unchanged) | 220.5 |
| semantic res2.fuse_conv.1 @ LoFi | 121.2 | 114.8 | 1.056x | 145.6 |
| res4.0.conv2 @ LoFi | 29.9 | 20.9 | 1.43x | 39.8 |

### Guard set (whole perf harness, 47 shapes, zoned, same session, base `0995ef092d3` vs final)
- **BLOCK:**
  - res4.0 HiFi4 47.3 -> 39.7 (-16.1%);
  - res4.x LoFi 24.1 -> 23.6;
  - res4.x HiFi4, res5.0/1/2, aspp.convs.1/2/3 HiFi4 within ±0.5%.
- **HEIGHT:**
  - center_head1 38.4 -> 35.4 (-7.7%);
  - res2 34.0 -> 32.7;
  - inst_res3_fuse_conv0 73.7 -> 71.5;
  - inst_res2_fuse_conv1 93.2 -> 90.4;
  - res3_conv2 33.4 -> 32.5;
  - sem_res2_fuse_conv1 HiFi4 331.1 -> 324.6;
  - sem_res3_fuse_conv1 -1.2%;
  - the rest within ±0.8% (stem.conv2/3, sem_res2_fuse_conv0, sem_res3_fuse_conv0, inst_res2_fuse_conv0: no spare L1, so their program does not change).
- **Interleaved:** 18 shapes within -0.5% to +1.0%, except H16_W16_Ci64 tile_bf8b 10.3 -> 10.6 (+2.8%). That shape takes no new path, and it was the +2.7% outlier (RM) in the intermediate run of this round and in rounds 1 and 2. Noise; no carve-out.
- **Golden:** `eval/golden_tests/conv2d_3x3/` 1502 passed, 480 skipped (INVALID), 0 failed, on each graduation. Unit dir 198 / 198.
- **`--dev`:** the HEIGHT and BLOCK unit files plus the six focus cases are clean (92 / 92).

### Bottleneck now (zoned, p50 core)
- **sem_res2 (115.5 us):**
  - walk 97.0 us, tilize 11.6 of it (was 16.6); untilize 10.3; bias 3.1.
  - The weight sender's chain reaches 95.9 us (max core), just behind compute.
  - Next levers: K-blocked weight prefetch or staggered senders (enablers), and slab reuse if L1 can be found for both it and the center copy.
- **res4.0 (21.8 us):**
  - walk 9.3, tilize 4.2 (was 12.1), `reader_gather` 4.7 (was 12.6), untilize 2.6, combine 1.5 + wait 1.8.
  - New: `writer_slab_dead_wait` 2.9 us. The slab now lives until compute's last tilize, so the peers' combine credits wait for it.
- **res5.x / aspp:** unchanged. Compute and weights are co-bound.
  - The weight chain's cause is now known: every column sender sits on grid row 0. Staggered senders halve it and are host-only.
  - Next levers: staggered senders together with a compute-side cut, e.g. the deeper N-slice ring (res5 -2%, single run).

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` | capability | The row stride is tied to the block width, and the source is always the CB read pointer. A strided-window tilize needs stride 2c with width 3c, which the compute API `fast_tilize_init(2c)` + `fast_tilize_block(3c)` can do but the helper cannot express. | 29500 (per-pixel gather + helper; res4.0) | 22300 (window, compute API) | conv2d_3x3_compute.cpp:82 (`window_tilize`) |
| `fast_tilize_block` (compute API) | capability | No tilize from an arbitrary L1 address: the unpacker base is always the CB read pointer, so a run that already sits contiguous in the slab must be copied first. Raw `_llk_unpack_fast_tilize_block_` with the base taken from a per-unit address word. | 22300 | 21537 (res4.0, median of 3) | conv2d_3x3_compute.cpp:73 (`window_unpack_chunk`), conv2d_3x3_reader.cpp:522 |
| no helper covers this (CB / dataflow_buffer family) | capability | Same gap as Perf 1/2, now compute-side: one RISC cannot fill CB slots that another RISC reserves and pushes. The center-tap slots are written by the reader at index-addressed positions (`base + (k % 2) * bm * bk`) of `cb_im2col_tiles`, and compute reserves and pushes them unpacked. Correctness rests on a ring-depth invariant (reserving K-block k's first im2col row implies K-block k - 2's matmul is done), which a static_assert pins. | n/a (no helper) | res2_conv2 34.0 -> 32.7, center_head1 38.4 -> 35.4 | conv2d_3x3_reader.cpp:538 (`issue_center_rows`), conv2d_3x3_compute.cpp:129 (`tilize_rows_skip_center`) |

Helper feedback measured but not taken (not a bypass):
- **`ckl::tilize`:** no option to leave output tiles empty between blocks. The center slots split the call into InitOnly / Neither / UninitOnly one-block calls; that costs nothing measurable, but the lifecycle has to be got right by hand.
- **`compute_kernel_lib::matmul_block`:**
  - `ShortAfterPreKBlock` re-initializes every K-block even when the hook leaves the matmul state alone (raw init-once: -0.6% on the isolated walk).
  - One `MatmulBlockShape` per call, so no mixed subblock heights inside a K-block.
  - No in0 base offset. That blocked streaming slice-0 without a raw path (`perf_experiments/nslice_stream_first_slice/`).
- **`dataflow_kernel_lib` multicast (`McastFixedSenderConfig`):** the default sender placement puts every per-column sender on one grid row. Their DRAM responses share one NoC row, and `McastSenderPlacement.Staggered` halves aspp's weight chain (`perf_experiments/weight_bank_spread/`). Not taken this round: the weights are hidden.

## Perf 4
- Date: 2026-11-18
- Scope: perf tournament round 4 on the six `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights, pinned shard grids; bias + ReLU). No SUPPORTED change. Base `20cb18c0947` (end of Perf 3). Board: Blackhole p150b, AICLK 1350 MHz. DEVICE KERNEL DURATION from one fresh-cache run per variant, unless marked as a median.
- Instrumentation: the Perf 1-3 zones are kept. Every new path runs inside an existing zone: the ring fill in `reader_fill` / `reader_fill_reserve` / `reader_fill_fwd_wait` / `writer_fill`, the wide untilize in `compute_untilize`, the tap-direct gather in `reader_gather`, `direct_tilize` in `compute_tilize`.
- Ablation hooks for this tree: `perf_experiments/ablation_r4/ablation_hooks.patch` (against `20cb18c0947`). Same tokens as round 3; `gather` also stubs the strided-window gather and the center-tap copies, `tilize` also stubs `window_tilize` and the center-slot tilize.
- **Correction to my first breakdown.** The first version of the r4 patch emitted `if constexpr (false)` in both arms of the `gather` hook, so the tap-major / row / window gathers still ran under that token. On it I read the N-slice prelude as "paced by something other than payload" (gather loop 7.1 / 14.7 us with the gather "stubbed") and briefed `nslice_prelude_pacing` on that premise. Two optimizers found the bug independently; the patch is fixed and every gather-bearing column below was re-run with the fix. The prelude is the reader's own tap-block copy.

### Measured breakdown (before, zoned, us)
DM first: full > gather > fill > weights > scatter > untilize > tilize > matmul (last column: everything stubbed).

| case | full | −gather | −fill | −weights | −scatter | −untilize | −tilize | −matmul |
|---|---|---|---|---|---|---|---|---|
| res5.1 | 45.5 | 44.3 | 45.1 | 42.5 | 40.5 | 39.5 | 36.7 | 15.8 |
| aspp1 | 76.8 | 73.2 | 72.2 | 71.1 | 70.5 | 64.9 | 55.3 | 14.0 |
| sem_res2 | 115.6 | 115.4 | 115.6 | 115.3 | 115.2 | 105.3 | 93.9 | 50.2 |
| res4.0 | 22.1 | 19.6 | 19.5 | 20.2 | 16.3 | 14.6 | 14.0 | 9.4 |

Compute first: full > matmul > tilize > untilize > gather > fill > scatter (the last value is the weights-only chain).
- res5.1: 45.5, 42.4, 42.4, 43.2, 42.5, 42.9, **34.4**
- aspp1: 76.8, 65.9, 65.4, 64.8, 64.5, 65.7, **58.3**
- sem_res2: 115.6, 98.4, 97.9, 95.7, 94.4, 94.1, **94.2**
- res4.0: 22.1, 20.5, 20.3, 19.3, 17.4, 17.7, **12.8**

Blocking on the focus cases: res5 bm 7, bk 6, 3 K-blocks, N-slice, ~348 KB L1 free; aspp bm 8, bk 18, N-slice, ~1 KB free; sem_res2 bm 5, bk 8, 9 K-blocks, 2 M-blocks per core, center mode 1, ~88 KB free; res4.0 bm 8, bk 3, strided window.

Ranked, with the ceiling gate (LoFi 16 cycles per tile-MAC):
1. **N-slice prelude (res5, aspp):** untilize then gather/tilize, serial before the walk: 10.5 / 25.6 us zoned. The reader's tap-block copy paces the tilize (aspp gather solo −3.6 us). The walk itself is at 92% (aspp) and 82% (res5) of the floor.
2. **sem_res2 compute chain:** untilize 10.3 us for 2 x 21 slab rows of which 16 are shared between the two M-blocks; walk 97 us including tilize 11.6; bias 3.1. All DM stubbed = full, so DM is hidden.
3. **Weights co-bound:** weights-only chain res5 34.4, aspp 58.3, sem_res2 94.2 us.
4. **res5 walk:** 29 us vs a 23.9 us floor at 1x2 slice subblocks (out_n 2, bm 7).
5. **K-split combine tail:** res5 4.7, aspp 3.7, res4.0 5.0 us.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Every option listed meets the precision contract and, unless noted, is bit-identical to the baseline output.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `nslice_prelude_pacing`: the reader publishes each tap block's L1 source (slab run, zero block, or the slot it gathered for edge taps) in a 16 B-per-slot table; compute tilizes from it ("tap-direct") | WIN | aspp1 76.7 -> 74.7, res5.1 46.0 -> 45.0 (medians of 3); aspp2 −4.1%, res5.2 −0.6% | every tap-major N-slice prelude, tap width 1 and up. BLOCK HiFi4: aspp_convs3 −3.6%, res5_0 −1.1%, res4 LoFi −3.5%, res4 HiFi4 −2.8%. Inexpressible: row / split row gather, face-direct, strided window. Ring depth alone −2% on res5, impossible on aspp; windowed multi-trid gather: no change |
| `nslice_prelude_overlap`: gather/tilize band by band into [bm, K] slots, untilize of band kh+1 hidden under band kh | NULL | best aspp1 77.9 -> 78.0, res5.1 46.6 -> 46.5 (medians of 3); per-unit interleave +9 to +12% | the reader (fill, then gather) is the bottleneck in both phases; a 2-unit ring and 2-batch input CB cannot decouple them, and each untilize/tilize switch costs ~0.75 us |
| `wide_untilize`: half-row (5-8 tiles) and multi-row (2-4) DEST sections, each one contiguous pack stream; input synced per batch | WIN | per-core untilize (ct 8) 12.2 -> 8.7; sem_res2 115.6 -> 113.8 | every fast-untilize-capable width >= 2. Isolated ns/tile: ct 2 75.6 -> 44.9, ct 6 43.7 -> 28.6, ct 8 36.2 -> 25.9, ct 16 39.9 -> 33.2. Op: stem_conv3 −3.7%, inst_res2_fuse_conv0 −2.6%; aspp flat (fill-paced); res5 −1.1 / −1.7% slower (medians of 3, inside noise) |
| `nslice_paired_slices`: two owner slices per whole-K matmul (1x4 subblocks on res5) | NULL | res5.1 45.8 -> 45.8, res5.2 46.6 -> 45.9 (best form, medians of 3); naive pairing +6 to +16% | walk −2.2 us, but res5 is gated by the weight stream. res4 LoFi +2.6% (measured regression); aspp inexpressible (L1). Pairing + staggered senders untested |
| `ring_slab_reuse`: `cb_halo_slab` as a ring across M-blocks; each block fills and untilizes only its new rows | WIN | sem_res2 115.5 -> 112.0 (median of 3, non-overlapping) | TILE, tap-major, no K split, `c_in_tiles > 1`, where the ring fits the L1 left after the solve (sem_res2 +32 KB: the ring holds M-block 0's whole span, gaps included). stem_conv3 −2.5%, DRAM H256 W512 Ci128 bf16 −9.0% / bf8_b −5.0%; one-M-block shapes flat. Counting the ring inside the solve: +34% on a DRAM C_in 512 shape (rejected) |
| `ksplit_combine_tail`: one-pass sum + bias, own slice kept local, send-order and NoC-split variants | NULL | res5.1 45.9 (11-run median) -> 45.8 / 46.1 (`fused_x`); aspp1 −1.65% (median, in noise) | the sum runs ~47 cycles per add and is already minimal; the op end is set by the slowest core: a late walk or the remote slice from k/2 columns away (~5 us send). Send-order changes +6 to +11%; a folded-bias variant is not bit-exact (≤ 1 bf8_b step on ~0.35% of elements) |

Carried-over enabler (not a new experiment): `McastSenderPlacement.Staggered` for the per-column weight senders, round 3's `weight_bank_spread` option A (weights-only aspp chain 65.5 -> 38.3 us; whole op flat in round 3 while compute hid it). Re-measured on the round-4 tree once the prelude got shorter.

### Graduated (one path each)
1. **Ring halo slab** (`8137eab85db`).
   - `ring_capable` + `ring_plan` decide it per shape; no menu knob survives (`RING_SLAB`, `RING_SLAB_SINGLE`, `RING_IN_SOLVE`, `TEST_BM_CAP` deleted). It also runs where every core has one M-block (flat there; it drops the slab's batch rounding).
   - Carve-outs (inexpressible): K split, ROW_MAJOR, row gather, strided window, `c_in_tiles == 1` (the narrow untilize's sections would cross the ring end), and a ring that does not fit the L1 left at the solved blocking (sem_res2_fuse_conv0, DRAM C_in 512). Those keep the compacted per-M-block slab.
2. **Wide slab untilize** (`6b2a90b0515`).
   - Raw LLK in `kernels/conv2d_3x3_wide_untilize.hpp`; the experiment's diagnostic stages, the four other CB-sync modes and the folded unpack were dropped.
   - Merge fix: on the ring slab, a ct = 2 section group (2 tile-rows, one stream) could cross the ring end, so groups are capped there (`ring_rows` / `ring_pos`).
   - `compute_kernel_lib::untilize` remains only where `fast_untilize` cannot run (format / tile shape), which is the helper's own fallback, not a second path of this pattern.
3. **Tap-direct N-slice prelude** (`6f3b47c3cd3`).
   - CT args instead of the experiment's defines; the windowed / pipelined gathers and the diagnostic sources were dropped. CB 13 `cb_src_table` (64 B).
   - Not implemented on the K-blocked walk: there the gather runs under the previous K-block's matmul (sem_res2: all DM stubbed = full), so it has nothing to win.
4. **Staggered weight senders** (`3e291fb76ba`). A/B on the final tree, 3 alternating pairs, medians: aspp1 72.1 -> 71.0, aspp2 74.4 -> 69.8 (non-overlapping); res5.x, sem_res2 and the HiFi4 BLOCK shapes flat; res4.0 focus 21.3 -> 21.7 and res4 LoFi 22.1 -> 22.7 with overlapping ranges, inside the noise band, so no carve-out.

Per-graduation focus deltas (unzoned, sequential single runs): ring sem_res2 114.7 -> 111.1; wide untilize sem_res2 111.1 -> 109.9; tap-direct aspp1 75.8 -> 71.9, aspp2 75.6 -> 74.4, res5.1 46.1 -> 45.1, res5.2 47.0 -> 45.1.

### Whole-op result (unzoned, us; same session; ttnn.conv2d target from feature_spec)

| case | before (`20cb18c0947`) | after (`3e291fb76ba`) | speedup | ttnn target |
|---|---|---|---|---|
| res5.1.conv2 @ LoFi | 45.8 | 44.8 | 1.02x | 75.5 |
| res5.2.conv2 @ LoFi | 46.5 | 45.6 | 1.02x | 79.8 |
| aspp.convs.1 @ LoFi | 76.9 | 71.0 | 1.08x | 159.3 |
| aspp.convs.2 @ LoFi | 76.2 | 69.7 | 1.09x | 220.5 |
| semantic res2.fuse_conv.1 @ LoFi | 114.6 | 110.1 | 1.04x | 145.6 |
| res4.0.conv2 @ LoFi | 20.9 | 21.4 | flat (noise; only the stagger reaches it) | 39.8 |

### Guard set (whole perf harness, 47 shapes, unzoned, same session, base `20cb18c0947` vs final)
- **BLOCK:** aspp.convs.1 / 2 / 3 HiFi4 201.8 / 201.7 / 106.2 -> 196.5 / 195.6 / 102.6; res5.0 / 5.1 / 5.2 HiFi4 −1.4 / −1.4 / −2.4%; res4.x HiFi4 −2.1%, res4.x LoFi −2.7%; res4.0 HiFi4 flat.
- **HEIGHT:** stem_conv3 194.9 -> 186.7 (−4.2%); stem_conv2 122.1 -> 118.3; res2_conv2 32.4 -> 31.3; inst_res2_fuse_conv0 117.5 -> 113.7; sem_res2_fuse_conv1 (HiFi4) 323.8 -> 318.7; the other 9 within −1.0% to +0.1%.
- **Interleaved:** 18 shapes within −2.2% to +1.2% (largest: H128 W256 C_in 64 tile_bf16 63.0 -> 61.6; RM bf16 184.9 -> 187.2, a shape whose program did not change).
- **Bit-exact:** all 47 harness outputs are identical to the base commit's (`torch.equal`).
- **Golden:** `eval/golden_tests/conv2d_3x3/` 1502 passed, 480 skipped (INVALID), 0 failed, after each graduation. Unit dir 245 / 245.
- **`--dev`:** the HEIGHT and BLOCK unit files plus the six focus cases are clean (92 / 92).

### Bottleneck now (zoned, p50 core)
- **res5.x (45 us):** untilize 2.3, tilize 6.1 (paced by the 5.8 us edge-tap gather), walk 30.2 (floor 23.9), combine 1.1 + 2.5. Next: pairing slices now that the weights are staggered (walk −2.2 us measured, untested in combination).
- **aspp (70-71 us):** untilize 8-9 (paced by the 7.5-8.8 us fill), tilize 12.0 (now tilize-bound, ~465 ns per 18-tile unit), walk 44.8 (92% of floor), combine ~3.
- **sem_res2 (110 us):** walk 97.3 including tilize 11.5; untilize 5.3 (was 10.3); bias 3.1. The matmul at ~20 cycles per tile-MAC is now the bulk.
- **res4.0 (21 us):** walk 9.1, tilize 4.2, gather 4.8, untilize 2.6, combine 1.5 + wait 2.3.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::untilize<ct>` (BH `fast_untilize` path, ct >= 2) | capability | Its unit of work is one tile-row, and a DEST section holds at most 4 whole tiles of it. For ct > 4 every slab row of a chunk is its own L1 stream closed by a Last PACR (64 packer drains per 8-wide tile-row), and CB sync is per tile-row. There is no half-tile-row section (faces 0-1, then 2-3, of up to 8 tiles), no multi-row section for ct <= 4, no batch-level sync, and no way to keep a section group from crossing the end of a ring-addressed output CB. | 12156 (per-core untilize, ct 8; 36.2 ns/tile) | 8689 (25.9 ns/tile); op sem_res2 111.1 -> 109.9 us | conv2d_3x3_compute.cpp:509 (`untilize_wide_rows`), conv2d_3x3_wide_untilize.hpp:362 (`untilize_wide`) |
| `compute_kernel_lib::tilize` / `fast_tilize_block` (compute API) | capability | The unpacker base is always the CB read pointer. Tap-direct tilizes each tap block from a per-unit published L1 address (slab run, zero block, or the unit's slot), so the reader does not copy it. Same gap as Perf 3's `window_tilize`; a source-address hook on the tilize helper would cover both. | 76672 (aspp1, median of 3) | 74731 (median of 3); final tree 75.8 -> 71.9 us | conv2d_3x3_compute.cpp:157 (`direct_tilize`), conv2d_3x3_reader.cpp:747 (tap-direct loop) |
| `tilize_block` (compute API, one-tile-wide block) | capability | Same gap for tw = 1: `llk_unpack_tilize_impl` with the published address as base, then the usual datacopy and pack. | res4 LoFi 23287 (median of 3) | 22477 | conv2d_3x3_compute.cpp:143 (`direct_unpack_tilize`) |

The ring slab and the staggered senders use no raw LLK: the ring is op dataflow code (no helper covers a ring-addressed staging slab), and the stagger is the multicast helper's own `McastSenderPlacement.Staggered`.

Helper feedback measured but not taken (not a bypass):
- **`dataflow_kernel_lib` multicast:** with the round-4 compute cuts, `Staggered` placement is now a measured whole-op win (aspp.convs.2 −6.2%). The default per-column placement (all senders on one row) is the wrong default for any column-per-group weight stream.

## Perf 5
- Date: 2026-11-18
- Scope: perf tournament round 5 on the six `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights, pinned shard grids; bias + ReLU). No SUPPORTED change. Base `2ac42333eb2` (end of Perf 4). Board: Blackhole p150b, AICLK 1350 MHz. DEVICE KERNEL DURATION from one fresh-cache run per variant, unless marked as a mean or median.
- Instrumentation: the Perf 1-4 zones are kept. New zones on new paths: `reader_ring_wait_step`, `reader_ring_send` (inside `reader_combine_send`, which now spans the ring forwarding), `writer_ring_credit`, `compute_bias_wait`, `compute_combine_bias_wait`. The windowed / split fill runs inside `reader_fill*` / `writer_fill`; the ring walk inside `compute_matmul_walk` / `compute_combine`; the planned tap-direct gather inside `reader_gather`.
- Ablation hooks for this tree: `perf_experiments/ablation_r5/ablation_hooks.patch` (against `2ac42333eb2`), plus `zone_timeline.py` (one core's zone start/end times). Every token was checked solo against its own zone. My first `untilize` hook emitted `if constexpr (false) {} else` ahead of the real untilize, so the untilize ran after the stub and the program hung; fixed before any number below was taken.

### Measured breakdown (before, zoned, us)
DM first: full > gather > fill > weights > scatter > untilize > tilize > matmul (last column: everything stubbed).

| case | full | −gather | −fill | −weights | −scatter | −untilize | −tilize | −matmul |
|---|---|---|---|---|---|---|---|---|
| res5.1 | 45.3 | 43.8 | 42.8 | 42.8 | 40.9 | 38.8 | 36.4 | 16.2 |
| aspp1 | 71.1 | 71.3 | 68.0 | 67.4 | 66.8 | 63.9 | 54.0 | 13.0 |
| sem_res2 | 111.1 | 110.3 | 110.3 | 109.8 | 110.0 | 105.3 | 93.6 | 50.3 |
| res4.0 | 21.4 | 20.3 | 19.2 | 20.0 | 16.2 | 14.6 | 13.9 | 9.3 |

Compute first: full > matmul > tilize > untilize > gather > fill > scatter (the last value is the weights-only chain).
- res5.1: 45.3, 32.7, 33.4, 33.1, 32.1, 33.4, **24.9**
- aspp1: 71.1, 46.4, 43.6, 43.7, 38.6, 38.9, **33.0**
- sem_res2: 111.1, 97.2, 96.8, 95.8, 94.4, 94.1, **93.8**
- res4.0: 21.4, 20.5, 19.7, 18.2, 16.1, 16.6, **12.0**

Solo: fill aspp1 −3.1, res5.1 −1.4; gather res5.1 −1.5, res4.0 −1.1; scatter res4.0 −3.3, res5.1 −1.8; weights 0 everywhere. Tilize alone leaves res5 flat and aspp −2.7 because the walk then waits on weights (walk zone res5 30.3 -> 36.6).

Ranked:
1. **res4.0 / res5: the K-split slice sends.** Scatter solo −3.3 / −1.8 us. On res4.0 the depth-1 slice CBs back-pressure the walk (+1.35 us) and the last remote slices land inside the combine.
2. **res5 walk:** 30.3 us at 20 cycles per tile-MAC (floor 23.9): bm 7 x out_n 2 slices only take 1 x 2 subblocks.
3. **aspp prelude:** untilize 9.7 us zoned, 4.7 payload, paced by a fill of one 2-tile-row batch in flight (~22 GB/s); tilize 11.9 us, payload-bound.
4. **sem_res2 compute chain:** 110 us (all DM stubbed). Matmul ~83 us at 19.4 cycles per tile-MAC, tilize 13.8, untilize ~6, bias 2.9. The walk starts at 9.1 us (fill-paced untilize; weight round 0 behind a 3.3 us bias read). Weight rounds (5.2 us) co-pace the K-blocks (5.45 us).
5. **res5 prelude gather:** the tap-direct reader's per-unit work paces the tilize (−1.5 us solo).

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Precision is bit-identical to the baseline unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `nslice_send_decouple`: deeper slice CBs, deferred fences, send order; became a ring reduce-scatter | WIN (ring) | res4.0 21.1 -> 16.8, res5.1 45.2 -> 41.2, res5.2 45.2 -> 41.5, aspp −1.8% | every N-slice-outer K split. Slice CB depth 2 / 4: −2.5%, inside noise; farthest-first order: regression. Not bit-identical (below) |
| `nslice_paired_weight_lead`: pair owner slices into 1 x 4 and give the weights more lead | WIN (tall subblocks); pairing and lead NULL / REGRESSION | res5.1 45.4 -> 42.2, res5.2 45.4 -> 42.0 (medians) | per-slice zones showed the walk compute-bound (weight wait 0.03 us per slice). All 8 weight rounds resident: 47.1 (+3.9%); lead + pair 45.0. 4 x 1 subblocks with a 3-row last group: 1.25 / 1.33 unpacks per tile-MAC instead of 1.5. Every N-slice walk; flat at HiFi4 and out_n 1 |
| `tap_direct_gather_hide`: hide the tap-direct edge gather | WIN | res5.1 45.6 -> 44.2, res5.2 45.3 -> 44.4 (medians) | the reader's per-unit work (0.26 us vs compute's 0.16) paced it, not read latency. Per-tile-row tap plan, zero-once edges with fixed-size stateful reads, groups of 3 units (ring 6 from leftover L1). res4 LoFi −5.6%, HiFi4 −3.1%, aspp3 −1.1%; aspp1/2 flat (no spare L1). No raw LLK |
| `block_fill_pipeline`: make the BLOCK TILE fill keep the untilize fed | WIN | aspp1 71.4 -> 68.3, aspp2 −2.2%, res5 −1.6 / −2.6% | windowed fill (4 batches in flight), whole-slab input ring in `cb_im2col_tiles`' L1 where every core runs one M-block, BLOCK NoC split on column receivers, no startup lane zeroing. Measured carve-outs below. Per-core compile-time split: 76.9 (launch skew); per-batch hand-over on the 2-batch ring: 75.8 |
| `fast_tilize_chunking`: uniform chunks / no per-chunk reinit for tw 6 | chunking NULL; lean pack WIN | isolated tw 6: 27.68 -> 27.62 ns/tile (3 + 3); lean pack 25.04 (bit-exact); op aspp1 71.5 -> 70.4 | the BH fast tilize is pack-bound (27.6 ns/tile pack alone, 18.6 unpack + math); the chunk reinit hides behind it. Lean pack: 1.06-1.13x per tile for tw 2..16, every BH fast tilize in the op |
| `height_kblock_startup`: start sem_res2's walk earlier, keep weights ahead | WIN (lazy bias wait); the rest NULL / REGRESSION | semantic 111.0 -> 109.7 (median of 3) | a kernel-start bias wait held the unpacker (first untilize) behind every core's 3.6 us bias read. Weights first / column-wise stateful weight reads / sender pushes its own weights first: flat. Sender reading round r+1 under round r's multicast: +1.0 us (DRAM replies share the sender's NoC1 with the multicast). Two fill batches in flight on the ring slab: +1.8 us. HEIGHT 13 of 14 faster |

Not floated (cap of 6):
- **sem_res2 N-outer whole-K walk** (no L1-acc passes, no per-K-block re-init): in0 `[5 x 72]` is 392 KB against 55 KB of L1 left.
- **bf16 K-split partials:** a precision change.
- **sem_res2 subblock / K-block shape:** probed in rounds 1-3, and bk is L1-locked at 8.

### Graduated (one path each, old code deleted)
1. **Ring reduce-scatter** (`fcf905f031b`): replaces the N-slice walk's direct all-to-all and the combine's slot sum. `CB_SLICE_R` becomes the 16 B `cb_ring_token`; `cb_combined` is no longer allocated on this walk; `send_slice` and `nslice_reader_sends` are deleted. No carve-out: every N-slice-outer K split. The K-blocked fallback keeps its post-walk scatter (a different walk).
2. **Lazy bias wait** (`717194a0d6d`), compute only. Merge fix: the ring's bias add calls `add_bias_bcast_rows` directly, so it got its own first-use wait, and the end-of-kernel pop now waits first (a core owning no output columns never used the bias).
3. **Tall ring subblocks** (`04e4625494e`), merged into the ring walk by hand: ring row groups are the subblock height (4, last group shorter), and odd owners take the short group first so each step starts at the height the previous one ended on. Compute and reader derive the group order from the owner (`ring_group_row0`), since the two columns a partial moves between must agree. Bit-identical to the ring commit on 15 shapes.
4. **Planned tap-direct gather** (`37cb9aabc18`). Reader CT arg 69 (`tap_direct_group`) after the ring's 68.
5. **Windowed TILE fill + BLOCK split fill** (`7c04b278db9`), merged with the ring's writer by hand: the ring writer's credit poll and slab-dead wait now serve fill hand-overs. The fill's "no im2col zeroing on tap-major" supersedes tap-direct's narrower rule. Carve-outs, both measured:
   - **Weight senders keep the reader-only fill:** +9-11% with the split (their writer blocks in DRAM reads and multicast sends).
   - **Small hand-overs keep the reader-only fill** (`SPLIT_FILL_MIN_BYTES = 8192` from higher shards per M-block). Re-measured on the combined tree, threshold 0 vs 8 KB, 2 alternating runs each: res4 HiFi4 38.08 / 38.09 vs 36.81 / 36.86 (+3.5%), res4 LoFi 17.38 / 17.36 vs 16.57 / 16.49 (+5%); res5 / aspp flat.
   - Inexpressible: the whole-slab ring needs one M-block per core, no center-tap slots, no face-direct, and a slab that fits `cb_im2col_tiles`; the windowed fill is not on the multi-block ring slab.
6. **Lean fast-tilize pack** (`4d4369d8992`), raw LLK in `kernels/conv2d_3x3_lean_tilize.hpp`, every BH fast tilize in compute (`window_tilize`, `direct_tilize`, the center-slot and K-block tilizes); `ckl::tilize` stays where the fast tilize does not apply. On the combined tree, 2 runs: aspp1 68.13 -> 67.29, aspp2 68.72 -> 67.64, sem_res2 110.14 -> 109.20; bit-identical on 15 shapes.

Per-graduation focus deltas (zoned, sequential): ring res5.1 45.3 -> 41.15, res4.0 21.4 -> 16.5; bias sem_res2 111.2 -> 110.0; tall res5.1 41.2 -> 38.1; tap plan res5.1 38.1 -> 37.1; fill aspp1 70.3 -> 68.1 (3 runs); lean aspp1 68.1 -> 67.3.

### Precision
Every graduation except the ring is bit-identical to its predecessor (15 focus + BLOCK harness shapes). The ring's running sum is an fp32 packer L1 accumulation; the old combine passed each fp32 partial through SrcA/SrcB (19 bits). Against base `2ac42333eb2`, 0.83-0.90% of the outputs differ (aspp.convs.3 1.46%), and against the fp32 torch reference:
- LoFi (the focus config): mean abs error lower on all six focus cases and res4 LoFi (−0.5% to −0.7%; aspp1 0.973 -> 0.967), max error unchanged.
- HiFi4 BLOCK harness: mean abs error +0.2% to +0.3% (aspp_convs1 0.6206 -> 0.6221), max error unchanged or within 0.5.
The precision contract (fp32 DEST, fidelity, dtypes, fp32 partials) is unchanged; golden stays green.

### Whole-op result (unzoned, us; two alternating base/final pairs, mean; ttnn.conv2d target from feature_spec)

| case | before (`2ac42333eb2`) | after (`4d4369d8992`) | speedup | ttnn target |
|---|---|---|---|---|
| res5.1.conv2 @ LoFi | 44.74 | 36.10 | 1.24x | 75.5 |
| res5.2.conv2 @ LoFi | 45.19 | 36.40 | 1.24x | 79.8 |
| aspp.convs.1 @ LoFi | 70.86 | 67.05 | 1.06x | 159.3 |
| aspp.convs.2 @ LoFi | 69.83 | 67.49 | 1.03x | 220.5 |
| semantic res2.fuse_conv.1 @ LoFi | 110.17 | 107.78 | 1.02x | 145.6 |
| res4.0.conv2 @ LoFi | 20.82 | 16.19 | 1.29x | 39.8 |

### Guard set (whole perf harness, 47 shapes, unzoned, same session, two alternating pairs)
- **BLOCK:** res4 LoFi 22.34 -> 15.90 (−28.8%); res4 HiFi4 40.33 -> 36.80; res4.0 HiFi4 39.01 -> 37.05; res5.0 / 5.1 / 5.2 HiFi4 −3.9 / −4.5 / −4.4%; aspp.convs.1 / 2 / 3 HiFi4 −1.9 / −1.4 / −2.8%.
- **HEIGHT:** center_head1 34.64 -> 32.98 (−4.8%); res3_0_conv2 −3.4%; res2_conv2 −3.3%; res3_conv2 −2.2%; inst_res2_fuse_conv0 −2.0%; inst_res2_fuse_conv1 −1.8%; stem_conv3 186.6 -> 184.0; sem_res2_fuse_conv1 HiFi4 318.9 -> 316.2; the rest −1.2% to +0.2% (sem_res3_fuse_conv0 131.15 -> 131.45).
- **Interleaved:** 18 shapes within −1.2% to +0.6%, except H16_W16_Ci64 tile_bf8b 9.92 -> 10.12 (+2.1%), the same ~10 us shape flagged in rounds 1-4. Re-measured 3 alternating pairs with both TILE dtypes: base 9.80-9.96, final 9.88-10.29, ranges overlapping with single-run spikes on the final side; medians +0.1 / +0.04 us. Noise: no carve-out.
- **Golden:** `eval/golden_tests/conv2d_3x3/` 1502 passed, 480 skipped (INVALID), 0 failed, after every graduation. Unit dir 245 / 245 after every graduation.
- **`--dev`:** the BLOCK unit file after the tall merge (45 / 45), BLOCK + HEIGHT after the fill merge (86 / 86), the whole unit dir on the final tree (245 / 245).

### Bottleneck now (zoned, core (3,3))
- **res5.x (36.4 us):** untilize 3.4, tilize 4.9, walk 26.6 (17.8 cycles per tile-MAC; floor 23.9), combine 0.7. The walk is 73% of the op, at the 4 x 1 subblock unpack rate.
- **aspp (67 us):** untilize 7.3 (payload ~4.7; fill 3.6 p50), tilize 10.8 (lean pack, payload-bound), walk 44.6 (floor 41), combine 0.4.
- **sem_res2 (108 us):** per M-block untilize 6.4 / 1.3, walk 48.7 / 47.7 (tilize 1.1 per tilized K-block, matmul ~4.6 per K-block at 19.4 cycles per tile-MAC), bias 1.45. Weights co-pace the K-blocks (`writer_weight_fetch` 90.8 us p50), and the multicast sender's NoC1 is the next lever: a read-ahead under the multicast measured +1.0 us.
- **res4.0 (16.2 us):** untilize 3.1, tilize 4.1, walk 7.9, combine 0.4.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::matmul_block` | capability | It packs with L1 accumulation only on K-blocks after the first, so a whole-K call cannot accumulate its only K-block onto a partial another core forwarded into L1; and it has no hook to gate each subblock row's pack on a CB another RISC frees (the ring token). | 21116 (res4.0, all-to-all with the helper walk) | 16807 (ring, raw walk) | conv2d_3x3_compute.cpp:615 |
| `compute_kernel_lib::matmul_block` | capability | One `MatmulBlockShape` per call, and its subblock must divide the block (width first). A bm 7 x out_n 2 slice therefore gets 1 x 2 subblocks (1.5 matmul unpacks per tile-MAC on BH). There is no "4-long subblocks with a shorter last group" mode, and a ragged group costs one re-init, which a caller can only amortize by ordering groups across calls. | 45370 (res5.1, median) | 42170 (median) | conv2d_3x3_compute.cpp:615 (ring walk, `ring_group_row0`) |
| `compute_kernel_lib::tilize` -> `fast_tilize_block` -> `llk_pack_fast_tilize.h` | capability | The BH fast tilize is pack-bound, and its pack MOP's per-tile address update (`ADDDMAREG; STALLWAIT(CFG, THCON); WRCFG`) stalls on the ADDDMAREG issued just before it. No helper or API knob reaches the replay. The fix (update for the next tile after the WRCFG, no start_op, 1-tile MOP) belongs in `llk_pack_fast_tilize.h`, where every BH fast tilize would get it; the chunk reinit in `fast_tilize_block` is harmless. | 27.68 ns/tile (tw 6, isolated); aspp1 68.13 | 25.04 ns/tile; aspp1 67.29 | conv2d_3x3_lean_tilize.hpp:42, :97; conv2d_3x3_compute.cpp:94, :168, :230, :292, :598 |
| no helper covers this (NoC write family) | capability | The dataflow API has only a blocking per-transaction-id write barrier. The ring's forwarding loop must poll its own in-flight writes, the incoming arrival count and the token CB together, so it uses `noc_async_write_one_packet_with_trid` and polls `ncrisc_noc_nonposted_write_with_transaction_id_flushed`. A non-blocking "write trid flushed" query would cover it. | n/a | ring forwarding (res4.0 tail 3.6 -> 0.95 us) | conv2d_3x3_reader.cpp:1100, :1160 |
| no helper covers this (CB family) | capability | A RISC that is neither producer nor consumer of a CB cannot wait for the consumer to pop it. The writer credits column my_col - 1 only after compute popped the previous M-block's `cb_combine_slots`, so it reads the CB's acked count register directly (`get_cb_tiles_acked_ptr` + `reg_read`). | n/a | n/a | conv2d_3x3_writer.cpp:713 |

The BLOCK split fill reuses two Perf 2 bypasses at new sites: the polled multicast receive (`receive_serving`) and the polled trid flush (`ncrisc_noc_read_with_transaction_id_flushed`, now also the windowed fill's retire, conv2d_3x3_writer.cpp:407). The planned tap-direct gather and the lazy bias wait use no raw LLK (`set_read_state` / `read_with_state` from the kernel library).

Helper feedback measured but not taken (not a bypass):
- **`fast_tilize_block` chunking:** 4 + 2 vs 3 + 3 vs one reinit per call, all within 0.3% at every tw: the pack hides the reinit.
- **`dataflow_kernel_lib` multicast sender:** reading round r+1 from DRAM under round r's multicast measured +1.0 us on sem_res2, because the DRAM replies and the multicast share the sender's NoC1. A sender read-ahead option would need the reads on the other NoC.

## Perf 6
- Date: 2026-11-18
- Scope: perf tournament (run round 1 of 5) on all 22 `[perf]` LOOSE cases: every Panoptic 3x3 layer at LoFi, fp32 DEST, bf8_b activation and weights (stem.conv1 bf16 RM), bias + ReLU (aspp.convs.3 none), pinned shard grids. Goal: >= 70% compute utilization against an ideal on 110 cores, worked in priority order. No SUPPORTED change. Base `5f23b256d1a` (end of Perf 5). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION, one fresh run per variant unless noted.
- Outcome: no case reaches 70%. All 3x3 layers of one inference: 1751 -> 1541 us (-12.0%), every focus and guard cell flat or faster.
- Instrumentation: Perf 1-5 zones kept. New zones on new paths: `server_read`, `server_send`, `server_wait_ack`, `server_wait_read`, `server_flush` (weight servers), `reader_kwf_seed`, `compute_kwf_bias_block`, `compute_kwf_matmul_shift`, `compute_kwf_weight_wait`, `compute_kwf_final` (kw-fold). The K-blocked tap-direct path runs inside `reader_gather` / `compute_tilize`.
- Ablation hooks: `perf_experiments/ablation_r6/` (the round-5 patch rebased onto `5f23b256d1a`, plus hooks for the writer's windowed / BLOCK fill, the ring forwarding writes, the planned tap-direct gather and the raw ring matmul). My first DM-first peel lacked the last two hooks: with "everything" stubbed the BLOCK cases still ran 31-58 us. The table below is the re-run.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel: full > gather > fill > weights > scatter > untilize > tilize > matmul (last column: everything stubbed). Compute-first, last value: the weights-only chain.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | weights only |
|---|---|---|---|---|---|---|---|---|---|
| stem.conv2 | 68.4 | 52.3 | 50.2 | 50.2 | 50.2 | 49.6 | 33.9 | 25.9 | 25.9 |
| stem.conv3 | 85.8 | 74.0 | 72.5 | 72.5 | 72.4 | 70.0 | 57.1 | 43.2 | 42.4 |
| res3.x | 18.4 | 18.4 | 17.9 | 12.9 | 12.9 | 12.9 | 11.9 | 6.6 | 16.0 |
| center_head.1 | 26.4 | 19.2 | 17.8 | 17.8 | 17.8 | 15.3 | 11.2 | 7.1 | 8.8 |
| sem dec res3 fc0 | 59.7 | 59.8 | 59.7 | 42.8 | 42.8 | 39.5 | 34.6 | 9.0 | 50.6 |
| sem res2 fc1 | 109.3 | 108.8 | 108.6 | 105.8 | 105.8 | 102.1 | 92.2 | 48.9 | 94.0 |
| aspp.convs.1 | 67.2 | 67.2 | 64.0 | 64.0 | 63.2 | 59.9 | 51.2 | 11.9 | 34.5 |
| aspp.convs.3 | 37.6 | 37.6 | 35.9 | 36.0 | 35.2 | 32.6 | 27.8 | 10.1 | 16.7 |
| res4.x | 16.6 | 15.5 | 14.5 | 15.0 | 14.3 | 13.6 | 11.2 | 8.3 | 9.8 |

All 22 cases are in `ablation_r6/peel_results.txt`. Ranked, with the gate (LoFi 16 cycles per tile-MAC):
1. **HEIGHT im2col chain** (untilize, reader gather, tilize). It paces stem.conv2/3, center_head.1, res2 and inst dec res2 fc0. On stem.conv2 the reader's tap copy runs 14.5-19.5 us per K-block while compute's tilize waits.
2. **HEIGHT weight multicast.** The weights-only chain is res3.x 16.0 of 18.4 us, sem dec res3 fc0 50.6 of 59.7, fc1 41.5 of 49.5, inst dec res3 fc0 29.6 of 35.3. One sender reads each round on NoC1 and then multicasts it rectangle by rectangle, at 13-17 GB/s, after a 2.7-3.9 us bias read.
3. **BLOCK walk on the FPU.** Stubbing the matmul alone takes aspp.convs.1 67.2 -> 35.6 and aspp.convs.3 37.6 -> 23.2. They compute on 88 of 110 cores (res4 on 64).
4. **N-slice prelude tilize.** Payload-bound since round 5: aspp.convs.3 6.1 us for 216 tiles.
5. **HEIGHT walk skeleton.** With everything stubbed: stem.conv2 25.9, stem.conv3 43.2, sem res2 fc1 48.9 us (unpack / L1-acc packs at small N, the bias pass). Small configs' skeleton alone is at or above the 5.7 us target: res4.x 8.3, res3.x 6.6, center_head.1 7.1.
- HEIGHT at 86 / 103 cores cannot gain from more M rows: whole output tile-rows per core stay at ceil(M / 110) = 3 / 10, the same as now.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Precision is bit-identical to the base op unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `kw_fold_output_shift`: fold kw into N (K' = 3 C_in, N' = 3 C_out) on the input's own tiles; shift Q_0 / Q_2 by +-d rows at the output with fp32 packer L1 accumulation into a row-major accumulator; final pass tilizes with ReLU | WIN (with exceptions) | vs base: stem.conv2 68.4 -> 40.6, stem.conv3 85.7 -> 60.0, center_head.1 26.3 -> 16.6, inst dec res2 fc0 46.9 -> 36.3, res2 19.8 -> 17.9 | HEIGHT TILE, bias, stride 1, p == d <= 32, W % 32 == 0, whole-tile C_in, L1 permitting. Measured regressions: res3.x +4.8%, sem dec res3 fc1 +4.0%, fc0 +2.3%. Not bit-identical: PCC and mean abs error equal to the base op's. Not built for BLOCK |
| `height_weight_mcast`: one spare core per channel rectangle serves every round (DRAM reads on NoC0, chunked multicast on NoC1, per-slot flags, slot-reuse-only acks, bias completed after round 0) | WIN | res3.x 18.2 -> 15.6, sem dec res3 fc0 59.8 -> 46.2, fc1 49.6 -> 36.3, inst dec res3 fc0 35.3 -> 29.0 (medians of 3), at the weights-stubbed bound | HEIGHT-sharded K-blocked walk with spare cores. Measured regression: interleaved (+1.5% to +4.9%). In-grid split senders collapse (concurrent multicasts into one rectangle: 1.6-3.7x slower) |
| `kblock_tap_direct`: tap-direct on the K-blocked walk (one 16 B source entry per unit as a FIFO a K-block deep; d = 1 units of three adjacent taps tilized as one strided window) | WIN | stem.conv2 68.6 -> 47.6, stem.conv3 85.4 -> 69.8, center_head.1 26.3 -> 22.4, res2 19.9 -> 17.0, inst dec res2 fc0 46.6 -> 44.5 | every tap-major K-blocked gather, HEIGHT and interleaved, both layouts, all center modes; no regression |
| `block_full_grid`: BLOCK M split over every device row (7 tile-rows per row on 10 rows instead of 8 on 8); extra rows fill from their column's shards and store into the pinned output shard | WIN | aspp.convs.3 37.6 -> 34.7, aspp.convs.2 67.4 -> 62.1, aspp.convs.1 67.0 -> 61.2; res4 LoFi flat | applies everywhere, switches itself off where more rows do not lower tile-rows per row (res5). Spare-column M-groups: res5 1.75x slower (own weight streams; the multicast helper cannot form two-rectangle groups) |
| `nslice_center_taps`: kw = 1 taps of the N-slice prelude copied from the whole-slab input ring into `cb_center_tiles` (the round-3 carve-out re-floated: the prelude tilize became payload-bound) | WIN | aspp.convs.3 37.4 -> 36.2 (median of 5), aspp.convs.1 -1.8%, aspp.convs.2 -2.1%; res4 / res5 flat | tap-direct N-slice K splits, stride 1, p == d, W % 32 == 0, whole slab in the input ring. Copying from owner shards instead: +2.5% to +11.9% on aspp.convs.1 |
| `height_bias_fold`: bias seeded in DEST at K-block 0, last K-block reloads the partials and packs the output | REGRESSION | sem res2 fc1 109.1 -> 110.5, stem.conv3 85.7 -> 87.2, inst res2 fc1 36.1 -> 36.8 (best form, medians of 3) | each seed / reload forces an unpack / math re-init per subblock (~150-260 cycles per 4-tile subblock), while the separate pass streams at ~50 cycles per tile; the reload alone already costs one bias pass |

Not floated (cap of 6): splitting HEIGHT N across 110 cores (it duplicates the im2col); a bidirectional BLOCK ring for res4's ~4 us token chain (hidden under the walk in the full run).

### Graduated (one path each)
1. **K-blocked tap-direct** (`0307f3091a3`). Raw LLK `fifo_tilize`. The copying tap-major gather (`TapMajorGather`) is deleted (`d360438ce90`): `_row_split` excludes tap-major, so every tap-major unit is now tap-direct. A static_assert pins that.
2. **Weight servers** (`6eed6f00ca1`).
   - Two new kernels. Every work core of the channel becomes a pure receiver and takes the split fill.
   - Carve-out (measured): interleaved input keeps the base sender, +1.5% to +4.9% there.
   - Merge fix: a server must sit on a core that holds neither work nor an input shard. stem.conv1's input is sharded on 110 cores and its work on 108, so the optimizer's "spare" cores held the last two input shards. The last grid row's fill read from them and slowed: 22.7 -> 24.1 us, consistent over 2 A/B pairs. stem.conv2 / conv3 have the same 110-core input, so they keep the base sender.
3. **Full-grid BLOCK M split** (`1cdc5b44c1f`).
   - No raw LLK: the remote store uses a TensorAccessor over the sharded output.
   - The unit test `..._column_weight_mcast_ragged_row` moved its grid to the device's bottom rows to keep its precondition, and two extended-row tests were added.
4. **N-slice center taps** (`2643b1442d8`), merged by hand onto (1) (reader CT 74, compute CT 44). On the combined tree it gained more than in isolation: aspp.convs.1/2/3 61.3 / 61.7 / 35.0 -> 58.4 / 59.0 / 33.5 us.
5. **kw-fold + output shift** (`110e5ad71a8`, `c04b3139725`). Raw LLK in `kernels/conv2d_3x3_kw_fold.hpp`. Merge work:
   - Under kw-fold, tap-direct (1) is off, since kw-fold owns CB 13 as its accumulator-ready token.
   - The weight server reads kw-fold's (c, kw, n) tile order and its whole-K round count.
   - The bias is completed before the first pack, because the reader seeds the accumulator from it. (2)'s "after round 0" bias had put the seed at 8 us on res3.x.
   - Where kw-fold does not pay, the K-blocked tap-direct walk runs. These are not new dual paths: that walk is needed anyway for stride 2, ROW_MAJOR, interleaved, no-bias and L1-limited cases. Carve-outs, each earned by a measured regression on the combined tree and stated as its mechanism:
     - **K-blocked kw-fold:** three fp32 Q pack passes over bm + 2 tile-rows against a tilize proportional to C_in * bm. It is taken only at Ct * bm >= 0.8 * N * (bm + 2). Measured: res3.x 0.6 (+3.5%), sem dec res3 fc0 0.75 (+3.6%) and fc1 0.6 (+7.7%) lost; inst dec res2 fc0 0.83 (-2.9%), res2 fc1 1.04 (-22%) and inst dec res3 fc0 1.5 (-8.5%) won.
     - **Whole-K kw-fold:** it exposes the bias read and the whole weight stream before its first matmul (~7 us on res2 / center_head.1). It is taken only where the im2col it removes (tilized taps plus untilized slab, 25 ns per tile) is at least that. res2 has 4.4 us of im2col (16.9 -> 18.2 us with kw-fold); center_head.1 has 8.8 us (21.6 -> 16.4).
     - **Halo MACs at the user's fidelity** (found in the guard set): kw-fold recomputes about 6 * Ct * N tile-MACs of halo per M-block. It is taken only where they cost less than the removed im2col. inst dec res3 fc0 @ HiFi4 went 69.3 -> 74.3 us with kw-fold (halo 11.4 us vs im2col 7.8 us); its LoFi twin wins 24.9%.
   - The optimizer's K-blocked / whole-K threshold (128 weight tiles) was re-checked on the combined tree, with the fast weights in. Whole-K everywhere: res3.x 18.9, sem dec res3 fc0 64.1. K-blocked everywhere: stem.conv2 66.7, stem.conv3 108.3. The threshold stays.

### Precision
- (1)-(4) are bit-identical to their predecessors.
- kw-fold changes the summation order: three kernel rows accumulate in DEST, three kw partials in fp32 packer L1 accumulation, with the bias seeded in fp32. The tf32 final read gave the same outputs as a lossless fp32 read.
- Against a fp32 torch conv, PCC and mean abs error equal the base op's: res3 0.999765 / 0.2645 vs 0.999765 / 0.2646, center_head.1 0.999760 vs 0.999761, stem.conv2 0.999764 vs 0.999764.

### Whole-op result (unzoned, us; base `5f23b256d1a` mean of 3, final `c04b3139725` mean of 2, alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 37.03 | 32.86 | 1.13x | 22.7 | 43% -> 48% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 108.05 | 107.25 | 1.01x | 90.8 | 59% -> 59% |
| 3 | res4.{1..5}.conv2 | 15.91 | 14.71 | 1.08x | 5.7 | 25% -> 27% |
| 4 | res3.{1,2,3}.conv2 | 17.95 | 15.01 | 1.20x | 5.7 | 22% -> 26% |
| 5 | stem.conv2 | 68.32 | 40.15 | 1.70x | 22.7 | 23% -> 40% |
| 6 | aspp.convs.2 | 67.15 | 58.65 | 1.14x | 45.4 | 47% -> 54% |
| 7 | aspp.convs.1 | 66.62 | 58.33 | 1.14x | 45.4 | 48% -> 54% |
| 8 | res2.{0,1,2}.conv2 | 19.12 | 16.45 | 1.16x | 5.7 | 21% -> 24% |
| 9 | center_head.1 + offset_head.1 | 25.97 | 15.55 | 1.67x | 5.7 | 15% -> 26% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 35.90 | 33.64 | 1.07x | 22.7 | 44% -> 47% |
| 11 | stem.conv3 | 84.97 | 59.25 | 1.43x | 45.4 | 37% -> 54% |
| 12 | sem dec res2.fuse_conv.0 | 135.29 | 134.88 | 1.00x | 102.1 | 53% -> 53% |
| 13 | sem dec res3.fuse_conv.0 | 59.14 | 44.98 | 1.31x | 28.4 | 34% -> 44% |
| 14 | sem dec res3.fuse_conv.1 | 49.12 | 35.66 | 1.38x | 22.7 | 32% -> 45% |
| 15 | stem.conv1 | 22.84 | 22.42 | 1.02x | 1.1 | 3% -> 3% |
| 16 | inst dec res3.fuse_conv.0 | 34.87 | 26.20 | 1.33x | 14.2 | 28% -> 38% |
| 17 | inst dec res2.fuse_conv.0 | 46.42 | 34.28 | 1.35x | 28.4 | 43% -> 58% |
| 18 | res5.2.conv2 | 36.52 | 36.53 | 1.00x | 22.7 | 44% -> 43% |
| 19 | res5.1.conv2 | 36.00 | 36.02 | 1.00x | 22.7 | 44% -> 44% |
| 20 | res5.0.conv2 | 35.60 | 35.48 | 1.00x | 22.7 | 45% -> 45% |
| 21 | res3.0.conv2 | 17.78 | 17.48 | 1.02x | 5.7 | 22% -> 23% |
| 22 | res4.0.conv2 | 16.17 | 15.52 | 1.04x | 5.7 | 25% -> 26% |

All 3x3 layers of one Panoptic inference (each case x its calls): 1751 -> 1541 us (-12.0%).

### Guard set (whole perf harness, 41 shapes plus the 22 focus cases, unzoned, same session, alternating)
- **HEIGHT (HiFi4):**
  - stem.conv2 117.6 -> 82.3 (-30.0%), stem.conv3 184.0 -> 161.6 (-12.2%), center_head1 32.9 -> 28.3 (-13.8%);
  - inst res2 fuse_conv0 -7.0%, sem res3 fuse_conv0 -6.7%, sem res3 fuse_conv1 -5.8%, inst res2 fuse_conv1 -4.1%, res2 -3.8%, inst res3 fuse_conv0 -3.0% (after the halo carve-out; +7.3% before it), res3 -2.7%;
  - the rest within -0.8% to +0.1%.
- **BLOCK:**
  - aspp.convs.1 / 2 / 3 HiFi4 192.1 / 192.6 / 99.7 -> 167.6 / 168.9 / 88.0 (-12%);
  - res4 HiFi4 -10.0%, res4.0 HiFi4 -9.1%, res4 LoFi -6.4%;
  - res5.x within +-0.2%.
- **Interleaved (18 shapes):** -4.4% to +0.3%, except H16_W16_Ci64 tile_bf8b +2.0% (means; ranges 9.93-10.14 vs 10.03-10.39). Three more alternating pairs of it and its bf16 twin gave medians +1.0% / +0.3% with overlapping ranges. It is the same ~10 us shape flagged in rounds 1-5, and its rm twin is -2.6%. Noise: no carve-out.
- **Golden:** `eval/golden_tests/conv2d_3x3/` 1518 passed, 480 skipped (INVALID), 0 failed, after every graduation and on the final tree. Unit dir 266 / 266 (261 before the two new BLOCK tests).
- **`--dev`:** the BLOCK unit file after (3) and (4) is 50 / 50; the HEIGHT unit file after (5) is 41 / 41.

### Bottleneck now
- **Small configs (res2 / res3 / res4 / center_head.1, ~15-16 us vs a 5.7 us target):** the fixed cost is at or above the target with everything stubbed, so 70% needs a different structure, not a faster stage.
  - kw-fold puts the bias read and the weight latency in front of the first pack. Next lever: pack Q_1 as an overwrite and add the bias in the final pass.
  - The BLOCK ring token chain is ~4 us on res4.
- **sem res2 fc1 / sem dec res2 fc0 (priority 2 / 12, flat):** kw-fold does not fit L1, and the weight servers do not move them. Weights co-pace a walk at ~19.4 cycles per tile-MAC, with 2 M-blocks per core.
- **res5.x (flat):** 80 cores, no spare rows; spare columns cannot take a 64-channel K slice evenly.
- **kw-fold on BLOCK** (res4.x, res5, aspp): untested. The shift is linear, so each core could shift its K-slice partial before the ring. That would remove the N-slice prelude's untilize / tilize.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` / `fast_tilize_block` | capability | Same gap as Perf 3/4: the unpacker reads at the CB read pointer, with the row stride tied to the block width. The K-blocked tap-direct walk tilizes each unit from a published L1 address, and a window unit at a one-pixel row stride (3 * Ct wide, Ct stride). | 68600 (stem.conv2, copying gather + helper tilize) | 47600 | conv2d_3x3_compute.cpp:282 (`fifo_tilize`) |
| `dataflow_kernel_lib` multicast (`SenderPipe` / `ReceiverPipe`) | capability | Five things are missing: (1) a round cannot be streamed in chunks under one handshake; (2) the receiver has no per-slot data flag; (3) the receiver cannot ack only when a slot is reused; (4) the sender cannot be a core outside the receivers' kernel set; (5) the sender's reads cannot go out on the other RISC / NoC. Chunks under the per-round ack protocol measured 18190 vs 15490 ns on res3.x. | 18206 (res3.x, one in-grid sender, median of 3) | 15617 | conv2d_3x3_weight_server_writer.cpp:56, conv2d_3x3_writer.cpp:636 |
| `ckernel::matmul_block_init` / `matmul_block` (compute API) | capability | When rt > ct the unpacker steps in0 rows with the `KT_DIM` GPR, which only the init sets. Reading tap 1 from `cb_center_tiles` (row stride ct) between in0 rows (stride 6 ct) needs a per-call stride; the raw fix is `TT_SETDMAREG` of `KT_DIM` per CB switch. | 37389 (aspp.convs.3, median of 5) | 36165 | conv2d_3x3_compute.cpp:242 |
| `compute_kernel_lib::matmul_block` | capability | It takes in0 as one contiguous [M, K] block. kw-fold's K' row of a tile-row comes from three slab tile-rows (whole input tile-rows apart), and out-of-image kernel rows are skipped. | 47650 (stem.conv2, tap-direct walk) | 40620 | conv2d_3x3_kw_fold.hpp:182 |
| `pack_untilize_dest` (compute API) | capability | No destination row offset, and the untilize MOP closes every row. kw-fold packs each Q_kw subblock into the fp32 accumulator at a computed +-d row offset with L1 accumulation, as one packer stream per subblock. | stem.conv2: row-by-row packs 32.45 us bench | 29.62 us bench (stream per subblock) | conv2d_3x3_kw_fold.hpp:206, :245 |
| `compute_kernel_lib::tilize` | capability | It cannot tilize fp32 rows from a computed address. kw-fold's final pass reads the accumulator rows of each output tile-row. | n/a (no helper form) | in the 40620 above | conv2d_3x3_kw_fold.hpp:375 (final pass) |

The full-grid split and the weight-server placement use no raw LLK.

Helper feedback measured but not taken (not a bypass):
- **`dataflow_kernel_lib` multicast:** several senders multicasting concurrently into one rectangle measured 1.6x to 3.7x slower than one sender, and chaining them was still slower. A multicast helper that offers several senders should serialize them per rectangle.
- **`Mcast` grouping:** groups are equal consecutive chunks of one ordered core set, so a slice whose participants are two rectangles (its column plus a row of spare columns) cannot be one group. This blocked the spare-column BLOCK split.

## Perf 7
- Date: 2026-11-18
- Scope: perf tournament (run round 2 of 5) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores, worked in priority order. No SUPPORTED change. Base `e40c435b625` (end of Perf 6). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION, one fresh run per variant unless noted.
- Outcome: 3 of 6 ideas graduated, 2 null, 1 regression. No case reaches 70%: aspp.convs.3 (priority 1) goes 48% -> 60%, stem.conv3 reaches 65%. All 3x3 layers of one inference: 1538 -> 1456 us (-5.3%). No focus or guard cell regressed beyond noise.
- Instrumentation: Perf 1-6 zones kept. kw-fold's `reader_kwf_seed` and `compute_kwf_bias_block` are gone with the code they timed. New zone: `reader_kwf_bias_rows` (the reader's bias decode, wait included). The kernel-row skip and the column rounds run inside `compute_matmul_walk` / `compute_kwf_matmul_shift`.
- Ablation hooks: `perf_experiments/ablation_r7/` (the round-6 patch rebased onto `e40c435b625`, plus hooks for the weight servers, the K-blocked tap-direct FIFO, `fifo_tilize`, kw-fold's matmul, and a `direct_tilize` stub that skips the N-slice center tap). Also `timeline.py` (one core's zones over time) and `endgrid.py` (every core's kernel end).

### Measured breakdown (before, zoned, us)
DM-first cumulative peel: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only": the matmul alone stubbed.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only |
|---|---|---|---|---|---|---|---|---|---|
| aspp.convs.3 | 33.6 | 33.3 | 31.5 | 31.6 | 30.9 | 29.4 | 26.9 | 11.4 | 22.3 |
| sem res2 fc1 | 107.7 | 107.8 | 105.8 | 105.7 | 105.7 | 101.6 | 93.4 | 53.5 | 71.6 |
| res4.x | 16.2 | 13.9 | 13.3 | 13.6 | 12.5 | 12.7 | 11.1 | 8.3 | 15.6 |
| res3.x | 15.6 | 15.2 | 13.6 | 12.9 | 12.9 | 11.7 | 10.5 | 5.9 | 13.1 |
| stem.conv2 | 40.7 | 40.7 | 39.8 | 35.3 | 35.3 | 35.3 | 35.3 | 33.9 | 34.8 |
| res2 | 16.7 | 16.1 | 14.3 | 14.2 | 14.2 | 14.2 | 11.0 | 8.2 | 15.2 |
| center_head.1 | 16.2 | 16.5 | 16.5 | 11.6 | 11.6 | 11.6 | 11.6 | 9.4 | 11.2 |
| inst res2 fc1 | 33.9 | 33.6 | 33.2 | 33.1 | 33.1 | 33.0 | 33.2 | 33.1 | 33.1 |

All 22 cases are in `ablation_r7/peel_results.txt`. kw-fold cases (stem.conv2/3, center_head.1, inst res2 fc1, inst dec res2/res3 fc0) keep their pack stream, final pass and seed under every hook.

Ranked:
1. **The K-blocked kw-fold walk is pack-bound.**
   - inst res2 fc1 is 33.1 us with the matmul stubbed, and its packer zone is 25.5 us: 432 fp32 L1-acc untilize tile packs at ~80 cycles per tile, 3 passes x 3N columns x (bm + 2) rows.
   - Same structure on inst dec res2 fc0 (34.5) and inst dec res3 fc0 (26.5).
2. **aspp.convs.3 multiplies a kernel row of zeros.**
   - At d = 18 on a 64-tile-row image, R = 36 tile-rows, so no output pixel has more than 2 in-image kernel rows. The ring walk still multiplies all 3: 21.4 us of 1512 tile-MACs per core, at ~19 cycles per MAC.
   - The prelude (fill-paced untilize 5.2 us, tilize 4.4 us) runs before the walk.
3. **kw-fold start-up.**
   - The writer's blocking bias read (~3 us: ~100 cores on the same DRAM pages) gates the weight stream, and compute packs a bias block before its first matmul.
   - Whole-K waits for all weights: center_head.1's compute sits until 8.3 of 16.2 us, stem.conv2's until 9.9 of 40.7.
4. **HEIGHT K-blocked weight-slot reuse.** The server waits on acks before round 2: 3.5-8.7 us on res2, and round 2 lands at 12.0 us on res3.x.
5. **res4.x prelude.** The 1-wide tap tilize zone is 4.7 us, and the gather runs alongside it.
6. **sem res2 fc1 walk.** 22.4 cycles per tile-MAC. 43 us of the K-blocked skeleton remains with everything stubbed.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`).

| idea | verdict | focus result (us) | domain / notes |
|---|---|---|---|
| `nslice_kernel_row_skip`: the BLOCK N-slice ring walk multiplies, per ring row group, only the kernel rows in-image for some row of the group (host-sent 3-bit masks) | WIN | aspp.convs.3 33.73 -> 27.30 (median of 3), bit-identical | Every N-slice ring walk, both loops. Flat elsewhere: aspp.convs.1/2, res5.x and res4 keep all 3 kernel rows on their middle grid rows, so the slowest core does not move. A per-kernel-row mask loop measured +2.7% on a 4 + 1-row-group check shape, so the contiguous-range loop graduated instead. Skipping the zero units' prelude tilize was flat: the reader's gather paces that prelude. |
| `kw_fold_column_rounds`: kw-fold streams weights per output column kw (3 Ct x N per round) instead of per kernel row; each Q tile-row accumulates every in-image kh in DEST and is packed once | WIN | inst res2 fc1 33.82 -> 28.09, bit-identical | Every kw-fold case: stem.conv2 -10.9%, stem.conv3 -13.8%, center_head.1 -26%, inst dec res2 fc0 -3.4%, inst dec res3 fc0 flat. The faster kw-fold supports one gate, halo_ns <= 0.55 x im2col_ns: wins measured up to 0.52, losses from 0.57. That gate adds res2 (16.68 -> 13.97) and res3.x (flat). Measured exceptions, left to the gate: sem dec res3 fc0 (LoFi +3.4%) and fc1 (+6.1%), and the HiFi4 res3-scale layers (+5.7% to +16%). 2x2 subblocks: much slower. |
| `kw_fold_seedless`: kw-fold's Q_1 packed as an overwrite before Q_0 / Q_2 accumulate; no reader seed, no bias block or bias wait in front of the weights; bias added in the final pass (the tilize MOP's srcB op loads 8 rows of fp32 bias the reader decodes) | WIN | stem.conv2 40.62 -> 37.62, stem.conv3 60.00 -> 53.34 (medians of 3) | Every kw-fold case: the inst cases -2.5% to -3.1%, center_head.1 flat. Precision equal to the base op, not bit-identical: tf32(Q) + bias instead of tf32(bias + Q), with 99.6-99.98% of outputs identical and the rest one bf8_b step apart. Writer-side decode and compute-built bias rows were measured as options and lost. |
| `height_weight_resident`: HEIGHT K-blocked weights fully resident (no slot reuse, no ack waits) | NULL | res3.x 15.36 -> 15.62 (median of 3) | The slowest cores never wait on a reused slot. Their bound is the fill-paced untilize and the round-0 latency, which depth cannot change. Taking depth before the leftover-L1 features measured +8.8% on sem res2 fc1. |
| `nslice_narrow_tilize`: batch the 1-wide tap tilize of res4 (4 tiles per DEST section) | NULL | res4.x 15.85 -> 15.75 (2 runs each), bit-identical | Split zones show the reader paces the prelude. `TapDirectPlanned::unit` costs ~200 cycles per unit, and the tilize waits 2.7 of its 5.3 us. Its own work is ~2.3 us for 42 tiles. |
| `height_nslice_outer`: HEIGHT walk as N-slice-outer (in0 [bm, K] resident, whole-K matmul per N-slice, one pack per output tile) | REGRESSION | sem res2 fc1: bench +5.2% to +8.9% at the only fitting slice width (s = 1) | s = 1 runs bm 5's row 4 as 1x1 subblocks (~33 cycles per MAC). s >= 2 measured -4% to -6% but does not fit L1. Shrinking bm to fit measured +13.2% whole-op. sem dec res2 fc0: inexpressible (in0 alone 430 KB). |

Not floated (cap of 6): the reader's per-unit tap-direct planning cost (res4's prelude bound); round-0 weight latency (~4.3 us on every server-fed HEIGHT case); a non-uniform BLOCK M split by real kernel rows. On that last one, the row-skip optimizer counted the possible cut of the busiest grid row: aspp.convs.2 -31%, res5.2 -24%, aspp.convs.1 -20%, aspp.convs.3 -14%.

### Graduated (one path each)
1. **Kernel-row skip** (`e6f05eb034b`).
   - It is the only N-slice ring walk; the loops it replaced are deleted. No carve-out.
   - Raw LLK on the existing raw ring walk.
2. **Column rounds** (`6f76657e49c`).
   - Rounds of the K-blocked and whole-K forms; the kh-round loop is deleted.
   - The two old gates (K-blocked `Ct*bm >= 0.8*N*(bm+2)`, whole-K im2col >= 7 us) became the single measured halo-share rule above. The gate is the only carve-out, and it is earned by the measured losses in the table.
3. **Seedless start** (`2b848923eac`), merged by hand onto (2).
   - Column rounds now stream kw = 1, 0, 2: Q_1's round is the overwrite. The writer, the weight server and the compute kernel share that order.
   - **Merge fix (measured on the combined tree).** The optimizer's blocking bias read after the weights cost the short whole-K server cases: the bias landed ~2.8 us late, then the reader decoded it at ~1.3 us per tile in front of the final pass (res2 13.9 -> 14.8, center_head.1 12.1 -> 12.6 us).
     - Issuing the bias at kernel start fixed those two but slowed round 0 on the K-blocked cases by ~1 us.
     - Issuing it once round 0 landed and pushing it when it lands, polled inside the round waits, wins everywhere.
     - The bf8_b decode's `__builtin_clz` became a <= 6-step normalize.
     - Cores without servers (stem.conv2 / conv3) keep the optimizer's read after the weights.
   - Deleted: the reader seed, the token CB, the compute bias block and the writer's bias wait.
   - Added `tests/.../test_conv2d_3x3_kw_fold.py`: 10 forced-kw-fold shapes, because no other unit shape takes kw-fold.

Combined-tree steps (zoned, us; full focus set after each step in `ablation_r7/peel_results.txt`):

| case | base | +row skip | +column rounds | +seedless (as built) | + bias at kernel start | + after round 0, blocking finish | final |
|---|---|---|---|---|---|---|---|
| stem.conv2 | 40.7 | 40.7 | 36.2 | 34.2 | 34.2 | 34.2 | 34.2 |
| res2 | 16.7 | 16.7 | 13.9 | 14.8 | 13.9 | 13.9 | 14.0 |
| res3.x | 15.6 | 15.7 | 15.3 | 15.5 | 15.7 | 17.6 | 15.2 |
| inst res2 fc1 | 33.9 | 33.9 | 28.2 | 27.3 | 28.0 | 27.2 | 27.2 |
| center_head.1 | 16.2 | 16.5 | 12.1 | 12.6 | 11.5 | 11.7 | 11.9 |
| aspp.convs.3 | 33.6 | 27.5 | 27.1 | 27.5 | 27.2 | 27.2 | 27.3 |

### Precision
- Row skip and column rounds are bit-identical to their predecessors. Column rounds sum kh in fp32 DEST instead of the fp32 packer; outputs on the 6 kw-fold cases were identical.
- Moving res2 and res3.x onto kw-fold changes 0.061% / 0.039% of outputs by one bf8_b step. PCC 0.999804 unchanged.
- Seedless: PCC and mean abs error vs a fp32 torch conv equal the base op's on every kw-fold case. For example stem.conv2 is 0.999922 / 0.35011 vs 0.999922 / 0.35012.

### Whole-op result (unzoned, us; base `e40c435b625` and final `2b848923eac`, mean of 2 each, alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 32.84 | 26.66 | 1.23x | 22.7 | 48% -> 60% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 106.54 | 106.45 | 1.00x | 90.8 | 60% -> 60% |
| 3 | res4.{1..5}.conv2 | 14.75 | 14.72 | 1.00x | 5.7 | 27% -> 27% |
| 4 | res3.{1,2,3}.conv2 | 14.95 | 15.07 | 0.99x | 5.7 | 27% -> 26% |
| 5 | stem.conv2 | 40.13 | 33.83 | 1.19x | 22.7 | 40% -> 47% |
| 6 | aspp.convs.2 | 58.64 | 58.97 | 0.99x | 45.4 | 54% -> 54% |
| 7 | aspp.convs.1 | 57.94 | 57.94 | 1.00x | 45.4 | 55% -> 55% |
| 8 | res2.{0,1,2}.conv2 | 16.42 | 13.30 | 1.23x | 5.7 | 24% -> 30% |
| 9 | center_head.1 + offset_head.1 | 15.62 | 11.49 | 1.36x | 5.7 | 25% -> 35% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 33.84 | 26.88 | 1.26x | 22.7 | 47% -> 59% |
| 11 | stem.conv3 | 58.99 | 48.61 | 1.21x | 45.4 | 54% -> 65% |
| 12 | sem dec res2.fuse_conv.0 | 134.87 | 134.87 | 1.00x | 102.1 | 53% -> 53% |
| 13 | sem dec res3.fuse_conv.0 | 44.76 | 44.57 | 1.00x | 28.4 | 44% -> 45% |
| 14 | sem dec res3.fuse_conv.1 | 35.79 | 36.25 | 0.99x | 22.7 | 44% -> 44% |
| 15 | stem.conv1 | 22.52 | 22.77 | 0.99x | 1.1 | 3% -> 3% |
| 16 | inst dec res3.fuse_conv.0 | 26.11 | 25.13 | 1.04x | 14.2 | 38% -> 40% |
| 17 | inst dec res2.fuse_conv.0 | 34.36 | 32.04 | 1.07x | 28.4 | 58% -> 62% |
| 18 | res5.2.conv2 | 36.53 | 36.60 | 1.00x | 22.7 | 43% -> 43% |
| 19 | res5.1.conv2 | 36.08 | 36.03 | 1.00x | 22.7 | 44% -> 44% |
| 20 | res5.0.conv2 | 35.50 | 35.27 | 1.01x | 22.7 | 45% -> 45% |
| 21 | res3.0.conv2 | 17.56 | 17.46 | 1.01x | 5.7 | 23% -> 23% |
| 22 | res4.0.conv2 | 15.41 | 15.41 | 1.00x | 5.7 | 26% -> 26% |

All 3x3 layers of one Panoptic inference (each case x its calls): 1538 -> 1456 us (-5.3%).

### Guard set (whole perf harness, 41 shapes plus the 22 focus cases, unzoned, same session, alternating; `ablation_r7/ab_results.txt`)
- **HEIGHT (HiFi4):** center_head1 28.20 -> 24.45 (-13.3%), res2 29.14 -> 24.91 (-14.5%), stem_conv3 -6.3%, stem_conv2 -6.1%, inst_res2_fuse_conv1 -2.5%, inst_res2_fuse_conv0 -2.2%; the other 8 within -0.5% to +0.1%.
- **BLOCK:** aspp_convs3 HiFi4 87.96 -> 63.16 (-28.2%); all others within -1.1% to +0.1%.
- **Interleaved (18 shapes):** -1.4% to +1.1%. The interleaved path is unchanged.
- **Focus cells above +0.5%:**

  | case | before (two runs) | after (two runs) | change | path |
  |---|---|---|---|---|
  | sem dec res3 fc1 | 35.67, 35.91 | 35.85, 36.65 | +1.3% | unchanged |
  | stem.conv1 | 22.30, 22.73 | 22.83, 22.70 | +1.1% | unchanged |
  | res3.x | 15.00, 14.89 | 15.06, 15.09 | +0.8% | now kw-fold; the optimizer measured it flat, median of 3 |
  | aspp.convs.2 | 58.82, 58.46 | 59.25, 58.69 | +0.6% | row skip, nothing to skip on its slowest row |

  All are inside the noise band. Flat is not a carve-out, so no carve-out was taken.
- **Golden:** `eval/golden_tests/conv2d_3x3/` 1518 passed, 480 skipped (INVALID), 0 failed after every graduation and on the final tree.
- **Unit tests:** the unit dir passed 266/266 before the new file and 276/276 with it. The forced-kw-fold file is 10/10, also under `--dev`.

### Bottleneck now
- **Small configs (res3.x / res4.x / res2 / center_head.1 / res3.0 / res4.0, 11.5-17.5 us vs 5.7).**
  - **Round-0 weight latency.** Round 0 lands at ~4.3 us on every server-fed HEIGHT layer: center_head.1, res2 and res3.x all wait for it.
  - **res3.x's fill-paced untilize** (5 us).
  - **res4's prelude**, paced by the reader's per-unit tap-direct planning (~200 cycles per unit).
  - The 5.7 us target stays below these fixed costs.
- **aspp.convs.1/2, res5.x (flat):** their middle grid rows keep all 3 kernel rows. A non-uniform M split by real kernel rows would cut the busiest row by up to 31% (aspp.convs.2).
- **sem res2 fc1 / sem dec res2 fc0 (priority 2 / 12, flat):** these are the K-blocked walk at ~22 cycles per MAC. kw-fold does not fit L1. N-slice-outer only fits at s = 1, which loses.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::matmul_block` | capability | No per-subblock K window. The ring walk must multiply only a row group's in-image kernel rows, a contiguous K range that differs per group; the helper multiplies the whole K of every subblock. | n/a (the ring walk had no helper form; raw full-K walk 33729, aspp.convs.3) | 27297 | conv2d_3x3_compute.cpp:901, :953 (ring walk); descriptor `_nslice_kh_masks` |
| `compute_kernel_lib::matmul_block` | capability | A column round's K loop runs over three non-contiguous in0 bases (one per kernel row, whole input rows apart) with out-of-image rows skipped, all accumulated in DEST, then packed at a computed ACC row offset. The helper takes one contiguous in0 block and packs at the CB write pointer. | 34710 (inst res2 fc1 on the tap-direct helper walk) | 28090 | conv2d_3x3_kw_fold.hpp:257 (`subblock_body`), :325 (column-round loop) |
| `compute_kernel_lib::tilize` / `binary_dest_reuse_tiles` | capability | No tilize with a second operand. An add after the tilize via dest-reuse needs an unpacker re-init per DEST batch plus a second unpack and FPU pass per tile. The fp32-DEST tilize copy is already an ELWADD with a zeroed srcB, so the raw MOP loads 8 fp32 bias rows into srcB instead. | not built (estimated several us more on stem.conv2, unmeasured) | 37620 (stem.conv2; final pass 3.0 us with and without the add) | conv2d_3x3_kw_fold.hpp:134 (`tilize_bias_mop_config`), :146, :386 |
| no helper (dataflow format conversion) | capability | Nothing converts a bf8_b / bf16 tile's row 0 to fp32 row-major on a data-movement RISC. The reader decodes it in plain RISC code. | n/a | ~1.3 us per tile with `__builtin_clz` (res3.x, N 4: 5.4 us); normalize loop not timed separately, hidden on every focus case | conv2d_3x3_reader.cpp:810 |

The row skip's host masks and the weight-round order use no raw LLK beyond the walks above.

## Perf 8
- Date: 2026-11-18
- Scope: perf tournament (run round 3 of 5) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores, worked in priority order. No SUPPORTED change. Base `3ca4eb28f46` (end of Perf 7). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 of 6 ideas graduated, all bit-identical. Two focus cases reach 70%: sem res2.fuse_conv.1 (priority 2) 60% -> 74%, sem dec res2.fuse_conv.0 (priority 12) 53% -> 75%. aspp.convs.3 (priority 1) 60% -> 65%. All 3x3 layers of one inference: 1456 -> 1248 us (-14.3%). No focus or guard cell regressed beyond noise.
- Instrumentation: Perf 1-7 zones kept. New zones: `server_send_bias` (server bias multicast), `compute_kwf_halo_wait`, `writer_khx_send`, `writer_khx_drain` (halo exchange), `compute_tilize_wait` and `reader_slab_dead_wait` (padded slab). The n-group passes, narrow planes and balanced split run inside existing zones.
- Ablation hooks and merge gates: `perf_experiments/ablation_r8/` (README). The roofline gate used the measured rates against the LoFi FPU floor (16 cycles per tile-MAC) and the NoC link rate, not the `noc_estimate` CLI.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only": the matmul alone stubbed; "weights only": everything but the weight stream stubbed.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only | weights only |
|---|---|---|---|---|---|---|---|---|---|---|
| aspp.convs.3 | 27.5 | 27.0 | 25.2 | 25.3 | 24.5 | 22.9 | 20.5 | 11.5 | 21.8 | 17.9 |
| sem res2 fc1 | 107.7 | 107.8 | 105.8 | 105.7 | 105.7 | 101.6 | 93.5 | 53.6 | 74.2 | 65.4 |
| res4.x | 16.1 | 14.2 | 13.4 | 13.4 | 12.4 | 12.4 | 10.9 | 8.5 | 15.6 | 10.1 |
| res3.x | 15.0 | 14.9 | 14.7 | 11.3 | 11.3 | 11.3 | 11.4 | 11.1 | 14.6 | 13.9 |
| stem.conv2 | 34.2 | 34.2 | 33.1 | 32.2 | 32.2 | 32.2 | 32.2 | 28.5 | 28.5 | 28.5 |
| aspp.convs.2 | 59.4 | 58.9 | 56.0 | 54.7 | 54.1 | 52.4 | 47.2 | 13.0 | 34.5 | 35.2 |
| res2 | 14.0 | 14.0 | 12.8 | 11.7 | 11.7 | 11.7 | 11.7 | 9.3 | 11.5 | 10.5 |
| center_head.1 | 12.1 | 12.2 | 11.5 | 10.0 | 10.0 | 10.1 | 10.0 | 6.9 | 9.6 | 9.4 |
| sem dec res2 fc0 | 135.8 | 136.3 | 133.2 | 133.1 | 133.2 | 120.4 | 104.1 | 57.3 | 92.3 | 76.5 |
| sem dec res3 fc0 | 45.0 | 45.4 | 43.4 | 42.7 | 42.7 | 39.7 | 33.6 | 8.8 | 24.4 | 24.3 |

All 22 cases are in `ablation_r8/peel_results.txt`. kw-fold cases keep their pack stream and final pass under every hook.

Ranked:
1. **sem res2 fc1 / sem dec res2 fc0 run the K-blocked tap-direct walk** at ~22.5 cycles per tile-MAC, with six per-K-block tilizes (~1 us each) per M-block and a fill-paced first untilize (0.5-7.3 us). Their everything-stubbed skeleton is 53.6 / 57.3 us of per-K-block fp32 L1-acc partial packs. kw-fold would pay (halo / im2col 0.43 / 0.34) but missed L1: 1104 vs 1027 KB and 1285 vs 1005 KB (two slabs plus double-buffered whole-N column rounds).
2. **BLOCK grid rows are unbalanced after the kernel-row skip.** `compute_matmul_walk` p50 / max: aspp.convs.3 13.7 / 16.2, aspp.convs.2 28.5 / 39.8, aspp.convs.1 37.3 / 39.5, res5.2 24.2 / 26.8 us. The middle grid rows keep all three kernel rows.
3. **HEIGHT round-0 weight latency.** Round 0 lands at ~4 us on every server-fed layer: a 1 us kernel-start ack wait, reads with 2 chunks in flight (9-22 GB/s), and ~0.3-0.4 us per 8.7 KB chunk multicast. res3.x's compute waited 6.6 of its 12.9 us.
4. **The N-slice prelude is gather-paced.** res4.x: the reader's gather (4.3 us, per-unit planning plus edge-tap copies) paces a 4.6 us tilize whose own work is ~2.3 us. aspp.convs.3: gather 3.9, tilize 4.4 us.
5. **kw-fold recomputes a halo tile-row per side**: +22% MACs at bm 3. The gate keeps sem dec res3 fc0 / fc1 off kw-fold (0.57 / 0.73).
6. **kw-fold at N = 2** (stem.conv2): 28.5 us with the matmul stubbed, 2x2 subblocks, a 2-tile-wide fp32 stream at ~116-140 cycles per tile. Its optimizer then showed the op is matmul-unpack bound (stubbing the pack moved it only 34.3 -> 34.0 us).

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). All six are bit-identical to the base op.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `kwf_ngroup_outer`: kw-fold walks one output-column group (sbw tiles) per pass, one fp32 accumulator plane, weight sub-rounds of [3 C_in, sbw] tiles, at its own bm (largest divisor whose CBs fit) | WIN | sem res2 fc1 107.6 -> 89.7, sem dec res2 fc0 135.9 -> 100.3 | Every kw-fold case. Existing ones have N <= 4 (one group): flat. bm 5 x 2 blocks measured 96.1, all planes resident 89.4 at 215 KB more L1. |
| `block_balanced_msplit`: per-grid-row M ranges from a fitted cost model (untilize per slab tile, tilize per unit, walk per k-tile of each ring row group) | WIN | aspp.convs.3 27.1 -> 26.1 (median of 3) | aspp.convs.2 -14.0%, aspp.convs.1 -8.6%, res5.1 -4.5%, res5.2 -3.5%, res5.0 / res4 flat; HiFi4 BLOCK -3% to -13%. Taken where the model drops the costliest row >= 2%; runtime row sizing on a uniform split measured +5.0% on aspp.convs.3. Found: aspp.convs.3 is now paced by the weight multicast (rounds 3+ leave the sender after its `slab_dead_wait`). |
| `server_stream_latency`: one linked multicast chain per run of landed chunks, 3 chunks per round, early slot-reuse acks, start ack first, the server multicasts the bias | WIN | res3.x 14.95 -> 12.12 (median of 3) | Every server-fed layer: res2 -10.8%, center_head.1 -6.5%, inst res2 fc1 -2.5%, the rest flat. The linked chain alone: 14.45; the bias multicast alone: flat; together they unblock each other. Server reads are issue-bound (37-50 cycles per one-tile read). |
| `tapdirect_cheap_publish`: the N-slice slab untilized into zero-gapped row slots (W + d pixels), so every tap is one slab run or the zero block; compute computes its tilize sources | WIN | res4.x 15.90 -> 13.43 (median of 3) | Every N-slice center-tap prelude (stride 1, p == d, W % 32 == 0, one M-block) where the padded slab fits leftover L1: res5.x -2.7% to -5.1%, aspp flat (its 6-wide tilize already ran at its own rate). Of the old 4.3 us gather, ~1.7 us were edge-tap copies and ~0.8 us planning. |
| `kwf_halo_exchange`: with one M-block per core, neighbours pass the d boundary rows (A: Q_1 up, B: Q_1 + Q_0 down, C: the completed rows up) instead of recomputing a halo tile-row | WIN | res3.x 15.26 -> 15.15 (flat: its final pass waited for the bias decode); sem dec res3 fc0 44.91 -> 40.97, fc1 36.22 -> 33.01 (now kw-fold), inst dec res3 fc0 25.21 -> 22.63 | Every kw-fold case with one M-block per core; more than one keeps the recompute (blocks would chain). NoC atomic increments for the counters slowed a concurrent multicast (center_head.1 +3.3%); inline absolute-value writes did not. |
| `kwf_narrow_planes`: 1-tile-wide fp32 accumulator planes packed by a replayed z-stepped PACR stream (`pack_zst`); 4x1 subblocks where 3 or 4 does not divide N | WIN (narrow) | stem.conv2 34.31 -> 32.21 | Every kw-fold case: res2 -2.9%, the rest flat. Isolated fp32 pack: 75 cycles per tile vs 100 (2-wide MOP) and 122 (1-wide MOP). 4x1 on N = 4 measured +2% to +22% (four srcB unpacks per 4 tile-MACs), so N = 4 keeps 1x4. |

Not floated (cap of 6): stem.conv1 (its RM row gather is untouched by every hook: 21.9 us with everything stubbed), the BLOCK ring tail (bias + ReLU after the last hop, ~1.5 us on res4.x), and 110-core use for res4 / res5 (spare columns cannot take a channel slice).

### Graduated (one path each, in commit order)
1. **Weight-server stream** (`e3c0367d159`). Clean merge. The bias multicast is off only under the ROW_MAJOR split slab fill, where receivers do not take the bias from `cb_bias` (no focus case).
2. **kw-fold n-groups** (`4586341772d`).
   - Merge fix: on top of (1)'s multicast bias, the writer finishes the bias after the first n-group's three rounds (not the M-block's last), or it can block on a slot only a later n-group frees.
   - `graduation_tests.patch`: the ragged weight-multicast unit test pins kw-fold's bm to its forced one-tile-row blocks.
3. **Narrow planes** (`e0f8401c6f4`), merged by hand onto (2).
   - The n-group's ACC is gw one-tile-wide planes.
   - Merge fix (measured): with the n-group width tied to the subblock width, N = 2's 4x1 subblocks made two passes and res2 regressed 12.2 -> 13.0 us. The pass width is now `_kw_fold_group` (the largest divisor of N up to 4), decoupled from the subblock: res2 12.0, stem.conv2 33.5 -> 32.1.
4. **Halo exchange** (`238c5636264`), ported by hand: the optimizer's protocol ran once per column round; on the merged tree it runs once per n-group.
   - Semaphore counts: A 2g + 1 and C 2g + 2 from below, B g + 1 from above.
   - Writer polls the exchange inside its weight-slot reserve. With n-groups and streamed sub-rounds, compute can hold both slots at a gate only this writer opens.
   - `cb_halo_in` holds 3 x n-groups pages. A per-group `pop_front(halo_seen)` straddled the 3-page ring end and tripped an LLK assert under `--dev`.
   - Adds the exchange-vs-recompute bit-exact test (11 shapes, incl. a program-cache re-run) to `test_conv2d_3x3_kw_fold.py`.
5. **Padded slab** (`3a91e2bac2c`). Clean merge. The old reader-planned tap-direct prelude stays for the shapes the padded slab cannot take (stride 2, no center taps, more than one M-block, or no leftover L1 for the slab).
6. **Balanced BLOCK split** (`faf15c942a9`), merged by hand with (5).
   - `padded_tilize` takes the row's runtime prelude rows; the slab wait and tail use the runtime slab rows.
   - The padded slab's runtime args follow the balanced split's per-M-block args.
   - `padded_slab_extent` walks the balanced ranges.
   - The exchange's writer args take the descriptor's local range list (the name `core_m_ranges` is now a function).
   - `unit_tests.patch`: the ragged-row weight-multicast unit test pins the uniform split.

Carve-outs, each stated as its reason:
- **Halo exchange: more than one M-block per core** (inexpressible): a core's first block would wait on its upper neighbour's last block. stem.conv3 keeps the recompute.
- **Padded slab: where the slab does not fit the leftover L1 plus the freed unit ring** (inexpressible). In the optimizer's set that was C_in 512 split two ways at d 4, bf16.
- **Balanced split: taken only where the model drops the costliest row by >= 2%.** Its runtime row sizing on the uniform split measured +5.0% (aspp.convs.3, median of 3) and +1.1% on a no-ragged-row shape.
- **Narrow planes: N = 4 keeps 1x4 subblocks.** This is the rule itself (4x1 measured +2% to +22%), not a guard.

Combined-tree steps (zoned, us; `ablation_r8/peel_results.txt` rows g1-g6):

| case | base | +server | +n-groups | +narrow planes | +exchange | +padded slab | +balanced split |
|---|---|---|---|---|---|---|---|
| aspp.convs.3 | 27.5 | 27.5 | 27.2 | 27.3 | 27.2 | 26.8 | 25.3 |
| sem res2 fc1 | 107.7 | 107.9 | 89.6 | 89.6 | 86.4 | 86.5 | 86.4 |
| res4.x | 16.1 | 15.6 | 15.9 | 16.1 | 16.3 | 13.9 | 14.1 |
| res3.x | 15.0 | 12.1 | 12.1 | 12.0 | 11.3 | 11.3 | 11.3 |
| stem.conv2 | 34.2 | 34.2 | 33.5 | 32.1 | 31.5 | 31.5 | 31.5 |
| aspp.convs.2 | 59.4 | 59.2 | 59.7 | 59.1 | 59.0 | 59.0 | 50.5 |
| sem dec res2 fc0 | 135.8 | 136.1 | 100.3 | 100.4 | 95.8 | 96.1 | 96.1 |
| sem dec res3 fc0 | 45.0 | 45.1 | 45.1 | 45.0 | 37.8 | 37.8 | 37.9 |

### Precision
- Every focus case is bit-identical to the base op on the final tree: kw-fold cases 16 / 16 (with and without ReLU), BLOCK cases 8 / 8, and the HEIGHT decoder layers 6 / 6, including the four that changed path (sem res2 fc1 / sem dec res2 fc0 / fc0 / fc1 onto kw-fold): PCC 0.999762 and MAE 0.42167 / 0.37630 / 0.40108 / 0.37740 on both sides. stem.conv1 and res3.0 run unchanged paths.
- The padded slab gate: 25 / 25 shapes bit-identical against the old prelude on the merged tree, under `--dev`.
- The exchange: 11 / 11 shapes bit-identical against the forced recompute, under `--dev`, and stable across a program-cache re-run.

### Whole-op result (unzoned, us; base `3ca4eb28f46` and final `faf15c942a9`, mean of 3 each, alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 26.66 | 24.49 | 1.09x | 22.7 | 60% -> 65% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 106.71 | 85.78 | 1.24x | 90.8 | 60% -> **74%** |
| 3 | res4.{1..5}.conv2 | 14.80 | 12.74 | 1.16x | 5.7 | 27% -> 31% |
| 4 | res3.{1,2,3}.conv2 | 14.79 | 10.93 | 1.35x | 5.7 | 27% -> 36% |
| 5 | stem.conv2 | 33.84 | 31.11 | 1.09x | 22.7 | 47% -> 51% |
| 6 | aspp.convs.2 | 58.80 | 49.71 | 1.18x | 45.4 | 54% -> 64% |
| 7 | aspp.convs.1 | 57.89 | 52.56 | 1.10x | 45.4 | 55% -> 60% |
| 8 | res2.{0,1,2}.conv2 | 13.20 | 11.10 | 1.19x | 5.7 | 30% -> 36% |
| 9 | center_head.1 + offset_head.1 | 11.50 | 10.02 | 1.15x | 5.7 | 35% -> 40% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 26.87 | 25.19 | 1.07x | 22.7 | 59% -> 63% |
| 11 | stem.conv3 | 48.62 | 48.53 | 1.00x | 45.4 | 65% -> 65% |
| 12 | sem dec res2.fuse_conv.0 | 134.86 | 95.39 | 1.41x | 102.1 | 53% -> **75%** |
| 13 | sem dec res3.fuse_conv.0 | 44.72 | 37.17 | 1.20x | 28.4 | 44% -> 53% |
| 14 | sem dec res3.fuse_conv.1 | 35.97 | 30.44 | 1.18x | 22.7 | 44% -> 52% |
| 15 | stem.conv1 | 22.83 | 22.55 | 1.01x | 1.1 | 3% -> 3% |
| 16 | inst dec res3.fuse_conv.0 | 25.23 | 21.71 | 1.16x | 14.2 | 39% -> 46% |
| 17 | inst dec res2.fuse_conv.0 | 31.99 | 30.19 | 1.06x | 28.4 | 62% -> 66% |
| 18 | res5.2.conv2 | 36.68 | 33.82 | 1.08x | 22.7 | 43% -> 47% |
| 19 | res5.1.conv2 | 36.06 | 33.69 | 1.07x | 22.7 | 44% -> 47% |
| 20 | res5.0.conv2 | 35.26 | 34.21 | 1.03x | 22.7 | 45% -> 46% |
| 21 | res3.0.conv2 | 17.55 | 17.53 | 1.00x | 5.7 | 23% -> 23% |
| 22 | res4.0.conv2 | 15.49 | 15.42 | 1.00x | 5.7 | 26% -> 26% |

All 3x3 layers of one Panoptic inference (each case x its calls): 1456 -> 1248 us (-14.3%).

### Guard set (whole perf harness, 41 shapes plus the 22 focus cases, unzoned, same session, 3 alternating pairs; `ablation_r8/ab_results.txt`)
- **HEIGHT (HiFi4):**
  - kw-fold / server layers: res3 -11.8%, sem res2 fuse_conv0 -10.2%, center_head1 -9.6%, res2 -8.7%, inst res3 fuse_conv0 -8.1%, sem res2 fuse_conv1 -6.0%, inst res2 fuse_conv1 -5.5%, inst res2 fuse_conv0 -5.3%, sem res3 fuse_conv0 -3.9%, sem res3 fuse_conv1 -3.1%, stem_conv2 -2.0%;
  - unchanged paths: stem_conv1 +0.3%, stem_conv3 -0.1%, res3_0 0.0%.
- **BLOCK (HiFi4 unless noted):** aspp_convs2 -21.0%, res5_2 -14.0%, res5_1 -13.9%, aspp_convs1 -13.2%, res4 LoFi -13.0%, aspp_convs3 -12.2%, res4 -6.1%, res5_0 -4.9%, res4_0 +0.3% (stride 2, unchanged path).
- **Interleaved (18 shapes):** -0.5% to +0.6%, except two cells.
  - H32_W64_Ci128 tile_bf16: +1.6% over the 3 pairs, from one 25.61 us outlier. Four more pairs: -0.3%.
  - H16_W16_Ci64 tile_bf16 / tile_bf8b: +1.1% / +2.1% over four more pairs, ~0.1-0.2 us on 10 us.
    - Two passes at every round-8 commit (`ab_guard_bisect.txt`) were non-monotonic: 9.92-10.22 and 9.98-10.25 us, moving at commits that do not touch the interleaved path. Final vs base in that pass: 10.005 vs 10.00 and 10.18 vs 10.10.
    - No graduated pattern runs on that path (interleaved: no servers, no kw-fold, no N-slice), so there is nothing to carve out. It is the same ~10 us shape flagged in rounds 1-6; recorded as code-layout noise.
- **Golden:** `eval/golden_tests/conv2d_3x3/` 1518 passed, 480 skipped (INVALID), 0 failed after every graduation and on the final tree.
- **Unit dir:** 276 / 276 through graduation 3, 287 / 287 from graduation 4 on (the 11 exchange tests).
- **`--dev`:** the kw-fold file 21 / 21, the n-group extra shapes 11 / 11, the BLOCK file 50 / 50 and the padded-slab gate 25 / 25 on the merged trees.

### Bottleneck now
- **aspp.convs.3 (priority 1, 65%).** Prelude untilize 0.9-5.0 us (fill-paced) and tilize to 9.6 us; the walk then runs to 23.8 us, paced by the weight multicast. The column sender's writer only multicasts round 3 on after its own `slab_dead_wait` (7.2-10.1 us), then every ~2 us.
- **Small configs (res3.x 10.9, res2 11.1, center_head.1 10.0, res4.x 12.7 us vs 5.7).**
  - HEIGHT: round 0 still lands at ~3.6 us. Server reads are issue-bound at one-tile DRAM reads; a larger read page would need a weight layout change.
  - res4.x: prelude 4.5 us, walk 7.7 us on 80 cores (its 100%-of-FPU time on 80 cores is 6.0 us), ring tail 1.5 us.
- **stem.conv2 (51%).** The kw-fold walk is matmul-unpack bound at ~23.5 cycles per tile-MAC. Its optimizer's untested next step: fuse the Q_0 and Q_2 rounds into 1x4 subblocks that share one in0 tile (~3.5 us estimated).
- **res5.x (46-47%).** Ring wait ~20 us and weight fetch ~21 us per core.
- **stem.conv1 (3%).** Untouched: an RM row gather (reader 18.4 us, writer 12.6 us) that the ablation hooks do not reach.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `pack_untilize_dest` / `llk_pack_untilize` MOP (compute API) | capability | No destination row offset, and the untilize MOP closes every row. A run of fp32 tiles down a one-tile-wide accumulator plane is one stream only if the packer replays strided PACRs and steps z per face pair with no ADC instruction between tiles. The 1-wide MOP measured 122 cycles per fp32 tile isolated; the replayed stream 75. | 40400 (stem.conv2, 4x1 on the 1-wide stream MOP) | 32210 | conv2d_3x3_kw_fold.hpp:150 (`pack_zst`) |
| `compute_kernel_lib::tilize` / `fast_tilize_block` | capability | It reads at the CB read pointer. The padded slab's tap blocks are runs at computed L1 addresses (row slot pitch W + d), one per (tile-row, kernel row, tap). | 15901 (res4.x, reader-published tap-direct, itself raw) | 13431 | conv2d_3x3_compute.cpp:268 (`padded_tilize`) |
| `compute_kernel_lib::untilize` | capability | It packs each tile-row at the CB write pointer. The padded slab puts each image row's tile-rows at its own zero-gapped slot, a per-tile-row L1 destination. | n/a (no helper form) | in the 13431 above | conv2d_3x3_narrow_untilize.hpp:162 (`untilize_1wide_at`), conv2d_3x3_wide_untilize.hpp (`untilize_wide` destinations) |
| `compute_kernel_lib::matmul_block` | capability | One subblock height per call, which must divide the block. The balanced split's ring row groups are ragged per grid row (1-4 rows); with runtime subblock dims a 1-row group cost ~49 cycles per k-tile vs ~46 with the height as a template constant. | 26889 (aspp.convs.3, balanced split, runtime group heights) | 26072 | conv2d_3x3_compute.cpp:705 (group dispatch), :1132 |
| `dataflow_kernel_lib` multicast (`SenderPipe`) | capability | Two more gaps on top of Perf 6's five. (1) It cannot chain a variable run of already-landed chunks into one linked multicast and close it with a conditional flag write. (2) It cannot multicast a side buffer (the bias) with its own completion flag to the same receivers. | 14950 (res3.x, one multicast per chunk, 6 chunks) | 12120 | conv2d_3x3_weight_server_writer.cpp:75, :99 |
| no helper (dataflow neighbour strip exchange) | capability | Nothing hands a d-row strip of a core's fp32 L1 buffer to a neighbour core with an absolute-count semaphore and turns arrivals into compute-visible CB tokens. NoC atomic increments for the count slowed a concurrent multicast (center_head.1 +3.3%). | n/a (recompute instead: sem dec res3 fc0 44.91) | 40970 | conv2d_3x3_writer.cpp:661 (`khx_send`), kw_fold.hpp halo gates |

The n-group passes add no raw LLK beyond the existing kw-fold bypasses (Perf 6 / 7 rows).

## Perf 9
- Date: 2026-11-19
- Scope: perf tournament (run round 4 of 5) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores, worked in priority order. No SUPPORTED change. Base `587aef8ec99` (end of Perf 8). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated. 4 graduated, all bit-identical; 1 null; 1 regression. The breakdown also found two latent deadlocks, both fixed.
  - aspp.convs.2 (priority 6): 64% -> 68% (49.70 -> 46.82 us).
  - aspp.convs.3 (priority 1): 65% -> 66% (24.48 -> 23.95 us).
  - res4.x (priority 3): 31% -> 32% (12.73 -> 12.46 us).
  - No new focus case reaches 70%; sem res2 fc1 and sem dec res2 fc0 stay above it (75%, 76%).
  - All 3x3 layers of one inference: 1248 -> 1234 us (-1.2%). No focus or guard cell regressed beyond noise.
- Instrumentation: the Perf 1-8 zones are kept. `writer_ring_credit` now spans the credit send inside the weight-wait polls; the blocking credit wait runs inside `writer_slab_dead_wait`. The batched and dead-unit tilizes run inside `compute_tilize`.
- Ablation hooks re-cut for this tree, plus the merge gate: `perf_experiments/ablation_r9/` (README).

### Latent deadlocks (fixed first)
- **`a51305feccb`.** The writer's split-fill weight reserve polled `serve_fill()` in one loop and `khx_poll()` in a second.
  - Compute can hold both weight slots at a halo-exchange gate. The writer then spun on the fill loop and never pushed the halo token.
  - It hung on every kw-fold case as soon as the `weights` ablation made the rounds land instantly. Now one loop polls both.
- **`041c7a6ab2c`.** `khx_drain()` polled only the exchange. A writer that reached it while it still owed split-fill batches deadlocked against the reader.
  - Found by the `height_server_read_split` optimizer with the server reads skipped (103-core cases).
  - The drain now also serves the fill.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only" stubs only the matmul. kw-fold cases have no tilize / untilize hook (their pack stream and final pass run under every hook).

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only |
|---|---|---|---|---|---|---|---|---|---|
| aspp.convs.3 | 25.2 | 25.4 | 23.9 | 23.8 | 22.8 | 22.0 | 19.1 | 11.5 | 22.2 |
| res4.x | 13.9 | 14.1 | 13.9 | 14.1 | 12.9 | 12.5 | 10.2 | 7.5 | 13.5 |
| res3.x | 11.5 | 11.3 | 11.5 | 9.4 | 9.4 | 9.4 | 9.4 | 7.0 | 9.9 |
| stem.conv2 | 31.5 | 31.5 | 30.4 | 29.5 | 29.5 | 29.5 | 29.5 | 18.7 | 19.5 |
| aspp.convs.2 | 50.8 | 50.9 | 48.4 | 46.2 | 45.5 | 44.9 | 37.9 | 13.3 | 35.3 |
| aspp.convs.1 | 53.1 | 53.0 | 51.6 | 49.4 | 48.9 | 47.3 | 42.2 | 12.5 | 32.6 |
| res2 | 11.8 | 11.8 | 11.2 | 11.1 | 11.1 | 11.1 | 11.1 | 7.1 | 7.8 |
| center_head.1 | 10.7 | 10.7 | 9.9 | 9.8 | 9.8 | 9.8 | 9.8 | 5.6 | 7.0 |
| res5.0 | 34.5 | 34.4 | 34.4 | 33.6 | 33.2 | 33.4 | 30.6 | 12.3 | 24.2 |
| sem dec res3 fc0 | 37.8 | 37.9 | 37.8 | 33.4 | 33.5 | 33.5 | 33.5 | 14.2 | 31.5 |
| sem res2 fc1 | 86.5 | 86.6 | 85.8 | 82.4 | 82.3 | 82.3 | 82.4 | 21.3 | 29.6 |

All 22 cases are in `ablation_r9/peel_results.txt`. Balanced bounds, from the per-core zones (`compute_matmul_walk` on the pack thread, p50 / max):

| case | full | matmul stubbed | ring writes stubbed | everything stubbed |
|---|---|---|---|---|
| res4.x | 7.49 / 8.28 | 7.33 / 7.63 | 7.11 / 7.13 | 4.51 / 4.79 |
| aspp.convs.3 | 13.04 / 15.62 | 10.27 / 11.82 | 12.93 / 14.22 | 6.82 / 7.80 |

Ranked:
1. **aspp.convs.3 prelude.** Untilize 0.9-5.1 us (fill-paced), then tilize 5.2-9.7 us. About a third of the tilized units belong to kernel rows outside the image: the walk skips them (Perf 7) but the padded tilize built them from the zero block. The walk (14 us) is a balanced matmul / ring bound.
2. **res4.x is TRISC-busy end to end.** Untilize 1.4 + tilize 2.5 + walk 7.5-8.3 + combine 1.1 of 13.9 us. The 1-wide tilize ran at ~64 cycles per tile, one DEST section and one CB sync per tile. Its walk is balanced: the matmul alone and the ring alone each give ~7.1-7.3 us. The everything-stubbed skeleton is 4.5 us.
3. **aspp.convs.1 / .2 prelude.** The untilize ends 9.3 / 8.9 us in, against 4.7 / 4.1 us with the fill payload stubbed. After that, the walk waits on the per-column weight stream: with the fill stubbed, the aspp2 walk grows 30.7 -> 34.6 us.
4. **HEIGHT server-fed layers.** Round 0 lands at ~3 us (res3.x compute waits 0.9-3.9 us). The weights peel saves 1.1-4.4 us on 8 HEIGHT cases; res2 and center_head.1 are flat, since their slab fill ends at the same time.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`).

| idea | verdict | focus result (ns, isolated) | domain / notes |
|---|---|---|---|
| `nslice_skip_dead_tilize`: the padded tilize skips the units the ring walk never reads (out-of-image kernel rows); their pages are pushed unwritten | WIN | aspp.convs.3 25085 -> 24390 (median of 3; unzoned 24309 -> 23602) | Every `padded_tilize` (stride 1, p == d BLOCK N-slice). aspp.convs.2 -3.6% (unzoned -5.0%), HiFi4 aspp_convs3 -3.1%, aspp_convs2 -1.6%, the rest flat. A dead-unit range on every core cost res4.x's 1-wide tilize +0.2-0.3 us with nothing to skip. Making the balanced M-split model count only live units: NULL. Skipping the reader's dead center rows: NULL (res4 +0.12 us). |
| `block_fill_untilize_overlap`: BLOCK untilize batches capped at 4 tile-rows, the reader keeps up to 8 in flight | WIN (small) | aspp.convs.2 50407 -> 49672, aspp.convs.1 53279 -> 52291 (medians of 3) | Every K split, descriptor only. The brief's premise was wrong: the fill takes ~5 us on the critical core, not 1.6 us. The whole slab was issued at once over congested NoC0 column links, so all its batches landed within 0.4 us of each other at the end. Ceiling (untilize stubbed): -0.76 / -1.7 us. Smaller windows, 1-2 row batches, own-shard reads on NoC1: flat or slower. |
| `block_weight_read_rate`: raise the BLOCK column weight stream's rate | WIN (different mechanism) | aspp.convs.1 53.12 -> 53.02 us (flat); res5.0 35.16 -> 34.55, res5.1 35.04 -> 34.30 us (medians of 3) | Faster reads were NULL: the read issue is NIU-bound at ~42 cycles per read, and direct register writes and rotating senders did not help. The win is the ring credit: it went out only after the first weight rounds, and columns 0, 8, 9, 10 read the same DRAM bank each round, so their late rounds held their ring neighbours. Polling the credit inside every weight wait removed the whole weight cost once the fill was stubbed (aspp1 51.34 -> 49.35, aspp2 48.24 -> 46.13 us). Every N-slice K split. |
| `height_server_read_split`: split each HEIGHT server round's DRAM reads over helper spare cores (and the server's BRISC) | NULL | res3.x 11.19 -> 11.38 us (median of 3, same position in the session) | Round 0 is multicast-bound, not read-bound: ~25-30 GB/s into the 77-core rectangle beside the slab fill. The data was ready earlier, but the chain ended at the same time. The ceiling with the server reads skipped was -0.4 to -1.2 us. Measured wins: sem dec res3 fc0 -2.6%, inst dec res3 fc0 -1.4%. Not graduated: a new helper-core kernel and protocol for ~1.3 us per inference. Micro-bench of 48 one-tile reads: one RISC 1.62 us, both RISCs of one core 1.12 us, four engines 0.81 us. Server BRISC reads regressed (+6-9%): the BRISC is the multicaster. Found the `khx_drain` deadlock. |
| `res4_narrow_tilize_batch`: 1-wide tilizes four per fp32 half-DEST section, one CB reserve / push per section | WIN | res4.x 13494 -> 12738 (median of 3) | Every 1-wide tilize (padded, tap-direct, FIFO). HiFi4 res4 -1.4%, interleaved C_in 32 flat. Isolated: 58.1 -> 42.2 ns per tile. The BH 1-wide tilize is pack-bound. A leaner pack sequence was flat in the op, so not taken. The 1-wide untilize already packs 4 rows per section. |
| `block_ring_bidir`: route half the owner slices each way round the ring, so a hop has two steps of slack | REGRESSION | res4.x 13489 -> 14185, aspp.convs.3 25377 -> 25951 (medians of 3) | Not bit-identical: max 1 bf8_b step, PCC unchanged to 1e-6. Hiding the hop is worth at most 0.36-0.54 us on res4.x, and merging the second chain in the combine costs +0.35 us. The -x chain has to run on the writer's NoC1, where it slows the weight multicast. res5 +1-3%. It wins -22% only on a synthetic 5-of-11-owner shape. Side finding: aspp.convs.3's ~1.7 us ring stall comes from its 3 non-owning columns sitting together at the end of the ring (`block_ring_bidir/sched_model.py`). |

Not floated (cap of 6): stem.conv1 (its face-read gather, ~84 cycles per 512 B face on two RISCs), stem.conv2's kw-fold walk (matmul-unpack bound), and 110-core use for res4.x / res5.x.

### Graduated (one path each, in commit order)
1. **1-wide tilize batching** (`39dcb6c1568`). Clean apply. `NarrowTilizeBatch` serves the padded, tap-direct and FIFO tilizes at width 1.
2. **Dead-unit tilize skip** (`d898aa3d8f9`), merged by hand onto (1): a dead run's pages are pushed only after the open batched section is closed, so CB order holds.
3. **BLOCK fill batches and window** (`412ea2f4432`). Clean apply, descriptor only.
4. **Ring credit in the weight waits** (`f7f6b1a5b35`). Clean apply. `writer_ring_credit` is kept and now spans the credit send.
5. **`51d108bac8e`.** No behaviour change: (3)'s predicate is rewritten as a carve-out, and the comment records the measurement that earned it.

Carve-outs, each with the reason that earned it:
- **Fill batches and window: only under a K split.** The same values on every layout were measured unzoned, 2 alternating runs each with non-overlapping ranges (`ablation_r9/ab_results_uncond.txt`):
  - DRAM interleaved H64 W128 C_in 128 TILE bf8_b +1.9%, H128 W256 C_in 64 TILE bf8_b +2.1%;
  - HEIGHT stem.conv2 / res3.0 @ LoFi +1.4% / +1.7%.
- **Dead-unit skip: only on cores where some prelude row has a dead unit.** The per-row live range on a core with nothing to skip measured res4.x's 1-wide tilize +0.2-0.3 us. The choice is made once per M-block, between two instantiations.
- **Ring credit polling: only on the N-slice walk.** Not a guard: the ring credit exists only there.

### Precision
- Every focus case is bit-identical to the base op on the final tree. The 22 / 22 merge gate (`ablation_r9/merge_gates/test_focus_exact.py`) runs the final tree against `587aef8ec99` in one process, with 0 differing elements on every case.
- Each optimizer's own sweep was bit-identical across its domain (17 cases for the skip, the fill and the credit).

### Whole-op result (unzoned, us; base `587aef8ec99` and final `f7f6b1a5b35`, mean of 3 each, alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 24.48 | 23.95 | 1.02x | 22.7 | 65% -> 66% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 85.79 | 85.00 | 1.01x | 90.8 | 74% -> 75% |
| 3 | res4.{1..5}.conv2 | 12.73 | 12.46 | 1.02x | 5.7 | 31% -> 32% |
| 4 | res3.{1,2,3}.conv2 | 11.08 | 10.99 | 1.01x | 5.7 | 36% -> 36% |
| 5 | stem.conv2 | 31.12 | 31.03 | 1.00x | 22.7 | 51% -> 51% |
| 6 | aspp.convs.2 | 49.70 | 46.82 | 1.06x | 45.4 | 64% -> 68% |
| 7 | aspp.convs.1 | 52.62 | 52.01 | 1.01x | 45.4 | 60% -> 61% |
| 8 | res2.{0,1,2}.conv2 | 11.13 | 11.10 | 1.00x | 5.7 | 36% -> 36% |
| 9 | center_head.1 + offset_head.1 | 10.03 | 10.02 | 1.00x | 5.7 | 40% -> 40% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 25.05 | 25.07 | 1.00x | 22.7 | 63% -> 63% |
| 11 | stem.conv3 | 48.51 | 48.57 | 1.00x | 45.4 | 66% -> 65% |
| 12 | sem dec res2.fuse_conv.0 | 95.22 | 94.35 | 1.01x | 102.1 | 75% -> 76% |
| 13 | sem dec res3.fuse_conv.0 | 37.17 | 37.16 | 1.00x | 28.4 | 53% -> 53% |
| 14 | sem dec res3.fuse_conv.1 | 30.45 | 30.19 | 1.01x | 22.7 | 52% -> 53% |
| 15 | stem.conv1 | 22.50 | 22.70 | 0.99x | 1.1 | 3% -> 3% |
| 16 | inst dec res3.fuse_conv.0 | 21.71 | 21.77 | 1.00x | 14.2 | 46% -> 46% |
| 17 | inst dec res2.fuse_conv.0 | 30.23 | 30.28 | 1.00x | 28.4 | 66% -> 66% |
| 18 | res5.2.conv2 | 33.85 | 33.88 | 1.00x | 22.7 | 47% -> 47% |
| 19 | res5.1.conv2 | 33.70 | 33.75 | 1.00x | 22.7 | 47% -> 47% |
| 20 | res5.0.conv2 | 34.12 | 33.70 | 1.01x | 22.7 | 47% -> 47% |
| 21 | res3.0.conv2 | 17.47 | 17.47 | 1.00x | 5.7 | 23% -> 23% |
| 22 | res4.0.conv2 | 15.53 | 15.52 | 1.00x | 5.7 | 26% -> 26% |

All 3x3 layers of one Panoptic inference (each case x its calls): 1248 -> 1234 us (-1.2%).

Smaller in the combined tree than in isolation:
- res4.x -2.1% unzoned vs the tilize batch's -5.6% zoned. The zones alone cost res4.x ~1.2 us (13.9 zoned vs 12.73 unzoned at base).
- res5.1 is flat (33.70 -> 33.75) against its optimizer's -1.8% unzoned.

### Guard set (whole perf harness, 41 shapes plus the 22 focus cases, unzoned, same session, 3 alternating pairs; `ablation_r9/ab_results.txt`)
- **HEIGHT (HiFi4).**
  - kw-fold decoder / res layers: res3 -2.6%, sem res3 fuse_conv1 -2.7%, sem res3 fuse_conv0 -2.2%, inst res3 fuse_conv0 -1.2%, inst res2 fuse_conv1 -1.0%, inst res2 fuse_conv0 -0.9%, sem res2 fuse_conv1 -0.9%, sem res2 fuse_conv0 -0.7%. Of this round's changes, only the two writer fixes run there; not bisected.
  - The rest within ±0.5%: stem_conv1 -0.5%, stem_conv3 0.0%, stem_conv2 0.0%, res2 -0.1%, res3_0 -0.3%, center_head1 +0.2%.
- **BLOCK (HiFi4 unless noted).** aspp_convs3 -3.1%, res4 LoFi -3.0%, aspp_convs2 -2.0%, res4 -1.2%, aspp_convs1 -0.7%. res5_0/1/2 and res4_0 within +0.2%.
- **Interleaved (18 shapes).** -0.7% to +0.4%, except H16_W16_Ci64 tile_bf8b at +2.3% (base 10.00-10.10, final 10.17-10.46 us).
  - Bisected over the round's 7 commits, two passes (`ablation_r9/ab_guard.txt`). The cell moved 9.96-10.34 us non-monotonically, at commits that do not run on it.
  - Final read 9.96 / 10.04 us against base 10.13 / 10.27 us.
  - This is the same layout-noise cell as Perf 8: nothing to carve out.
- **Golden.** `eval/golden_tests/conv2d_3x3/` on the final tree: 1518 passed, 480 skipped (INVALID), 0 failed (test_op in 4 shape chunks, plus test_op_loose and test_regression).
- **Unit dir.** 224 / 224 (the perf file runs separately: 63 / 63 with PCC asserted, in each A/B run).
- **`--dev`.** The BLOCK, kw-fold and HEIGHT files, 112 / 112.

### Bottleneck now
- **aspp.convs.3 (priority 1, 66%).** Tilize 4.9-7.8 us (was 5.2-9.7). The walk then runs 7.8-23.1 us: it absorbed most of the prelude gain.
  - The walk is a balanced matmul / ring bound with 1-tile N slices.
  - The 3 non-owning columns sitting together at the ring's end cost ~1.7 us of stall (`block_ring_bidir/sched_model.py`). Spreading them needs the output written from any column.
- **res4.x (32%).** Tilize 1.8 us, walk 7.4-8.3 us, combine 1.5 us. Still TRISC-busy; 80 of 110 cores.
- **aspp.convs.1 / .2 (61% / 68%).** The walk is at the ~19 cycles per tile-MAC floor of 4-tile subblocks.
  - aspp1's prelude still ends ~15 us in: untilize to 9.4 us, tilize to 15.3 us.
  - Next is the reader's center copy after the fill (~2.1-2.5 us, ceiling -0.76 us on aspp2).
- **HEIGHT small configs (res3.x 11.0, res2 11.1, center_head.1 10.0 us vs 5.7).** Round 0 is multicast-bound beside the slab fill.
- **stem.conv1 (3%).** Unchanged.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` | capability | It reads only at the CB read pointer. At block width 1 it runs one DEST section, one math/pack handshake and one CB reserve / push per tile. A 1-wide tilize is pack-bound, so the per-tile section release is a third of its cost. Nothing fills a fp32 half-DEST section with four 1-tile tilizes from computed L1 sources and packs them as one section. | 13494 (res4.x zoned, median of 3; one section per tile: the helper's width-1 structure, raw at computed sources) | 12738 | conv2d_3x3_compute.cpp:194 (`NarrowTilizeBatch`) |
| `dataflow_kernel_lib` multicast (`ReceiverPipe::receive`) | capability | No polling form. The receive acks the sender and blocks on the data-ready semaphore, and the caller cannot run its own work during the wait. Here that work is giving the ring credit and serving the split fill. The op already re-implemented it once for the fill (`receive_serving`); the credit needed a second copy. | 35040 (res5.1 zoned, median of 3, isolated; credit after the first rounds; flat in the combined unzoned A/B) | 34300 | conv2d_3x3_writer.cpp:1003 (`receive_polling`) |

The dead-unit skip adds no raw LLK beyond `padded_tilize`'s existing bypass (Perf 8 row). The fill batches and window are host-only.

## Perf 10
- Date: 2026-11-19
- Scope: perf tournament (one round) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores, worked in priority order. No SUPPORTED change. Base `6981db2a54c` (end of Perf 9). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated and all 6 graduated. One carve-out each was earned by measured regressions on two of them (the tap shift and the early writer fill).
  - aspp.convs.3 (priority 1) crosses 70%: 66% -> 74% (23.93 -> 21.35 us).
  - aspp.convs.2 (priority 6) reaches 69.7% (46.96 -> 45.56 us, target 45.39).
  - stem.conv1 runs 2.18x faster (22.53 -> 10.36 us).
  - All 3x3 layers of one inference: 1233 -> 1183 us (-4.1%).
  - 21 of 22 focus cases are bit-identical to the base. stem.conv1 is within one bf8_b step on 0.43% of outputs, with PCC and MAE unchanged.
  - No focus or guard cell regressed beyond noise.
- Instrumentation: the Perf 1-9 zones are kept. New zones: `compute_kwf_side_rounds` (fused pass) and `writer_fill_early`. The tap shift runs inside `compute_tilize`; on that path `compute_untilize` is empty. The slab-direct stem path runs inside `compute_matmul_stream`.
- Priority: the feature_spec priorities come from `5f23b256d1a`. Recomputed on the base, the largest gaps x calls are res4.x 33.8, res5.x ~33 (three layers), stem.conv1 21.6, res3.x 21.2 and res2 16.2 us per inference. stem.conv1 (spec priority 15) was therefore floated too.
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r10/`. `ablation_hooks.patch` is the round-9 patch re-cut for `6981db2a54c`: the padded-tilize stub follows its new signature, and the N-slice weight hunk follows the polled credit. `focus.sh`, `ab_unzoned.sh` and `ab_unzoned_ids.sh` are the run scripts, and `merge_gates/` holds the bit-exact gate.
- Roofline gate: as in Perf 8 / 9, measured rates against the LoFi FPU floor (16 cycles per tile-MAC) and the NoC link rate (~86 GB/s), not the `noc_estimate` CLI.
  - At the floor (no ideas spent): the res5.x walk (2016 MACs in ~25 us) and the aspp.convs.1 walk.
  - Far from it:
    - HEIGHT weight rounds: res2's 13 KB round 0 lands 3.3 us in, and res3.x streams 157 KB in 7.3 us (21 GB/s);
    - kw-fold at N <= 2: 23-26 cycles per MAC;
    - the stem.conv1 gather: 233 KB of 512 B local reads in ~14 us.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only" stubs only the matmul. All 22 cases are in `ablation_r10/peel_results.txt`, and the per-stage zones in `zones_*.txt`.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only |
|---|---|---|---|---|---|---|---|---|---|
| stem.conv1 | 22.8 | 22.8 | 22.9 | 22.0 | 22.0 | 22.1 | 22.1 | 21.8 | 22.8 |
| stem.conv2 | 31.5 | 31.5 | 30.4 | 29.5 | 29.5 | 29.5 | 29.5 | 18.7 | 19.4 |
| res2 | 11.7 | 11.7 | 11.1 | 11.1 | 11.1 | 11.1 | 11.1 | 7.1 | 7.8 |
| res3.x | 11.6 | 11.6 | 11.2 | 9.4 | 9.4 | 9.4 | 9.4 | 7.1 | 9.9 |
| res4.x | 13.5 | 13.2 | 13.4 | 12.8 | 12.3 | 11.9 | 10.3 | 7.5 | 13.1 |
| res5.0 | 34.3 | 34.2 | 33.4 | 33.9 | 33.9 | 33.9 | 31.2 | 12.9 | 23.5 |
| aspp.convs.1 | 52.4 | 52.6 | 49.4 | 49.4 | 48.8 | 47.7 | 42.4 | 13.1 | 35.1 |
| aspp.convs.2 | 47.7 | 47.5 | 44.4 | 44.4 | 43.7 | 43.4 | 38.7 | 14.2 | 31.9 |
| aspp.convs.3 | 24.5 | 24.9 | 22.9 | 22.8 | 21.8 | 21.4 | 19.5 | 11.9 | 20.7 |
| sem dec res3 fc0 | 37.9 | 37.8 | 37.8 | 33.5 | 33.5 | 33.5 | 33.4 | 14.1 | 31.5 |
| center_head.1 | 10.7 | 10.7 | 9.9 | 9.8 | 9.8 | 9.8 | 9.8 | 5.6 | 6.7 |

Ranked:
1. **HEIGHT weight stream (every server-fed kw-fold layer, ~20 calls per inference).**
   - Round 0 lands 3.3 us in on res2 and center_head.1 (only 12 tiles, 13 KB), and 3.7 us in on res3.x. On res3.x the later rounds land at ~5.3 and ~7.7 us and co-pace the matmul to 10.1 us.
   - Server chain: one-tile DRAM reads at ~2 us per 52 KB round, then the multicast and its flush.
   - The weights peel is worth ~40 us per inference across the HEIGHT layers.
2. **BLOCK N-slice prelude.** Untilize plus tilize before the walk: aspp.convs.1 15 us, aspp.convs.3 7 us, res5.x 6.5 us, res4.x 3.6 us (res4.x is TRISC-busy end to end).
3. **aspp ring tail.** Column 0 ends ~2 us after the others on all three aspp layers; the three non-owning columns sit together at the ring's end.
4. **kw-fold at N <= 2.** 4x1 subblocks are unpack-bound: stem.conv2 ~23.5 cycles per tile-MAC, res2 ~26.
5. **stem.conv1.** The face-direct NoC gather (reader 18.2, writer 12.6 us) paces it; no hook reaches it.
- At the floor:
  - res5.x's walk on 80 cores is FPU-bound. Only more cores move it, and even a 110-core split leaves it under 70%.
  - res3.x's matmul is ~5.1 us of pure FPU time on 86 cores, above most of its 5.7 us target.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Numbers are unzoned unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `nslice_ring_nonowners`: spread the non-owning columns round the N-slice ring (owners {0,1,2,4,5,6,8,9} for 8 slices on 11 columns); each owner stores its slice through the remote output store | WIN | aspp.convs.3 23.66 -> 22.51 (median of 3) | aspp.convs.2 -2.3%, aspp.convs.1 flat (its column 0 is late from its prelude), res4 / res5 inactive. Every bit-identical-by-construction variant was flat or slower: each slice's fp32 sum order fixes the walk at 8 steps + 3 hops, which HEAD already reaches. The spread changes that order. It matched HEAD exactly on all 7 shapes run (bf8_b x bf8_b partials are exact in fp32 here), but that is not guaranteed in general. Inexpressible on a local output shard. |
| `height_weight_rate`: server-fed HEIGHT weights ND-sharded over the DRAM banks (one shard per n-group tile-row); the server reads page runs; a third K-blocked kw-fold weight slot where it fits | WIN | res3.x 11.15 -> 10.42 | sem dec res3 fc0 -2.8%, inst dec res3 fc0 -2.4%, inst dec res2 fc0 -2.3%, inst res2 fc1 -1.4%, the rest flat. Bit-identical. One server per sub-rectangle was not graduated: the multicast is not sender-bound (~59 B/cycle to 9 or 64 cores), and sem res2 fc1 measured +0.38 us. |
| `height_round0_latency`: a burst-sized round 0 read with all chunks in flight and the bias after it; cheaper fill hand-over; the writer issues its first-block fill at start; kw-fold skips the zero block | WIN | res2 11.17 -> 10.48 | center_head.1 -5.7%, inst res2 fc1 -1.6%, the rest flat. Bit-identical. Compute's first subblock waits on round 0 and on the neighbour's slab rows, which land together. Server placement by launch order measured flat: the "geometric kernel start" is mostly per-core profiler clock offset. |
| `kwf_fused_side_rounds`: at a 2-tile n-group, Q_0 and Q_2 run as one pass of 1x4 subblocks (one in0 tile feeds 4 MACs) over interleaved weight rounds | WIN | stem.conv2 31.13 -> 28.21 | res2 -6.5%; forced K-blocked N = 2 at HiFi4 -4.4%. Bit-identical. The fused pass runs at ~18.8 cycles per MAC instead of ~23.4. Exception (measured): 1-tile n-groups, center_head.1 +13.7% (1x2 calls, ~40 cycles of per-call unpack cost). Q_0, Q_1 and Q_2 in one pass is inexpressible under the seedless overwrite. Found and fixed a K-blocked 2-slot ring wrap (inf / NaN). |
| `fpu_tap_shift`: the kw = 0 / 2 taps built on the FPU from the input's TILE slab (whole-tile UNPACR to SrcB, one-row or 4-row MOVB2D into DEST, pack), kernel-row major behind the fill; no untilize, no row-major slab | WIN (carve-out) | res4.x 12.43 -> 12.05 (median of 3) | Per tap tile the shift is dearer than the 1-wide tilize (62 vs 42 ns). It wins on schedule: the untilize and the center-push wait go. A shift-matrix matmul is not exact at LoFi (in1 keeps 5 bits). Bit-identical. aspp.convs.1 +1.4% in isolation, earned a carve-out on the merged tree (below). |
| `stem1_slab_tilize`: the reader and writer fill a padded slab; compute unpacks each in0 tile as two 1 KB slab runs (patched SrcB ADDR_MODs); the bias is one extra K-step (ones x bias tile, split into two exact parts at LoFi) carried on the weight multicast | WIN | stem.conv1 22.79 -> 11.15 (median of 3) | RM face-direct shapes -16% to -52%. Option menu: fused bias 11153 ns, within 1 output ulp (fp32 instead of tf32 partials before the bias); separate bias pass 14041 ns, bit-identical. The fused form graduated: PCC 0.9998235 and MAE 0.066462 vs fp32 torch on both. Exceptions: W_out % 32 != 0 is inexpressible (a tile-row spans two output rows) and keeps the gather. 32x34x16 valid padding, 30 cores measured +7.6% (4544 -> 4890 ns): weight-round-bound and not in the model. Not carved out: no predicate states that mechanism, and the gather path stays only for the inexpressible shapes. |

Not floated (cap of 6):
- **BLOCK on 110 cores for res4 / res5.** It needs a (channel slice, kernel row) K-unit split: 48 units over 11 columns, -17% on res5's walk, which still leaves res5 under 70%. It re-keys every BLOCK stage off "column = channel slice" and collides with the ring and prelude ideas.
- **kw-fold final-pass overlap.** All three TRISCs are busy in both phases.
- **BLOCK ring-tail fusion.** The seed / reload re-init costs as much as the combine (Perf 6 `height_bias_fold`).

### Graduated (one path each, in commit order)
1. **Ring spread** (`0716fdec09e`). Clean apply, no raw LLK.
2. **ND-sharded weights and page runs** (`9666ae92144`). Clean apply. `prepare_conv2d_weights` places the weights with `weight_memory_config`. Placements without servers keep DRAM interleaved: they have no server to read runs.
3. **Round-0 latency** (`d39235c069f`). The server reader's three hunks were merged by hand onto (2)'s page runs. The burst-sized round 0 issues every chunk's page runs first.
4. **Fused side rounds** (`b0466aaf5d7`), with four K-blocked N = 2 shapes added to `test_conv2d_3x3_kw_fold.py`.
   - Merge fix 1: the fused rounds' slot-tile permutation now reads through (2)'s page runs. A run is one n-group's kw = 0 or kw = 2 columns of a (kh, c) row: gw tiles inside one shard.
   - Merge fix 2: a `static_assert` with a zero divisor in a discarded `if constexpr` failed to build, because `kernel_main` is not a template.
5. **Tap shift** (`1eef0835a5e`). One descriptor define hunk merged by hand.
6. **stem.conv1 slab-direct** (`aeba2aed511`). Five hunks were merged by hand around (1)'s `_ring_owner_defines` and (5)'s tap-shift include and define.
7. **Early writer fill carve-out** (`da787eee785`), found in the guard set (below).

Carve-outs, each with the measurement that earned it:
- **Tap shift: halo tile-rows per side (d W / 32) < 3 x slice tiles, on slices of 2+ tiles** (`TAP_SHIFT_MIN_HALO_PER_TILE`).
  - Merged tree, shift on vs off, unzoned, 3 alternating pairs: aspp.convs.1 52.75 vs 51.35 (+2.7%), res5.0 33.34 vs 32.98 (+1.1%), non-overlapping.
  - Winners: aspp.convs.3 -6.0%, res5.2 -3.4%, aspp.convs.2 -3.0%, res5.1 -2.9%, res4.x -2.4%.
  - Mechanism: the shift rebuilds every tap tile at ~60 ns and pays by removing the halo untilize. Losers sit at halo = 2 x tiles, winners at >= 4 x tiles or on 1-tile slices, whose padded tilize is the slow 1-wide one. 3 is the midpoint of the unmeasured gap.
  - Confirmed: aspp.convs.1 runs the untilize again (7.1 us zone), and res5.0 reads 33.0 us (= off).
- **Early writer fill: only under weight servers.**
  - The guard set showed stem_conv2 @ HiFi4 +0.6%. The commit bisect put it on (3): stem_conv2 LoFi 31.00 -> 31.21 and HiFi4 75.73 -> 75.83 at `d39235c069f`, both reproduced.
  - Under the in-grid base sender, round 0 is multicast right into the early fill's reads.
  - HEAD vs the carve-out, unzoned, 2 pairs: stem.conv2 @ LoFi 28.33 -> 27.42, @ HiFi4 76.09 -> 74.95, stem.conv3 48.31 -> 47.84.
  - Server-fed layers compile the same code.
- **Inexpressible, not guards:** fused rounds at 1-tile n-groups (measured), slab-direct at W_out % 32 != 0, ring spread on a local output shard.

Combined-tree steps (zoned, us; `ablation_r10/focus.txt`):

| case | base | +ring | +page runs | +round 0 | +fused | +tap shift (carved) | +stem1 | +early-fill carve |
|---|---|---|---|---|---|---|---|---|
| stem.conv1 | 22.8 | 22.9 | 22.7 | 23.0 | 23.0 | 23.0 | 11.2 | 11.2 |
| stem.conv2 | 31.5 | 31.5 | 31.5 | 31.8 | 28.9 | 28.9 | 29.0 | 27.8 |
| res2 | 11.7 | 11.7 | 11.7 | 11.3 | 10.8 | 10.7 | 10.6 | 10.4 |
| res3.x | 11.6 | 11.6 | 10.7 | 10.6 | 11.1 | 10.7 | 11.0 | 10.9 |
| res4.x | 13.5 | 13.6 | 13.9 | 13.6 | 13.6 | 13.5 | 13.3 | 13.4 |
| aspp.convs.2 | 47.7 | 46.8 | 46.8 | 47.3 | 47.1 | 45.8 | 46.2 | 46.2 |
| aspp.convs.3 | 24.5 | 22.9 | 23.5 | 23.7 | 23.4 | 21.4 | 21.4 | 21.7 |
| center_head.1 | 10.7 | 10.6 | 10.7 | 10.0 | 10.1 | 10.1 | 10.1 | 10.1 |

res3.x is bimodal zoned (~10.7 / ~11.1 us on identical code); unzoned it is -5.4% (below).

### Precision
- **21 of 22 focus cases are bit-identical to `6981db2a54c`** (`ablation_r10/merge_gates/test_focus_exact.py`, one process, 0 differing elements). That includes aspp.convs.1 / 2 / 3 under the ring spread.
- **stem.conv1:** 36089 of 8388608 outputs differ (0.43%), max 0.25, about one bf8_b step at its output magnitude. Cause: the fused bias adds to fp32 partials, where the base truncated them to tf32 first. PCC and MAE against a fp32 torch conv are equal (0.9998235 / 0.066462).
- **No-bias stem shapes** are bit-identical.

### Whole-op result (unzoned, us; base `6981db2a54c` and final `da787eee785`, mean of 3 each, alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 23.93 | 21.35 | 1.12x | 22.7 | 66% -> **74%** |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 85.12 | 84.11 | 1.01x | 90.8 | 75% -> 76% |
| 3 | res4.{1..5}.conv2 | 12.36 | 12.06 | 1.02x | 5.7 | 32% -> 33% |
| 4 | res3.{1,2,3}.conv2 | 11.01 | 10.41 | 1.06x | 5.7 | 36% -> 38% |
| 5 | stem.conv2 | 31.02 | 27.33 | 1.13x | 22.7 | 51% -> 58% |
| 6 | aspp.convs.2 | 46.96 | 45.56 | 1.03x | 45.4 | 68% -> 69.7% |
| 7 | aspp.convs.1 | 51.59 | 51.24 | 1.01x | 45.4 | 62% -> 62% |
| 8 | res2.{0,1,2}.conv2 | 11.12 | 10.31 | 1.08x | 5.7 | 36% -> 39% |
| 9 | center_head.1 + offset_head.1 | 10.03 | 9.73 | 1.03x | 5.7 | 40% -> 41% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 25.08 | 24.35 | 1.03x | 22.7 | 63% -> 65% |
| 11 | stem.conv3 | 48.57 | 47.80 | 1.02x | 45.4 | 65% -> 66% |
| 12 | sem dec res2.fuse_conv.0 | 94.43 | 93.93 | 1.01x | 102.1 | 76% -> 76% |
| 13 | sem dec res3.fuse_conv.0 | 37.21 | 35.86 | 1.04x | 28.4 | 53% -> 55% |
| 14 | sem dec res3.fuse_conv.1 | 30.30 | 29.39 | 1.03x | 22.7 | 52% -> 54% |
| 15 | stem.conv1 | 22.53 | 10.36 | 2.18x | 1.1 | 3% -> 7% |
| 16 | inst dec res3.fuse_conv.0 | 21.76 | 20.43 | 1.07x | 14.2 | 46% -> 49% |
| 17 | inst dec res2.fuse_conv.0 | 30.26 | 29.31 | 1.03x | 28.4 | 66% -> 68% |
| 18 | res5.2.conv2 | 33.85 | 32.24 | 1.05x | 22.7 | 47% -> 49% |
| 19 | res5.1.conv2 | 33.78 | 32.84 | 1.03x | 22.7 | 47% -> 48% |
| 20 | res5.0.conv2 | 33.69 | 33.47 | 1.01x | 22.7 | 47% -> 47% |
| 21 | res3.0.conv2 | 17.45 | 17.45 | 1.00x | 5.7 | 23% -> 23% |
| 22 | res4.0.conv2 | 15.63 | 15.50 | 1.01x | 5.7 | 25% -> 26% |

All 3x3 layers of one Panoptic inference (each case x its calls): 1233 -> 1183 us (-4.1%). Cases at or above 70%: sem res2 fc1, sem dec res2 fc0, aspp.convs.3.

### Guard set (whole perf harness, 41 shapes plus the 22 focus cases, unzoned, same session, 3 alternating pairs; `ablation_r10/ab_results.txt`)
- **HEIGHT (HiFi4).**
  - stem_conv1 -14.0%, inst_res3_fuse_conv0 -1.8%, center_head1 -1.3%.
  - stem_conv2, sem_res3_fuse_conv0, inst_res2_fuse_conv0 and inst_res2_fuse_conv1 -0.9% each; res3 -0.8%; sem_res3_fuse_conv1 -0.7%.
  - stem_conv3 -0.5%, res2 -0.4%, sem_res2_fuse_conv0 / conv1 -0.2%, res3_0 0.0%.
  - Before the early-fill carve-out stem_conv2 read +0.6% (`ab_results_pre_carve.txt`); that is what started the bisect.
- **BLOCK (HiFi4 unless noted).** aspp_convs3 -4.6%, res4 LoFi -2.7%, aspp_convs2 -1.2%, res5_2 -1.1%, res4 -0.9%, res5_1 -0.8%, aspp_convs1 -0.3%, res5_0 -0.2%, res4_0 +0.2% (stride 2, unchanged path).
- **Interleaved (18 shapes).** -1.6% to +0.4%, except H16_W16_Ci64 tile_bf16 / tile_bf8b at +2.9% / +2.9%.
  - No change of this round runs on that path: interleaved has no servers, no kw-fold, no N-slice and no slab-direct, and its weights stay DRAM interleaved.
  - Bisected over the round's 8 commits, two passes (`ablation_r10/ab_guard.txt`): the cells move 9.95-10.40 us non-monotonically at commits that do not run on them. Final 10.01 / 10.02 and 10.25 / 10.07 against base 10.03 / 10.02 and 10.03 / 10.26.
  - Same layout-noise cell as Perf 8 / 9: no carve-out.
- **Golden.** `eval/golden_tests/conv2d_3x3/`: 1518 passed, 480 skipped (INVALID), 0 failed. Run after (4), after (6) and on the final tree.
- **Unit dir.** 228 / 228, including the 4 new K-blocked fused shapes.
- **`--dev`.** BLOCK file 50 / 50 after (5); HEIGHT, layouts and kw-fold files 135 / 135 after (6).

### Bottleneck now
- **aspp.convs.2 (69.7%, 0.17 us short).** Its walk is at the 4-tile-subblock floor. The prelude now runs behind the fill.
- **aspp.convs.1 (62%).** The carve-out keeps its padded-slab prelude: untilize to ~9 us, then tilize to ~15 us, before a 36 us walk at the FPU floor. A tap shift that only beats the 6-wide fast tilize needs a cheaper per-tile shift than ~60 ns.
- **Small configs (res3.x 10.4, res2 10.3, center_head.1 9.7, res4.x 12.1 us vs 5.7).**
  - HEIGHT: round 0 still lands ~2.2 us in. res3.x's matmul alone is ~5.1 us of FPU time on 86 cores.
  - res4.x: TRISC-busy (prelude ~3 us, then a walk balanced between matmul and ring), on 80 of 110 cores.
- **stem.conv1 (7%).** Compute waits for the whole slab (~6 us on the slowest core); streaming the matmul behind per-row slab tokens is estimated at -2 to -3 us. Its matmul runs ~31 cycles per tile-MAC at 2x2 subblocks.
- **stem.conv2 (58%).** The Q_1 round, a third of the MACs, stays 4x1: N = 2 x 3 columns needs 6 fp32 DEST tiles. The rest is bound by ~40 cycles of unpack cost per matmul call.
- **res5.x (47-49%).** The walk is at the 80-core FPU floor. Only a (channel slice, kernel row) split over 11 columns moves it, and only by ~17%.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` / `untilize` | capability | No row shift and no tile assembled from row ranges of two tiles. A tap tile for column shift d is rows [d, 32) of input tile t plus rows [0, d) of tile t + 1, with zero rows at image-row edges. The helpers read and write whole tiles at CB pointers. Outside the matmul nothing unpacks a whole tile into one Src bank: the stock AB unpack flips the bank every face. The raw path is a full-tile UNPACR into SrcB plus one-row MOVB2D into DEST (4-row only when source and DEST rows agree mod 8). | 12490 (res4.x, merged tree, padded-slab untilize + raw `padded_tilize`, tap shift off, mean of 3) | 12190 | conv2d_3x3_tap_shift.hpp:67-156 (`load`, `move_rows`, `run`), conv2d_3x3_compute.cpp:1235 |
| `compute_kernel_lib::matmul_block` | capability | in0 is unpacked from the CB read pointer at a tile stride, one 4-face read per tile. stem.conv1's in0 tile is two 1 KB runs of the padded slab at computed addresses (`a` and `a + 32 d` bytes). That needs a per-tile L1 source with a patched SrcB x_dim / Z stride and matmul ADDR_MODs that step SrcB by 16 rows; nothing exposes those. | 22787 (stem.conv1, face-direct NoC gather + helper-shaped matmul) | 11153 | conv2d_3x3_slab_direct.hpp (`unpack_k_step`, `math_patch_addrmods`), conv2d_3x3_compute.cpp:1634-1674 |
| `compute_kernel_lib::add_bias_bcast_rows` | capability | The bias is a separate pass over the packed tf32 partials. Folding it into the matmul as one extra K-step (ones x bias) needs a matmul that takes an extra in0 / in1 pair from non-CB addresses. At LoFi the bias must also be split into two exact parts, because in1 keeps 5 significant bits. | 14041 (stem.conv1 slab-direct with the helper bias pass, bit-identical) | 11153 | conv2d_3x3_slab_direct.hpp:109-144 (bias split), conv2d_3x3_compute.cpp:1674 |
| `compute_kernel_lib::matmul_block` | capability | One in1 block per call and one pack destination per output tile. The fused side-round pass takes two weight columns' in1 tiles from one interleaved row and packs them at two different accumulator row offsets: Q_0 at +d and Q_2 at -d, with a packer drain between. | n/a (the kw-fold walk was already raw: 31130 at 4x1, stem.conv2) | 28210 | conv2d_3x3_kw_fold.hpp:87 (justification), :478 (`compute_kwf_side_rounds`) |

The ring spread, the ND-sharded weights with page runs, the round-0 changes and the early-fill carve-out use no raw LLK and bypass no helper. The page runs are plain `noc_async_read` through the existing TensorAccessor.
