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

## Perf 11
- Date: 2026-11-19
- Scope: perf tournament (one round) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores. No SUPPORTED change. Base `039e18f6c81` (end of Perf 10). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated (cap 6), 4 graduated, 2 null.
  - All 3x3 layers of one inference: 1182 -> 1143 us (-3.3%, whole perf file, 3 alternating pairs); 1183.5 -> 1143.2 us (-3.4%) in the focus-only context.
  - res5.0 / 5.1 / 5.2 -18% to -22% (47-49% -> 58-60%), on 104 cores instead of 80.
  - aspp.convs.2 crosses 70% (45.38 -> 42.13 us, 75%); aspp.convs.1 62% -> 66%.
  - All 22 focus cases are bit-identical to the base, stem.conv1 included.
  - Measured costs kept: stem.conv2 +0.66% (the slab-skip fill loop, below), res2 +1.6% focus-only (no single commit beyond noise).
- Instrumentation: the Perf 1-10 zones are kept. New zones: `compute_sd_weight_wait`, `writer_sd_weight_early`, `writer_sd_weight_land`, `writer_weight_send` (stem streaming), `writer_spare_forward` (spare groups). The new tap shift runs inside `compute_tilize`; the slab skip adds no stage.
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r11/`.
  - `ablation_hooks.patch` is the round-10 patch re-cut for `039e18f6c81`, plus stubs the round-10 paths lacked: the tap shift (`tilize`) and the fused side rounds' matmul (`matmul`).
  - Trap: shifting line numbers made two zone source locations collide in the profiler's 16-bit hash ("Source location hashes are colliding" at device close). `zone_hash_check.py` hashes every zone location offline; a one-line shift fixed it.
  - `focus.sh`, `run_peel.sh`, `ab_unzoned.sh`, `ab_unzoned_ids.sh`, `ab_report.py` as in round 10; `merge_gates/test_focus_exact.py` is the bit-exact gate (final vs base, all 22 must match).
- Roofline gate: as in Perf 8-10, measured rates against the LoFi FPU floor (16 cycles per tile-MAC) and the NoC link rate.
  - At the floor: res5.x's walk on 80 cores (it only moves with more cores) and aspp.convs.1's walk.
  - Far from it: HEIGHT round 0 (2 us fixed + ~26-38 GB/s), the BLOCK tap shift (~60 ns per tap against a ~33 ns pack floor), stem.conv1's slab wait.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel on `039e18f6c81`: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only" stubs the matmul alone. All 22 cases in `ablation_r11/peel_results.txt`; zones in `zones_base.txt`, `zones_dm_all.txt`. stem.conv1's slab-direct matmul has no stub.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only |
|---|---|---|---|---|---|---|---|---|---|
| stem.conv2 | 27.8 | 27.8 | 27.1 | 25.7 | 25.7 | 25.7 | 25.7 | 19.0 | 19.0 |
| res2 | 10.4 | 10.5 | 10.4 | 10.2 | 10.1 | 10.1 | 10.1 | 6.7 | 8.0 |
| res3.0 | 17.8 | 17.8 | 16.0 | 16.2 | 16.2 | 12.4 | 9.9 | 5.1 | 14.2 |
| res3.x | 10.9 | 10.9 | 10.5 | 9.5 | 9.6 | 9.5 | 9.6 | 7.1 | 9.4 |
| res4.0 | 16.7 | 14.8 | 13.8 | 13.1 | 12.6 | 11.6 | 11.0 | 8.2 | 16.4 |
| res4.x | 13.4 | 13.1 | 12.5 | 12.4 | 11.5 | 11.5 | 10.3 | 7.4 | 12.8 |
| res5.0 | 34.0 | 34.0 | 32.9 | 33.7 | 33.8 | 33.7 | 30.7 | 12.6 | 23.5 |
| aspp.convs.1 | 52.3 | 51.9 | 48.7 | 48.8 | 48.6 | 47.6 | 42.2 | 12.8 | 36.2 |
| aspp.convs.2 | 46.3 | 45.8 | 42.7 | 42.8 | 42.5 | 42.5 | 37.0 | 13.2 | 33.2 |
| sem dec res3 fc0 | 36.5 | 36.6 | 35.7 | 33.2 | 33.1 | 33.2 | 33.1 | 13.2 | 20.2 |
| inst dec res3 fc0 | 21.2 | 20.8 | 20.0 | 17.7 | 17.6 | 17.7 | 17.8 | 7.0 | 12.0 |
| center_head.1 | 10.1 | 10.1 | 9.4 | 9.7 | 9.7 | 9.7 | 9.6 | 5.5 | 6.2 |

Critical-core timelines (`ablation_r8/timeline.py`) and the ranked gap x calls (unzoned base, us per inference):
1. **res4.x (32.1).** TRISC-busy end to end on 80 of 110 cores: prelude 0.8 -> 4.0, walk 4.1 -> 11.4 (FPU floor 6.0 us), combine 11.4 -> 13.0. Everything stubbed still 7.4 us (ring skeleton).
2. **res5.x (30.4, three layers).** Balanced: matmul stubbed 23.5, all DM stubbed 30.7, full 34.0, all stubbed 12.6 (16 owner slices at ~0.43 us per ring step). Only fewer rows per core shrinks both halves.
3. **res3.x (19.5), res2 (13.0), center_head.1 (8.0), decoder res3 layers (20.6).** Round 0 lands 2.3 us after start on res2 (13 KB), 3.1 on res3.x (52 KB), 5.3 on the decoder res3 layers (130 KB); the final pass takes 1.1-3.2 us at the tail; the fill stub is worth 0.4-1.0 us. res3.x's FPU time alone (5.1 us on 86 cores) is near its 5.67 us target; at tile-row granularity 110 cores do not lower its rows per core (256 / 110 -> 3).
4. **aspp.convs.1 (12.2).** Its carve-out kept the padded-slab untilize + tilize prelude (~15 us) before a 36 us FPU-floor walk.
5. **res3.0 (11.8), res4.0 (10.0).** Stride 2: res3.0 is TRISC-bound (untilize 6.2 us, then tilize 3.6 + matmul 6.4 us inside the walk).
6. **stem.conv1 (10.1).** The slowest core waits 5.0 us for its whole slab before a 5.5 us matmul (slab wait p50 2.3 us).

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Numbers are unzoned medians of 3 alternating pairs unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `kwf_xchg_slab_skip` (user-requested): the kw-fold fill reads only the slab rows compute reads; the halo rows the exchange replaces (or an image-row-edge boundary makes unnecessary) are skipped | WIN | res3.x 10.48 -> 10.12, inst dec res3 fc0 20.71 -> 19.98, sem dec res3 fc0 36.09 -> 35.26 | Dead rows: res3 geometry 30.8%, res2 geometry 7.2% (2 of 28 rows, not the 4 the brief counted), stem.conv2 2.8%, stem.conv3 0.24% (2 M-blocks, recompute). Fill ends 0.3-0.7 us earlier on the res3 layers; round 0 does not land sooner; the gain shows downstream (bias-row decode, last weight wait). Costs: stem.conv2 +0.7%, res2 +0.9% (the plan loop's instruction time on a fill-critical reader; a control with the loop and nothing skipped costs the same). Rejected options: per-batch dead-row check (+3.2% stem.conv2), precomputed segment split (~0.33 us of planning at fill start). Found and fixed a HEAD deadlock: on a multi-M-block kw-fold core the writer's early fill spun on a later block's batch. Bit-identical, poison-checked. |
| `stem1_slab_stream`: round 0 issued before the fill (receivers ack at start); the slab filled in 1024-pixel chunks under transaction ids, the reader pushing the landed prefix, compute waiting per subblock row | WIN | stem.conv1 11076 -> 10196 ns | Early round 0 alone 11036, stream alone 11056, both 10196: each removes the other's bound. HiFi4 stem.conv1 -7.8%, no-bias variants -12.8%, smaller stride-2 RM shapes -4% to -8%. Measured regressions earn the carve-out: streaming blocks under 3 subblock rows (forced: +28-31%, +6-9%, +3.5%). The slow cores' fill comes almost all from higher shards over the writer's NoC1 (the 108-core split drifts past the 110 input shards); streaming hides it. Bit-identical. |
| `block_spare_mgroups`: res4 / res5 on the idle columns: 13 M-groups (3 transposed into spare columns 8-10, channel slice k on the spare column's row k, ring down the column), weights forwarded from each column stream | WIN | res4.x 12098 -> 11607, res5.0 33368 -> 28524, res5.1 32741 -> 28131, res5.2 32436 -> 26806 | Mechanism alone did not pay; four measured additions made it: (1) a forwarder (the column participant after the sender pushes its round, then multicasts it to its slice's spare members) — the column sender multicasting twice was +9% on res4.x, one DRAM stream per spare row +6-7% on res5.1/5.2 (same pages as the column sender); (2) fixed group heights with a 1 x N subblock for the short row of a 5-row group (runtime heights: flat); (3) spare groups take the cheapest M ranges (-0.7 to -1.4 us); (4) the N-slice bias read without blocking after round 0 issues (-3.3% on res4.x with the spare groups, flat alone). Fixed a latent ring-credit race (a core with no incoming partial could free its slots before the writer checked). HiFi4: res4 -25%, res4.0 -20.5%, res5.0 -22.5%. Inexpressible (fallback to grid rows): no spare columns (aspp), more than one M-block per core. Bit-identical. |
| `tap_shift_rate`: each input tile unpacked once into a pixel-contiguous DEST row buffer; the packer reads each tap as a shifted DEST window | WIN | res4.x 12079 -> 11379, aspp.convs.2 45326 -> 42095, aspp.convs.3 21213 -> 20689, res5.1 32725 -> 32394 | Per tap (isolated, d 1..18): ~58-60 ns -> 34-42 ns. HEAD's cost split: pack 33-37 ns (the floor), ~60-90 cycles per tile load, 64 one-row MOVB2D per tap. On the Perf 10 carve-out: aspp.convs.1 51360 -> 48102 (HEAD's shift there still +2.4%), res5.0 33339 -> 32512, so the carve-out is deleted. Rejected: 16-bit DEST (exact, bf8_b pack slower), unpacker SrcB row offset (4-row MOVB2D across a 16-row SrcB boundary reads wrong rows). Bit-identical, bf16 input included (a hard-coded 16 B row stride gave PCC 0.41 on bf16 and was fixed). |
| `kwf_final_pass`: cut the kw-fold final pass (tilize ACC + bias + ReLU) | NULL | res2 10.28 -> 10.10 (-1.8%), every other kw-fold case -0.5% to 0.0% | Split (zoned, critical cores): drain 0.2 us (real walk packs), setup 0.45 us (0.18 of it in `pack_reconfig_data_format`), per tile ~50 cycles, waits for the halo C strip and bias rows ~0.03-0.05 us. Plane-major unpack runs took unpack from ~46 to ~34 cycles per tile, but the pack then bounds at ~47 cycles per tile: the BH non-8-bit tilize workaround (a 32-PACR unswizzle per tile). Moving final tiles into the walk adds ~38 cycles of unpack to the walk's bound thread to save ~47 of pack. Not graduated: an in-noise gain for a new raw copy of `reconfig_packer_data_format`. |
| `kwf_round0_noc_split`: keep round 0's multicast NoC free of fill reads (or out of its time window) | NULL | res2 10.20 -> 9.96 with the deferred tail fill, center_head.1 +3.1% with it; the other three variants regress (up to +72% on center_head.1) | With no fill at all the server multicast still ran 43-53 GB/s; the fill costs round 0 at most ~0.15-0.4 us. Server NoCs swapped: big rounds read slower on NoC1 (sem dec res3 fc0 3.2 -> 4.1 us). Reader-only fill: the higher-shard reads go the long way round. Ack after the NoC1 fill: the server waits for the slowest of 86-103 receivers. |

Not floated:
- **A K split over all 11 columns for res4.x.** Three extra ring hops (~0.4 us each) against a 22% MAC cut; spare-column M-groups give fewer rows per core without lengthening the ring.
- **Stride-2 even / odd decimation for res3.0** (kw-fold-like, building E / O tiles on the FPU): one call per inference, a T3 restructure.
- **2x2 kw-fold subblocks** (Perf 7: much slower); **res4.x ring tail** (no mechanism beyond Perf 6's measured re-init cost).

### Graduated (one path each, in commit order)
1. **Tap shift from a DEST row buffer** (`ca44fde79ab`). Clean apply. The Perf 10 carve-out (`TAP_SHIFT_MIN_HALO_PER_TILE`) and the per-row shift are deleted.
2. **kw-fold live-row fill** (`bf6cdcde04e`). Clean apply, no new zone.
3. **stem.conv1 streaming** (`1e9aafe4133`). Merge change: the experiment's `SD_EARLY_WEIGHTS` host switch is removed and the early round 0 is unconditional under slab-direct (the old round-0 timing is no longer selectable).
4. **Spare-column M-groups** (`cc1b40769bd`), with the optimizer's unit-test patch: three grid-row structure tests pin `SPARE_MGROUPS = False` to keep covering the grid-row fallback, and `test_conv2d_3x3_block_sharded_spare_groups` (res4, res5.0, a 2-slice d 8 shape, COL_MAJOR, program-cache re-dispatch) is new. Merge change: the experiment's `NSLICE_ASYNC_BIAS` host switch is removed; the async bias is unconditional on the N-slice walk with a bias.
- Experiment files: a global `import torch` in `block_spare_mgroups/test_block_sharded_grad.py` failed the repo's import check and was moved into the functions. Run logs and profiler CSVs over 500 KB stay on disk, uncommitted (large-file hook): 8 in `block_spare_mgroups/runs/`, 19 in `kwf_final_pass/runs/` and `kwf_round0_noc_split/`.

Carve-outs, each with the measurement that earned it:
- **stem streaming: blocks under 3 subblock rows keep one slab push** (`SD_STREAM_MIN_SUBBLOCK_ROWS`). Forced streaming there: 64x64x8 -> 32 +28-31%, 32x34x16 -> 64 +6-9%, 64x64x16 -> 64 +3.5%. Mechanism: the first matmul call already needs nearly the whole slab.
- **Inexpressible, not guards:** spare groups without spare columns or with more than one M-block per core (grid rows only); slab-direct at W_out % 32 != 0 (gather path).
- **Not carved out: the slab skip on stem.conv2 (+0.18 us, +0.66%).** Bisected to `bf6cdcde04e` in 3 passes (27.27 -> 27.45 us). The fill loop is the single path; a second loop for one call per inference is not worth the debt.

### Precision
- **All 22 focus cases are bit-identical to `039e18f6c81`** (`ablation_r11/merge_gates/test_focus_exact.py`, one process, 0 differing elements on every case, stem.conv1 included).
- Every graduated change is data movement, scheduling or partition only; no change touches the precision contract.

### Whole-op result (unzoned, us; whole perf file, base `039e18f6c81` and final `cc1b40769bd`, mean of 3 alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 21.20 | 20.85 | 1.02x | 22.7 | 75% -> 76% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 84.23 | 83.62 | 1.01x | 90.8 | 75% -> 76% |
| 3 | res4.{1..5}.conv2 | 12.09 | 11.66 | 1.04x | 5.7 | 33% -> 34% |
| 4 | res3.{1,2,3}.conv2 | 10.42 | 10.08 | 1.03x | 5.7 | 38% -> 39% |
| 5 | stem.conv2 | 27.32 | 28.12 | 0.97x | 22.7 | 58% -> 57% |
| 6 | aspp.convs.2 | 45.38 | 42.13 | 1.08x | 45.4 | 70% -> **75%** |
| 7 | aspp.convs.1 | 51.50 | 47.82 | 1.08x | 45.4 | 62% -> 66% |
| 8 | res2.{0,1,2}.conv2 | 10.13 | 10.38 | 0.98x | 5.7 | 39% -> 38% |
| 9 | center_head.1 + offset_head.1 | 9.68 | 9.63 | 1.00x | 5.7 | 41% -> 41% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 24.34 | 24.31 | 1.00x | 22.7 | 65% -> 65% |
| 11 | stem.conv3 | 47.80 | 47.82 | 1.00x | 45.4 | 66% -> 66% |
| 12 | sem dec res2.fuse_conv.0 | 93.95 | 93.31 | 1.01x | 102.1 | 76% -> 77% |
| 13 | sem dec res3.fuse_conv.0 | 35.95 | 35.26 | 1.02x | 28.4 | 55% -> 56% |
| 14 | sem dec res3.fuse_conv.1 | 29.43 | 29.14 | 1.01x | 22.7 | 54% -> 55% |
| 15 | stem.conv1 | 10.48 | 10.25 | 1.02x | 1.1 | 7% -> 7% |
| 16 | inst dec res3.fuse_conv.0 | 20.52 | 20.37 | 1.01x | 14.2 | 48% -> 49% |
| 17 | inst dec res2.fuse_conv.0 | 29.33 | 29.32 | 1.00x | 28.4 | 68% -> 68% |
| 18 | res5.2.conv2 | 32.35 | 26.49 | 1.22x | 22.7 | 49% -> **60%** |
| 19 | res5.1.conv2 | 32.73 | 27.15 | 1.21x | 22.7 | 49% -> **59%** |
| 20 | res5.0.conv2 | 33.49 | 27.41 | 1.22x | 22.7 | 47% -> **58%** |
| 21 | res3.0.conv2 | 17.46 | 17.44 | 1.00x | 5.7 | 23% -> 23% |
| 22 | res4.0.conv2 | 15.53 | 14.98 | 1.04x | 5.7 | 26% -> 27% |

All 3x3 layers of one Panoptic inference: 1182 -> 1143 us (-3.3%). At or above 70%: aspp.convs.3, sem res2 fc1, sem dec res2 fc0, aspp.convs.2 (new).

Context matters for three cells (`ablation_r11/focus_only_ab.txt`, `bisect_runs.txt`; focus file alone, 3 alternating pairs):
- **stem.conv1** runs first in the focus-only process: 11.05 -> 10.26 (-7.2%). Inside the whole file: 10.48 -> 10.25 (-2.2%). HiFi4 stem_conv1 in the whole file: -8.8%.
- **stem.conv2**: focus-only 27.25 -> 27.47 (+0.8%: +0.18 us from the slab skip, the rest within noise). In the whole file +2.9%: a further +0.35-0.5 us appears at `cc1b40769bd` (spare groups), whose code never executes on stem.conv2. stem.conv2 alone at that commit vs its parent: 27.58 vs 27.60 / 27.61; right after the BLOCK perf tests: +0.12 us; only after the whole file: +0.4 us. No carve-out is possible or earned: a context effect, not a path this change runs.
- **res2**: focus-only 10.13 -> 10.30 (+1.6%, non-overlapping). Per commit (3 passes) every step is within the ±0.15 us run-to-run spread; the slab skip's own A/B measured +0.9%.
- Focus-only total: 1183.5 -> 1143.2 us (-3.4%).

### Guard set (whole perf file: 41 shapes plus the 22 focus cases, unzoned, 3 alternating pairs; `ablation_r11/ab_results.txt`)
- **BLOCK (HiFi4 unless noted).** res4 -25.8%, res5_0 -23.5%, res4_0 -20.3%, res5_2 -20.3%, res5_1 -16.4%, res4 LoFi -4.0%, aspp_convs1 -2.3%, aspp_convs2 -2.3%, aspp_convs3 -1.3%.
- **HEIGHT (HiFi4).** stem_conv1 -8.8%, res3 -1.4%, inst_res3_fuse_conv0 -0.9%, sem_res3_fuse_conv0 -0.6%, sem_res3_fuse_conv1 -0.4%, sem_res2_fuse_conv1 -0.3%, sem_res2_fuse_conv0 -0.2%, res2 / inst_res2_fuse_conv1 -0.1%, inst_res2_fuse_conv0 / res3_0 / stem_conv3 0.0%, center_head1 / stem_conv2 +0.3%.
- **Interleaved (18 shapes).** -0.4% to +1.1% (H16_W16_Ci64 rm_bf16 +1.1%, H32_W64_Ci128 rm_bf16 +0.8%). No change of this round runs on the interleaved path (no padded slab, kw-fold, slab-direct or N-slice walk there): the same layout-noise cells as Perf 8-10, no carve-out.
- **Golden.** `eval/golden_tests/conv2d_3x3/`: 1518 passed, 480 skipped (INVALID), 0 failed, on the final tree.
- **Unit dir.** 299 / 299 (the 8 new spare-group tests included). Each graduation was gated on its own files first: focus + kw-fold + BLOCK 97/97 (1-2), focus + HEIGHT + layouts + interleaved 151/151 (3), focus + BLOCK 80/80 (4).
- **`--dev`.** BLOCK file + focus 80/80 after (4); kw-fold, HEIGHT and layouts files 135/135 on the final tree.

### Bottleneck now
- **res4.x (34%).** The critical core is a spare-column member: the walk is paced by the column weight stream at ~1 us per round (9 one-tile DRAM read issues ~0.5 us plus the multicast ~0.35 us, serial on the sender's writer), and spare members get each round one forward later. A faster column stream is the next lever.
- **res5.x (58-60%).** The critical core is a spare member: its first weight round lands ~5.2 us in (sender, then forward), and its fill (8 members through one column's links) ends ~4.75 us in vs ~1.9 on a grid core. Stream and compute per step are now level (~2.4 us).
- **Small HEIGHT kw-fold layers (res3.x 39%, res2 38%, center_head.1 41%).** Round 0 (2-3 us), the final pass (~1-1.3 us, pack-bound at ~47 cycles per tile) and res3.x's FPU floor on 86 cores. At tile-row granularity 110 cores do not lower rows per core; an N- or K-split across cores is the remaining lever.
- **stem.conv1 (7%).** Bound by the weight round (~3.5 us: ~0.5 us of DRAM page-read issue, ~1.2 us DRAM, ~0.8-1.1 us multicast beside the fill); no core waits on its slab after the weights land.
- **Stride 2 (res3.0 23%, res4.0 27%).** TRISC-bound untilize + tilize around the matmul; an even / odd FPU decimation (kw-fold for stride 2) is the unfloated candidate.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` / `untilize`, `pack_tile` | capability | A tap tile is a 32-row window of an input image row starting at pixel 32 c +- d, spanning two input tiles. No helper packs a tile from a DEST window that is not a whole DEST tile (here: rows from an arbitrary 4-aligned or 2-aligned DEST row, Z stepping two channel-half blocks), none unpacks a whole tile into one Src bank outside the matmul, and none places rows at a DEST row offset. The raw path is one full-tile UNPACR per input tile, MOVB2D into a pixel-contiguous DEST buffer, and PACR with patched Y / Z / W strides. | 51360 (aspp.convs.1 on the padded-slab untilize + fast-tilize prelude, base) | 48102 | conv2d_3x3_tap_shift.hpp:41 (justification), :305 (`run`), conv2d_3x3_compute.cpp:1235 |
| `dataflow_kernel_lib` reads with transaction ids | capability | No chunked streaming read: nothing issues a window of chunks under rotating transaction ids, retires them in order with a non-blocking "is this id flushed" query, and publishes a landed-prefix count (here to the slab CB's single producer). Same missing non-blocking query as the Perf 9 bias row. | 11036 (stem.conv1, early round 0 with one slab push per block) | 10196 | conv2d_3x3_slab_stream.hpp:101, conv2d_3x3_writer.cpp:554-563, :1031 |
| `dataflow_kernel_lib` reads with transaction ids | capability | The N-slice async bias needs the same non-blocking per-id flushed query, polled inside the ring-credit and weight waits. | 12022 (res4.x, spare groups with the blocking bias) | 11607 | conv2d_3x3_writer.cpp:1244 (`issue_async_bias`), :1262 |
| `ReceiverPipe::receive` (multicast) | capability | No polling form (the Perf 9 bypass). It is now generalized to a second channel (`weight_mcast2`), whose sender is the forwarding column participant; the forward itself uses the helper's `send`. | not re-measured (existing Perf 9 bypass, extended to the forward channel) | — | conv2d_3x3_writer.cpp:1292 (`receive_polling_on`) |

The slab skip and the spare groups' host-side partition use no raw LLK. The 1 x N short-group subblock lives inside the existing raw ring walk (`compute_kernel_lib::matmul_block` takes one subblock shape per call; Perf 8 bypass).

## Perf 12
- Date: 2026-11-19
- Scope: perf tournament (one round, run round 1 of 4) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores. No SUPPORTED change. Base `739a271172e` (end of Perf 11). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated (cap 6). 3 graduated as WINs, 1 graduated on the merged tree after reading NULL alone, 1 NULL contributed a latent-race fix (graduated), 1 REGRESSION.
  - All 3x3 layers of one inference: 1143 -> 1119 us (-2.1%; whole perf file, 3 alternating pairs).
  - res3.0 -23% (17.49 -> 13.48 us), res4.0 -17% (14.96 -> 12.43), res4.x -7.2% (11.64 -> 10.80), center_head.1 -6.5%, inst dec res3 fc0 -6.7%, res2 -5.4%.
  - All 22 focus cases are bit-identical to the base, stem.conv1 included.
  - No case at or above 70% moved below it; none crossed 70% this round.
- Instrumentation: the Perf 1-11 zones are kept. New zone: `compute_kwf_chunk_wait` (round-0 kernel-row waits). The stride-2 tap build runs inside `compute_tilize` (`compute_untilize` is empty on that path, as on the tap-shift path).
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r12/`.
  - `ablation_hooks.patch` is the round-11 patch re-cut for `739a271172e`.
  - It adds two stubs the round-11 patch lacked. The kw-fold reader fill was rewritten in Perf 11 after that round's peel, and the re-cut hook missed it: this round's first peel pass left it running under `fill` (re-run with the stub; the kw-fold rows below use it). The spare-column forward now stops under `weights`.
- Zones inflate skeletons: res4.x with everything stubbed reads 7.7 us zoned and 5.99 us unzoned (two reader zones per ring group). Wins below are unzoned.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel on `739a271172e`: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only" stubs the matmul alone. Kw-fold rows use the corrected fill stub. stem.conv1's slab-direct matmul has no stub. All 22 cases are in `ablation_r12/peel_results.txt`.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only |
|---|---|---|---|---|---|---|---|---|---|
| stem.conv2 | 28.0 | 28.0 | 27.2 | 25.9 | 26.0 | 26.0 | 26.0 | 19.0 | 19.1 |
| res2 | 10.6 | 10.7 | 10.4 | 10.5 | 10.4 | 10.4 | 10.4 | 6.7 | 8.1 |
| res3.0 | 17.8 | 17.7 | 15.9 | 16.2 | 16.2 | 12.5 | 10.0 | 5.1 | 14.2 |
| res3.x | 10.6 | 10.6 | 10.5 | 9.4 | 9.4 | 9.5 | 9.6 | 7.1 | 9.4 |
| res4.0 | 16.2 | 14.6 | 12.5 | 12.5 | 11.2 | 10.3 | 9.8 | 7.9 | 15.7 |
| res4.x | 12.4 | 12.3 | 12.3 | 10.9 | 9.8 | 9.7 | 9.2 | 7.7 | 11.5 |
| res5.0 | 27.7 | 27.6 | 27.7 | 24.8 | 24.8 | 24.8 | 23.4 | 9.7 | 24.1 |
| aspp.convs.1 | 48.2 | 48.4 | 45.4 | 45.3 | 45.1 | 45.1 | 41.5 | 12.0 | 35.8 |
| aspp.convs.3 | 21.5 | 21.5 | 21.3 | 19.0 | 18.8 | 18.8 | 18.6 | 11.7 | 18.6 |
| sem dec res3 fc0 | 35.9 | 35.9 | 35.7 | 33.1 | 33.0 | 32.9 | 33.0 | 13.1 | 19.3 |
| center_head.1 | 10.0 | 10.0 | 9.4 | 9.8 | 9.7 | 9.7 | 9.7 | 5.5 | 6.3 |

Single-stage stubs (fill alone / weights alone): res2 10.4 / 10.9, res3.x 10.2 / 9.6, res4.0 13.7 / 15.8, res4.x 12.3 / 11.8, aspp.convs.1 45.4 / 48.4, center_head.1 9.4 / 10.4, stem.conv1 10.5 / 9.6.

Ranked (gap to target x calls, us per inference): res4.x 34, decoder res3 layers 21 (three layers), res3.x 20, res2 15, res5.x 15 (three layers), res3.0 12, res4.0 11, stem.conv1 9, center_head.1 9.
1. **res4.x.** Every core, the column sender included, finished a weight round every ~0.95-1.05 us. Round 0 landed 2.2-3.1 us after kernel start, against ~0.53 us of FPU work per ring step.
2. **res3.0 / res4.0 (stride 2).** res3.0 is TRISC-serial: fill, untilize 3.7 us, tilize 2.5 us, matmul 4.9 us.
3. **Kw-fold start-up.** The first subblock starts 2.3 us after kernel start on res2, 2.1 on center_head.1 and 2.8 on res3.x.
   - On res2 and center_head.1 the reader fill zone is still 2.3 us with its reads and the weights stubbed: control-bound.
   - On res3.x the 52 KB round 0 is the gate.
4. **86-core kw-fold layers.** 256 tile-rows give 3 rows per core on 86 or on 110 cores. The decoder res3 layers are matmul-dominant (mm only: sem dec res3 fc0 35.9 -> 19.3).
5. **aspp.convs.1.** Fill-bound start (fill alone -2.8 us).
6. **Kw-fold packer.** With the matmul stubbed, stem.conv2's packer still runs 14.9 us of its walk (fp32 L1-accumulated Q packs). The full walk is unpack-bound at ~20 cycles per tile-MAC. No idea was spent here (below).

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Numbers are unzoned medians of 3 alternating pairs unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `stride2_dest_taps`: build the stride-2 taps from the input's TILE slab in DEST. 4-row MOVB2D into a pixel-contiguous buffer plus one-row moves of the odd pixels into an O plane. The packer reads kw = 1 with interface mask 0101 (every other row), kw = 2 / kw = 0 as 4-row O-plane windows (kw = 0 one row earlier). No untilize, RM slab, window gather or strided tilize | WIN | res3.0 17.45 -> 13.45, res4.0 15.05 -> 12.60 (medians of 4) | Isolated, ns per tap tile: replaced path ~57; pixel-buffer masks only 47.3; O plane with direct PACRs 43.3; E + O planes 46.1; O plane with replayed PACRs 40.4 (graduated). Domain cells -6% to -17% (HiFi4 res3.0 -11.9%, res4.0 -8.6%, BLOCK 32x64x128 -16.4%, forced 2 / 4 M-blocks -14.5% / -10.3%). Fill batches of 4 rows beat 16 (14.60 vs 15.52 us zoned). Inexpressible (old path): W % 64 != 0, ROW_MAJOR input, K-blocks under one kernel row. Bit-identical. |
| `block_wstream_rate`: the BLOCK column weight stream at the multicast's rate | WIN | res4.x 11604 -> 10862 (with the ring data stubbed 11556 -> 10725) | The ~1 us round, measured: sender 0.40 us issuing 9 one-tile reads, 0.06 read wait, 0.15-0.22 ack wait, 0.34 blocking 9.8 KB send. The forwarder's forward (0.33-0.40 us) sat between two column multicasts, because it acked the next column round only after forwarding. Three parts, all needed: one DRAM shard per round (one read), the forwarder acking ahead, non-blocking multicasts. The ring credit is polled every round: without that a forwarder that never waits gave its credit ~2.5 us late (13.3 us zoned). Rejected: early ack + non-blocking without page runs (res4.x 10860, but aspp.convs.3 +3.1%); per-slot flags with a summed ack counter (10801, **incorrect**: wrong output on aspp.convs.2 HiFi4 with 2 M-blocks per core); spare members reading round 0 from DRAM (+2% to +6.7%); prefetching every free slot (+8.6% alone). Domain: res5.2 -4.5%, res5.1 -1.3%, rest flat. Bit-identical. |
| `kwf_first_subblock`: start the kw-fold walk sooner | WIN | res2 9.98 -> 9.78, res3.x 10.01 -> 9.82 | Two parts. (1) The fill is planned on the host: each RISC gets (owner x / y, offset, slab row, rows) groups of up to 8 rows. The device plan's per-run TensorAccessor lookups cost ~0.1-0.2 us of RISC time each and made the fill control-bound. The writer takes the slab tail, balanced against the reader. The device-side plan is deleted. (2) Round 0 streams by kernel row: each server chunk is one kernel row, with an early flag per chunk. First subblock: res2 2.8 -> 2.2 us, res3.x 3.3 -> 2.15, center_head.1 ~2.9 -> 2.35. Domain: inst dec res3 fc0 -6.2%, center_head.1 -4.7%, sem dec res3 fc0 -3.6%, rest -1.6% to +0.4%. res2's base swings ~0.4 us between sessions (one session read +0.9%). Rejected: the server reading ahead (res2 +5%), taller DRAM weight shards (res3.x +5%: a chunk in one bank). Poison-checked, bit-identical. |
| `block_ring_chain`: cut the N-slice ring's per-step chain | NULL alone, WIN merged | res4.x 11546 -> 11579 on the base; with weights stubbed 10653 -> 10436 | The chain, measured: pack onto the incoming partial 0.30-0.35 us; packer push to reader 0.13-0.19; write to ack 0.33. The step's matmul was always done before the partial arrived. Change: the packer polls the arrival semaphore itself (compute reads the semaphore base from the launch message), the reader only forwards, and the arrival count is written right behind each group's data (same command buffer, static VC, transaction id), so nothing waits on the write's ack. Everything stubbed, unzoned: 5990 -> 4964 ns. Synthetic chain-bound ring (11 columns, 5 owners): 13905 -> 11036 (-20.6%). On the weight-stream tree (below) it moves res4.x. Bit-identical. |
| `kwf_n_planes`: the 86-core kw-fold layers on ~104 cores, C_out split into planes | NULL | best (shared stream, annealed layout table, 6 weight slots): sem dec res3 fc0 +1.0%, fc1 +1.8% | Compute ceiling with fill and weights stubbed: N = 8 layers -10%, but N = 4 layers +6% / +16% (a 2-tile n-group runs Q_1 as 4x1 at ~31 cycles per MAC). Each input row goes to two remote cores: 2.1x the remote row reads, and per-plane placement routes them the long way round NoC0 (fill zone 2 -> 8-23 us). Per-plane weight rectangles force a bad fill layout and wrap the NoC1 multicast. The shared stream's round 0 lands ~1.5 us late and plane 1 runs a round behind, cancelling ~5.4 us of compute. **Found a latent HEAD race** (graduated, below). |
| `block_halo_mcast`: the round's reuse-shared broadcast. Shard owners push their rows down / up the column as multicasts (ring slot = tile-row mod S, so every receiver's address is common) instead of each core pulling overlapping halo rows | REGRESSION | aspp.convs.1 47786 -> 48950 (best of 20 variants); aspp.convs.2 +3.7% | The push halves the busiest column link's bytes (300-331 -> 134-172 KB) and the fill ends no later. But each row is needed first by one core and last by another, and a push delivers it to all at once. The pull already brought each core its first-needed rows within ~1.5 us. The tap-shift prelude ends 1-2 us later. Overlapping column multicasts stalled senders 0.7-11 us. The two fastest variants skip path reservation (deadlock risk); with it, ~+7.5%. Inexpressible: spare-column groups (res4.x, res5.x), stride 2. Side finding: each column's weight-sender row reads its higher shards on NoC0 the long way round (+57 KB per link of that column). |

Not floated (cap 6, measured reason each):
- **kw-fold shift-add in DEST (SFPU), one bf8_b pack per output tile.** It removes the fp32 Q packs, but the SFPU shares the math thread with the FPU. Per output tile that is ~480 / 768 cycles (K' = 6 / 12) against today's pack-bound ~320 / FPU-bound ~576.
- **kw-fold final-pass rework.** Perf 11 NULL (pack-bound at ~47 cycles per tile).
- **res4.x walk streamed behind the prelude.** TRISC-busy; Perf 4 `nslice_stream_first_slice` NULL.

### Graduated (one path each, in commit order)
1. **Stride-2 taps from DEST** (`d078da6e0b1`). Clean apply.
   - The old prelude stays only where the new one cannot be expressed: W % 64 != 0, ROW_MAJOR input, K-blocks under one kernel row, or the new CBs not fitting the L1 the RM slab and unit ring give back.
   - New unit file `test_conv2d_3x3_stride2_taps.py` (8 tests, exact equality against `S2_TAPS = False`).
2. **BLOCK weight stream** (`21ad39e3b36`). Clean apply, with the optimizer's 3 placement tests in `test_conv2d_3x3_block_sharded.py`.
   - `prepare_conv2d_weights` places BLOCK weights one ND shard per N-slice round. Page order is unchanged, so every other reader reads the same tiles.
   - C_out that does not split into whole N-slices stays DRAM interleaved (the shard shape cannot be expressed), and so does weight placed any other way; the writer keys off the actual shard shape.
3. **Host-planned kw-fold fill + round-0 kernel-row stream** (`69666a8f76f`). Two descriptor define hunks merged by hand onto (2)'s writer defines. The device-side `kwf_slab_plan` and its helpers are deleted.
4. **Weight-server slot acks per slot** (`db891506c54`, the race fix from `kwf_n_planes`).
   - With one shared counter, a receiver two acks ahead (the ragged short core is enough) can ack a later slot's reuse before a slower receiver acked an earlier one. The server then overwrites that slot under the slower core's matmul.
   - Now one counter per slot. Two descriptor hunks merged by hand onto (3)'s round-0 semaphores.
   - Taken where servers run with more rounds than slots and the semaphores fit (16 per program). No focus or unit case falls back (max 13 used, instrumented once on the merged tree).
   - Perf flat: sem dec res3 fc0 35153 vs 35161 ns.
5. **Ring arrival count** (`401e2e0ca27`). Two descriptor define hunks merged by hand. Graduated on the merged tree, where it measured (unzoned, 3 alternating pairs, base = after (4)):
   - res4.x 10.79 -> 10.62 (-1.6%, ranges 10.75-10.81 vs 10.59-10.63);
   - BLOCK HiFi4 res4 @ LoFi 10.93 -> 10.65 (-2.6%), res4.0 -1.4%;
   - res5.x -0.4% to -0.7%, aspp within +-0.8% (overlapping ranges).
- Experiment files: global `import torch` in `block_wstream_rate/` and `block_ring_chain/test_block_sharded_grad.py` moved into the functions (repo import check). Run logs and profiler CSVs over 500 KB stay on disk, uncommitted (large-file hook).

Carve-outs, each with the reason that earned it:
- **None earned by a measured regression.**
- **Inexpressible / feasibility only:**
  - stride-2 taps: W % 64, ROW_MAJOR input, K-blocks under one kernel row, L1 fit;
  - BLOCK round shards: C_out not a whole number of N-slices;
  - per-slot acks: semaphore budget;
  - round-0 streaming: only under weight servers with more than one chunk (one-burst rounds land together: nothing to stream).
- **Not carved out: sem res2 fc1 +0.16%** (83.61 -> 83.74 us, ranges 83.54-83.73 vs 83.74-83.74). It is below the run-to-run spread; its own optimizer measured it flat (+0.4%), and the race fix -0.2% there.

### Precision
- **All 22 focus cases are bit-identical to `739a271172e`** (`ablation_r12/merge_gates/test_focus_exact.py`, one process, 0 differing elements on every case).
- No change touches the precision contract. The stride-2 build moves bf8_b data through fp32 DEST without arithmetic. The ring change only reorders signals. The rest is data movement and scheduling.

### Whole-op result (unzoned, us; whole perf file, base `739a271172e` and final `401e2e0ca27`, mean of 3 alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 20.93 | 20.40 | 1.03x | 22.7 | 76% -> 78% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 83.61 | 83.74 | 1.00x | 90.8 | 76% -> 76% |
| 3 | res4.{1..5}.conv2 | 11.64 | 10.80 | 1.08x | 5.7 | 34% -> 37% |
| 4 | res3.{1,2,3}.conv2 | 10.08 | 9.80 | 1.03x | 5.7 | 39% -> 41% |
| 5 | stem.conv2 | 28.04 | 27.25 | 1.03x | 22.7 | 57% -> 58% |
| 6 | aspp.convs.2 | 42.10 | 42.19 | 1.00x | 45.4 | 75% -> 75% |
| 7 | aspp.convs.1 | 47.81 | 47.93 | 1.00x | 45.4 | 66% -> 66% |
| 8 | res2.{0,1,2}.conv2 | 10.35 | 9.79 | 1.06x | 5.7 | 38% -> 41% |
| 9 | center_head.1 + offset_head.1 | 9.69 | 9.06 | 1.07x | 5.7 | 41% -> 44% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 24.33 | 23.91 | 1.02x | 22.7 | 65% -> 66% |
| 11 | stem.conv3 | 47.84 | 47.72 | 1.00x | 45.4 | 66% -> 67% |
| 12 | sem dec res2.fuse_conv.0 | 93.17 | 92.74 | 1.00x | 102.1 | 77% -> 77% |
| 13 | sem dec res3.fuse_conv.0 | 35.45 | 34.24 | 1.04x | 28.4 | 56% -> 58% |
| 14 | sem dec res3.fuse_conv.1 | 29.14 | 28.58 | 1.02x | 22.7 | 55% -> 56% |
| 15 | stem.conv1 | 10.22 | 10.24 | 1.00x | 1.1 | 7% -> 7% |
| 16 | inst dec res3.fuse_conv.0 | 20.29 | 18.93 | 1.07x | 14.2 | 49% -> 52% |
| 17 | inst dec res2.fuse_conv.0 | 29.29 | 28.92 | 1.01x | 28.4 | 68% -> 69% |
| 18 | res5.2.conv2 | 26.47 | 25.56 | 1.04x | 22.7 | 60% -> 62% |
| 19 | res5.1.conv2 | 27.19 | 26.43 | 1.03x | 22.7 | 58% -> 60% |
| 20 | res5.0.conv2 | 27.39 | 27.29 | 1.00x | 22.7 | 58% -> 58% |
| 21 | res3.0.conv2 | 17.49 | 13.48 | 1.30x | 5.7 | 23% -> 29% |
| 22 | res4.0.conv2 | 14.96 | 12.43 | 1.20x | 5.7 | 27% -> 32% |

All 3x3 layers of one Panoptic inference: 1143 -> 1119 us (-2.1%). At or above 70%: aspp.convs.3, sem res2 fc1, sem dec res2 fc0, aspp.convs.2 (unchanged set). Overlapping base / final ranges: stem.conv1, res5.0, aspp.convs.1, aspp.convs.2 (`ablation_r12/ab_results.txt`).

### Guard set (whole perf file: 41 shapes plus the 22 focus cases, unzoned, 3 alternating pairs; `ablation_r12/ab_results.txt`)
- **HEIGHT (HiFi4).**
  - Wins: res3_0 -12.0%, res3 -4.6%, res2 / inst_res3_fuse_conv0 / center_head1 -3.2%, sem_res3_fuse_conv0 -1.7%, sem_res3_fuse_conv1 -1.6%, inst_res2_fuse_conv0 -1.0%, stem_conv2 -0.8%, inst_res2_fuse_conv1 -0.7%.
  - Flat: sem_res2_fuse_conv0 / conv1 -0.3%, stem_conv1 +0.1%, stem_conv3 -0.1%.
- **BLOCK (HiFi4 unless noted).**
  - Wins: res4_0 -8.9%, res4 LoFi -7.4%, aspp_convs3 -0.6%.
  - Flat: res5_1 / res5_2 / aspp_convs2 -0.3%, res5_0 -0.1%, res4 0.0%, aspp_convs1 +0.1%.
- **Interleaved (18 shapes).** -0.6% to +0.5%, except H16_W16_Ci64 rm_bf16 at +2.0%.
  - That cell's ranges overlap (9.98-10.06 vs 10.04-10.34); it is the same ~10 us layout-noise cell flagged in rounds 1-11.
  - H64_W128_Ci128 tile_bf16 reads +0.5% (+0.28 us, ranges 51.34-51.60 vs 51.71-51.72); its rm_bf16 / tile_bf8b twins read +0.2% / -0.1%.
  - No change of this round runs on the interleaved path: no weight servers, no kw-fold, no N-slice ring, stride 1. No carve-out is possible.
- **Golden.** `eval/golden_tests/conv2d_3x3/`: 1518 passed, 480 skipped (INVALID), 0 failed, on the final tree.
- **Unit dir.** 310 / 310 (11 new: 8 stride-2 taps, 3 BLOCK round-shard placement).
  - Each graduation was gated on its own files first: stride-2 unit + focus 30/30 (1), BLOCK + focus 83/83 (2), kw-fold + HEIGHT + focus 88/88 (3, 4), BLOCK + focus 83/83 (5).
- **`--dev`.** BLOCK, kw-fold, HEIGHT, stride-2 and layouts files 204 / 204 on the final tree.

### Bottleneck now (zoned final, `ablation_r12/zones_final.txt`, `focus.txt`)
- **res4.x (37%).** Round 0 still lands ~2.5-3.2 us after start on spare-column members (sender, then forward). Later rounds come every ~0.75-0.8 us against a real step cost of ~0.75 us: the 4 + 1 group's matmul, whose 1x1 short group costs ~0.19 us for 9 tile-MACs, and one matmul re-init per step. The ring chain is now hidden.
- **res3.0 (29%).** The tap build runs to ~6 us behind a 3.3 us fill, at ~49 ns per tap in the op against 40 in the bench. The walk (6.6 us, ~20.6 cycles per tile-MAC) is then paced by weight slots: the writer's `weight_reserve` sits 4.3 -> 8.9 us.
- **res4.0 (32%).** Fill 4.5 us; the walk is paced by the weight stream (~0.8 us per round).
- **Small HEIGHT kw-fold layers (res2 41%, res3.x 41%, center_head.1 44%).**
  - First subblock now ~2.2 us after start, bound by round-0 arrival (res2), the multicast rate at ~30-37 GB/s (res3.x) and fill bandwidth (center_head.1, a 122 KB slab).
  - With zero fill and instant weights the kernel ran 1.7-2.1 us faster than the base (zoned).
  - Reading the core's own shard in place would remove ~36% of the fill (not built).
- **86-core decoder layers.** C_out planes over the idle cores lose to the duplicated, long-way-round fill. A finer M or N partition needs a fill that does not double remote reads.
- **stem.conv1 (7%), aspp.convs.1 (66%), res5.0 (58%):** unchanged this round.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` / `untilize` (with the window gather) | capability | No helper decimates rows (a stride-2 tap is every other pixel row of two input tiles). None unpacks a whole tile into one Src bank outside the matmul, places rows at a DEST row offset, or packs a tile from a DEST window with a packer interface mask (0101 = every other row). | 17450 (res3.0, untilize + window gather + strided tilize prelude) | 13450 | conv2d_3x3_s2_taps.hpp:28 (justification), conv2d_3x3_compute.cpp:1293, :1916 |
| `dataflow_kernel_lib` `SenderPipe::send` / `ReceiverPipe::receive` (multicast) | capability | The send's ack wait has no polling form, so a forwarder cannot give its ring credit while it waits. The receive has no ack-ahead form: a receiver cannot ack round r + 1 before it forwards round r. The forwarder polls CONSUMER_READY with the credit, then calls the helper's own `send` (`SourceL1Guard::CallerManaged` for the non-blocking send is the helper's). It extends the Perf 9 `receive_polling` bypass. | 13331 (res4.x zoned, blocking ack wait, no per-round credit poll) | 11536 (zoned, both changes together, not separated) | conv2d_3x3_writer.cpp:1472 (`wait_acks_polling_on`), :1486 (`early_ack_on`) |
| `Semaphore<>::up` / `noc_semaphore_set_remote` | capability | Neither can signal ordered behind a data write without waiting for that write's ack, and neither takes a transaction id. Compute kernels have no semaphore accessor at all (the packer reads the base from the launch message). | 10653 (res4.x, weights stubbed); 13905 (11-column, 5-owner ring) | 10436; 11036 | conv2d_3x3_reader.cpp:1598 (count behind data), conv2d_3x3_compute.cpp:793 (`ring_arrival_count`) |
| `TensorAccessor` (sharded page -> NoC address) in the kw-fold fill | capability | No precomputed-run form: the accessor resolves page runs per call on the device (~0.1-0.2 us of RISC time each on a control-bound fill). It cannot take a host-built list of (owner core, offset, rows) runs for a sharded page range. The host already knows each shard's core and offset, so the descriptor now passes the runs. | 10110 (res2, device plan) | 9850 (fill change alone) | conv2d_3x3_reader.cpp:49 (justification), :738 (fill), conv2d_3x3_writer.cpp:530 (`kwf_fill_*`) |

The round-0 kernel-row push extends the existing raw polling receive (Perf 9 bypass, no new form). The per-slot server acks are raw semaphore increments and waits, like the rest of the server protocol (Perf 6 bypass), with no new helper gap.

## Perf 13
- Date: 2026-11-19
- Scope: perf tournament (run round 2 of 4) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores. No SUPPORTED change. Base `4f2f03f6d20` (end of Perf 12). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated (cap 6). 5 graduated (one of them a side finding of an idea whose own mechanism regressed), 1 NULL / REGRESSION.
  - All 3x3 layers of one inference: 1120 -> 1109 us (-1.0%; whole perf file, 3 alternating pairs).
  - res4.0 -10.7% (12.47 -> 11.13 us), stem.conv1 -4.9%, res5.2 -4.9%, res5.0 -4.3%, res3.0 -3.6%, res4.x -2.2%, res5.1 -2.0%, aspp.convs.3 -1.7% (means of the whole-file A/B below).
  - All 22 focus cases are bit-identical to the base.
  - No case crossed 70% this round; none at or above it moved below.
- Instrumentation: the Perf 1-12 zones are kept. New zones: `writer_round0` (kw-fold round-0 wait), `reader_r0_mcast` (slab-direct round 0), `writer_sd_weight_issue`, `writer_round0_issue` / `_send` / `_relay` (N-slice round 0). The host-planned BLOCK fill runs inside the existing `reader_fill` / `writer_fill_early` zones.
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r13/`. `ablation_hooks.patch` is the round-12 patch re-cut for `4f2f03f6d20`, plus stubs the round-12 patch lacked: the stride-2 DEST tap build (`tilize`), the kw-fold host-planned fill on both RISCs (`fill`), the BLOCK one-read weight rounds (`weights`).

### Measured breakdown (before, zoned, us)
DM-first cumulative peel on `4f2f03f6d20`: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only", "fill only", "weights only" stub that stage alone. stem.conv1's slab-direct matmul has no stub. All 22 cases in `ablation_r13/peel_results.txt`; zones in `zones_full.txt`, `zones_dm_scatter.txt`, `zones_x_matmul.txt`.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only | fill only | weights only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem.conv1 | 10.5 | 10.5 | 10.7 | 9.8 | 9.6 | 9.6 | 9.6 | 9.6 | 10.7 | 10.5 | 9.5 |
| stem.conv2 | 27.6 | 27.5 | 27.4 | 26.3 | 26.3 | 26.3 | 26.3 | 19.5 | 19.7 | 27.3 | 27.0 |
| res2 | 10.3 | 10.3 | 10.3 | 10.2 | 10.2 | 10.2 | 10.2 | 7.0 | 7.8 | 10.2 | 10.5 |
| res3.0 | 13.8 | 13.8 | 12.6 | 12.6 | 12.6 | 12.6 | 9.8 | 4.6 | 10.4 | 12.6 | 13.8 |
| res3.x | 10.2 | 10.3 | 10.2 | 9.4 | 9.4 | 9.4 | 9.5 | 6.9 | 9.0 | 10.3 | 9.5 |
| res4.0 | 12.8 | 12.7 | 11.0 | 10.4 | 10.4 | 10.3 | 8.6 | 4.9 | 10.9 | 11.1 | 12.6 |
| res4.x | 11.0 | 11.1 | 11.1 | 9.7 | 9.6 | 9.6 | 9.2 | 5.9 | 10.3 | 11.0 | 10.7 |
| res5.0 | 27.9 | 28.0 | 28.0 | 24.8 | 24.8 | 24.8 | 23.3 | 7.7 | 19.0 | 27.8 | 27.0 |
| aspp.convs.1 | 48.2 | 48.7 | 45.5 | 45.0 | 44.9 | 44.8 | 41.2 | 9.5 | 31.9 | 45.7 | 48.0 |
| aspp.convs.3 | 20.7 | 20.4 | 20.1 | 18.7 | 18.6 | 18.5 | 18.1 | 8.5 | 16.9 | 20.0 | 20.6 |
| sem dec res3 fc0 | 34.9 | 35.2 | 34.6 | 32.9 | 32.9 | 32.9 | 33.0 | 13.1 | 19.5 | 34.6 | 33.3 |
| center_head.1 | 9.7 | 9.7 | 9.5 | 9.5 | 9.5 | 9.5 | 9.5 | 5.6 | 6.2 | 9.5 | 9.9 |

Critical-core timelines (`ablation_r13/timeline.py`, us after the run's earliest kernel start), ranked by gap to target x calls (Perf 12 final, us per inference): res4.x 25.5, res3.x 16.4, decoder res3 layers 16.4, res2 12.3, res5.x 11.2, stem.conv1 9.1, res3.0 7.8, res4.0 6.7, center_head.1 6.7.
1. **kw-fold small layers are TRISC-bound with a control-bound start.** res2 does not move with every DM stage stubbed (10.3 -> 10.2). Its critical core: kernel start 0.4, first MAC ~2.2, Q_1 round to 4.7, side rounds to 8.8 (6.6 us walk against a 4.3 us FPU floor on 103 cores), final pass 8.8-10.3. With every read and the weights stubbed the reader's slab hand-over still ends ~2.1 us (waiting on the writer). center_head.1's rounds land 2.1 / 3.0 / 4.2; res3.x's round-0 chunks to 3.3.
2. **res4.x is co-bound by step compute and the column stream.** Weight rounds ~0.8 us on every member, step compute ~0.75 us (a 5-row group as 4x1 + 1x1). Peel: mm only -0.7, weights only -0.3, all DM -1.4, everything stubbed 5.9 (ring chain + store). Prelude (tap shift) 1.9 us; round 0 at 2.0 (grid) / 3.2 (spare) us.
3. **res5.x start.** Round 0 lands 3.3-5.4 us on grid members and 6.1-6.7 on spare members, after their fill-paced prelude (3.5-4.7): the critical core idles ~2.2 us before its first step. The walk then runs at the 4-tile-subblock rate.
4. **Stride 2.** res3.0 is TRISC-serial: with the fill stubbed the tap build still takes 0.7-4.6 and the walk 4.7-11.7. res4.0's spare member fills 0.3-4.8 (grid 1.6-2.5), which gates its prelude to 5.3.
5. **stem.conv1.** Compute waits for the weight round until 4.4 us on the critical core.
6. **decoder res3 / res3.x on 86 cores.** The walk at ~20 cycles per tile-MAC on 86 cores; Perf 12's C_out planes over the idle cores was NULL. No idea spent.

Roofline gate: measured rates against the LoFi FPU floor (16 cycles per tile-MAC; ~19.4 at 4-tile fp32 subblocks) and the NoC link rate, as in Perf 8-12. At the floor: res5.x's and aspp's walks, the decoder walks. Far from it: every start-up above, res4.x's step.

### Portfolio and verdicts
Each idea was built and measured in its own experiment dir (`perf_experiments/<slug>/`) by its own optimizer. Numbers are unzoned medians of 3 alternating pairs unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `kwf_startup`: kw-fold time to first MAC | WIN (small) | res2 9.86 -> 9.58 (-2.9%); res3.x 9.72 -> 9.66, center_head.1 9.10 -> 9.17 (flat) | Attribution: the server waited for 99 kernel-start acks (~0.8-1.0 us, ~10 ns apart; sending them on NoC0 changed nothing), the writer reached its round-0 wait ~1.2 us after kernel start, the server kernel launched last. Change: no server start ack (a program's go reaches every core only after the previous program finished and every core's config was written: dispatch_s `wait_for_workers` before the go), the writer goes from its fill issue straight into the round-0 wait (serving the fill, pushing chunks and the server's bias), every one-burst round issues all chunks at once. First matmul (median over cores): res2 1.63 -> 1.31, res3.x 1.88 -> 1.64, center_head.1 1.73 -> 1.44 us; it converts only partly because the later chunks and the slab tail land during Q_1. Rejected: server kernels listed first (res3.0 +3.0%), round 0 all at once on large rounds (decoder first matmul ~1.1 us later), row-band server rectangles (6 us multicast stall). Flag: the kw-fold fill self-reads the core's own shard rows; skipping them (wrong output) measured no change. Bit-identical. |
| `block_round0`: BLOCK N-slice round 0 before the prelude ends | WIN (small alone; larger on the host-planned fill) | res5.0 -1.9% to -2.6%, res4.x -0.9% to -1.7%; fill-stubbed res5.x -7%, res4.x -4% | Attribution: the writer reached its first read 0.9-1.4 us in; the 39 KB read took 0.8-2.2 us by column, set by NoC1 link sharing (DRAM responses climb the DRAM column, then run along the reader's row; a fitted link-sharing model: 0.8 us alone, 1.5-2.4 us with five to eight reads on one link); the spare relay added ~1.2 us; the multicast ~0.5 us; with the fill stubbed round 0 landed as late. Change: no handshake for round 0 (every slot is empty at kernel start), one round-0 reader per column (the row the NoC1 model picks on Blackhole, else the column sender), the column sender relays round 0 to its spare row, the async bias after round 0, no slot-reuse flush on never-used slots (without it the relay regressed res4.x +1.8%). Round 0 on res5.0 spare members 5.7 / 6.6 -> 4.3 / 4.5 us (p50 / max). Rejected: moving the column sender (1.4-3.5x slower), a distant reader multicasting to the spare row (+9%), holding round 1 until round 0 lands. Bit-identical. |
| `nslice_step_shape`: res4.x per-step compute | REGRESSION (2-tile slices) / NULL | w2u (2-tile slices, per-subblock ring units) res4.x 10.68 -> 11.17 (+4.6%); weights stubbed 10.43 -> 10.23 (-1.9%) | The walk is paced by the column stream (0.78-0.85 us per round): taking the whole 1x1 short group out (wrong sums) gains 0.23 us on the real stream, 1.27 us with the weights stubbed. A 1x1 call costs ~24 cycles against ~18 per tile-MAC in the 4x1 group; one matmul re-init ~52 ns. 2-tile slices: 19.6 KB round 0 lands 0.6-1.0 us later and the four owners' combine + store doubles the tail; aspp.convs.3 +11%, HiFi4 res4 +6.0%; aspp.convs.1 / .2 inexpressible (L1 falls to bm 1, 4.5-5x slower). Correction: res5.x runs 8 two-tile slices. No patch. |
| `s2_build_walk_overlap`: software-pipeline the stride-2 tap build with the walk | REGRESSION (overlap); side finding WIN | overlap: res3.0 +11.4% (build K-block kb + 1 after walk kb), +30.6% (build pieces between subblocks); side finding: res3.0 13.44 -> 12.98 (-3.5%) | The overlap cannot win: round 0 lands at ~3.5 us and the fill ends at 3.7, after which build and walk are pure TRISC; a tap section takes a whole fp32 DEST half, so any section between walk subblocks breaks the walk's double buffering (+118 ns per section); the walk is at its floor (~20 cycles per tile-MAC). res4.0 ceiling with step 0 free: -3.0% (not built). Side finding: on every K-blocked walk with a bias, the helper's per-row-group partials reserve / push made the unpacker wait at each K-block boundary for the last subblock's pack. With a bias the walk now owns the partials block (`caller_owns_pack_target`), as the K-split walk already does. Domain: res3.0 HiFi4 -1.6%, stride-2 unit shapes -2.2% to -5.7%, DRAM 16x16x64 -3.0%, the rest flat. Bit-identical. |
| `stem1_weight_round`: stem.conv1's weight round | WIN | stem.conv1 10.27 -> 9.83 (-4.3%), at the no-weights ceiling (9.80) | Landing the round earlier alone was flat; the NoC1 multicast slowed the last grid row's NoC1 forward fill, and those cores bound the op. Change: the DRAM read is issued at writer entry, the sender core's reader multicasts round 0 on NoC0 with its own handshake (receivers ack at kernel start, a ready flag), later rounds stay on the NoC1 channel, receivers stop polling for the weights in their slab issue loop. Weights land p50 3.0 / max 3.7 us (was 3.6 / 4.4). One-read weight placement: no further gain, not taken. Domain: smaller slab-direct shapes -5.5% to -23%, the rest flat. Bit-identical. |
| `block_fill_start`: BLOCK spare-member fill | WIN | res4.0 12.55 -> 11.27 (-10%) | Attribution: a spare group's 8 members read their slab from 1-2 owner rows, so every NoC0 response funnels through one spare column (res4.0's group 11 wraps round column 14): spare fills ended 4.1-4.8 us vs 1.6-2.5 on grid cores, and the backed-up reads slowed grid cores in columns 2-4 to ~4.5 us. The device-planned writer split was control-bound (~0.4 us planning, ~0.27 us per 4-row batch, identical on every spare core), which is why Perf 11's two-NoC split was flat. Change: the host plans every core's fill (owner, offset, slab row, rows), the writer reads the upper half of each spare member's slab on NoC1 from kernel start. Spare fill now ends 1.9-2.65 us. Host plan alone (spare reads on NoC0): +1%. Domain: res5.1 / 5.2 -1% to -3%, HiFi4 res4.0 -6.1%, the rest flat. Bit-identical. |

Not floated (cap 6, reason each):
- **Reuse-shared broadcast.** Every shared operand already reaches its consumers by multicast (weights to each column / rectangle, round 0 now included); the input halo push was a measured REGRESSION in Perf 12 (`block_halo_mcast`).
- **kw-fold final pass.** Perf 11 NULL (pack-bound at ~47 cycles per output tile).
- **decoder res3 / res3.x on 86 cores.** Walk at the 4-tile rate on 86 cores; Perf 12 `kwf_n_planes` NULL.
- **res4.x column stream rate alone.** Weights-only stub worth 0.3 us zoned.

### Graduated (one path each, in commit order)
1. **K-blocked walk owns its partials with a bias** (`09d969f2575`, from `s2_build_walk_overlap`). Clean apply. Without a bias the last K-block reloads the partials (the helper's Out target), which needs the helper's pushes: unchanged (inexpressible, not a guard).
2. **Slab-direct round 0 on NoC0** (`7c2f2a22db3`). Clean apply, plus a deadlock fix found at the merge gate: on a core with two or more M-blocks the sender's reader blocked in the next block's slab `cb_reserve_back` before its round-0 multicast had completed; its writer waits on that for round 1 and compute on round 1, so the slab never freed (hung `test_conv2d_3x3_height_sharded_slab_direct_whole_grid[H128_W256_Ci8_Co64_s2_bm1-no_bias]` outside `--dev` only: timing). The reserve now polls `r0_try`. Unit test file +6 tests (the optimizer's). Taken on every slab-direct shape with an in-grid sender; under weight servers round 0 comes from the server (inexpressible).
3. **kw-fold start without server start acks** (`bf5905f4a93`). Merged by hand onto (2): both moved the writer's weight-round helpers; the round-0 wait now follows the slab-direct early issue, and `k_blocks_issued` starts at 1 once round 0 is in.
4. **BLOCK host-planned fill** (`8aba3ae0650`). Merged with `git merge-file` (the patch had no blob ids): its writer hunks sat in code (2) and (3) had moved; only its new host-fill code was kept (the displaced copies matched the base). Unit test file +5 tests. The device-planned fill stays where the host plan is inexpressible: more than one M-block per core, the slab not whole in the input ring, non-sharded input, or more than `HOST_FILL_MAX_WORDS = 256` planned words per RISC (runtime-arg budget; no tested shape comes close).
5. **N-slice round 0 without a handshake** (`9969908e7f9`). Both (4) and this appended writer runtime args after the server args: the round-0 words (11) come first, the host-fill words after them. Unit test file +2 tests. Measured on the tree with (4), unzoned, 3 alternating pairs (`ablation_r13/ab_focus_round0.txt`): res5.0 27.09 -> 26.02 (-4.0%), res5.2 24.80 -> 24.26 (-2.2%), aspp.convs.3 -1.2%, res5.1 -0.8%, res4.0 / aspp.convs.1 / .2 flat, res4.x 10.56 -> 10.65 (+0.9%, ranges 10.47-10.62 vs 10.54-10.72: inside the noise band, no carve-out).
- Experiment files: run logs and profiler CSVs over 500 KB stay on disk, uncommitted (large-file hook).

Carve-outs, each with the reason that earned it:
- **None earned by a measured regression.**
- **Inexpressible / feasibility only:** the K-blocked walk without a bias (helper Out-target reload); slab-direct round 0 under weight servers; the BLOCK host fill with more than one M-block per core, a partial slab ring, non-sharded input or over 256 words per RISC.
- **Not carved out: center_head.1 +1.4%** (medians 9.05 -> 9.18 us, ranges 9.00-9.14 vs 9.03-9.23) and inst dec res3 fc0 +0.6%: inside the noise band. `kwf_startup`'s own A/B read center_head.1 +0.7% (flat).

### Precision
- **All 22 focus cases are bit-identical to `4f2f03f6d20`** (`ablation_r13/merge_gates/test_focus_exact.py`, one process, 0 differing elements on every case).
- No change touches the precision contract: the partials change keeps the same matmuls, packs and order; the rest is data movement and scheduling.

### Whole-op result (unzoned, us; whole perf file, base `4f2f03f6d20` and final `9969908e7f9`, mean of 3 alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 20.25 | 19.91 | 1.02x | 22.7 | 78% -> 80% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 83.75 | 83.44 | 1.00x | 90.8 | 76% -> 76% |
| 3 | res4.{1..5}.conv2 | 10.78 | 10.54 | 1.02x | 5.7 | 37% -> 38% |
| 4 | res3.{1,2,3}.conv2 | 9.84 | 9.77 | 1.01x | 5.7 | 40% -> 41% |
| 5 | stem.conv2 | 27.20 | 27.17 | 1.00x | 22.7 | 58% -> 58% |
| 6 | aspp.convs.2 | 42.16 | 42.10 | 1.00x | 45.4 | 75% -> 75% |
| 7 | aspp.convs.1 | 48.13 | 47.74 | 1.01x | 45.4 | 66% -> 67% |
| 8 | res2.{0,1,2}.conv2 | 9.94 | 9.81 | 1.01x | 5.7 | 40% -> 40% |
| 9 | center_head.1 + offset_head.1 | 9.06 | 9.15 | 0.99x | 5.7 | 44% -> 43% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 24.00 | 23.95 | 1.00x | 22.7 | 66% -> 66% |
| 11 | stem.conv3 | 47.74 | 47.70 | 1.00x | 45.4 | 67% -> 67% |
| 12 | sem dec res2.fuse_conv.0 | 92.78 | 92.79 | 1.00x | 102.1 | 77% -> 77% |
| 13 | sem dec res3.fuse_conv.0 | 34.26 | 34.09 | 1.01x | 28.4 | 58% -> 58% |
| 14 | sem dec res3.fuse_conv.1 | 28.68 | 28.24 | 1.02x | 22.7 | 55% -> 56% |
| 15 | stem.conv1 | 10.25 | 9.75 | 1.05x | 1.1 | 7% -> 8% |
| 16 | inst dec res3.fuse_conv.0 | 18.88 | 18.95 | 1.00x | 14.2 | 53% -> 52% |
| 17 | inst dec res2.fuse_conv.0 | 28.89 | 28.87 | 1.00x | 28.4 | 69% -> 69% |
| 18 | res5.2.conv2 | 25.57 | 24.32 | 1.05x | 22.7 | 62% -> 65% |
| 19 | res5.1.conv2 | 26.53 | 25.99 | 1.02x | 22.7 | 60% -> 61% |
| 20 | res5.0.conv2 | 27.25 | 26.07 | 1.05x | 22.7 | 58% -> 61% |
| 21 | res3.0.conv2 | 13.48 | 13.00 | 1.04x | 5.7 | 29% -> 31% |
| 22 | res4.0.conv2 | 12.47 | 11.13 | 1.12x | 5.7 | 32% -> 36% |

All 3x3 layers of one Panoptic inference: 1120 -> 1109 us (-1.0%). At or above 70%: aspp.convs.3, sem res2 fc1, sem dec res2 fc0, aspp.convs.2 (unchanged set). res2's whole-file gain (-1.3%) is half its optimizer's isolated -2.9%. Per-case ranges: `ablation_r13/ab_results.txt`.

### Guard set (whole perf file: 41 shapes plus the 22 focus cases, unzoned, 3 alternating pairs; `ablation_r13/ab_results.txt`)
- **BLOCK (HiFi4 unless noted).** res4_0 -6.1%, res4 LoFi -2.1%, res4 -1.7%, res5_2 -1.7%, res5_0 -1.6%, res5_1 -0.8%, aspp_convs3 -0.8%, aspp_convs2 -0.5%, aspp_convs1 -0.1%.
- **HEIGHT (HiFi4).** res3_0 -1.8%, stem_conv1 -1.2%, res2 -1.1%, res3 -0.7%, inst_res2_fuse_conv1 -0.6%, inst_res2_fuse_conv0 -0.4%; flat: stem_conv2 / conv3, sem_res3_fuse_conv0 / conv1, sem_res2_fuse_conv1, center_head1 (0.0%), sem_res2_fuse_conv0 +0.1%, inst_res3_fuse_conv0 +0.2%.
- **Interleaved (18 shapes).** -1.4% to +1.1%: H16_W16_Ci64 rm_bf16 +1.1% (ranges overlap; the layout-noise cell of rounds 1-12), the rest within ±0.7%. The partials change runs on the interleaved K-blocked walk with a bias; its optimizer measured DRAM 16x16x64 -3.0% and the other DRAM shapes -1.0% to +0.4%.
- **Golden.** `eval/golden_tests/conv2d_3x3/`: 1518 passed, 480 skipped (INVALID), 0 failed, on the final tree.
- **Unit dir.** 323 / 323 (13 new: 6 slab-direct whole-grid, 5 BLOCK host fill, 2 N-slice round 0). Each graduation was gated on its own files first: stride-2 + focus 30/30 and interleaved + HEIGHT + layouts 129/129 (1); HEIGHT + layouts + interleaved + focus 157/157, the new slab-direct tests 3 times (2); kw-fold + HEIGHT + layouts + interleaved + focus 182/182 (3); BLOCK + stride-2 + focus + kw-fold + HEIGHT 168/168 (4); BLOCK + stride-2 + focus 98/98 (5).
- **`--dev`.** BLOCK + focus 90/90; kw-fold, HEIGHT, stride-2 and layouts 149/149 on the final tree.

### Bottleneck now (zoned final, `ablation_r13/zones_final.txt`, `focus.txt`)
- **res4.x (38%).** Prelude 0.5-2.8, round 0 at 2.3 us, walk 2.6-9.7 at ~0.85 us per round (stream and step compute co-bound), store to 10.6. With the stream as fast as compute, the step's 1x1 short group and per-step re-init are ~0.3 + 0.4 us of the walk.
- **res5.x (61-65%).** Round 0 now lands with the prelude (4.4 vs 4.6-4.9 us on the critical spare member); fill 0.7-3.1. The walk (20 us) runs at the 4-tile rate.
- **res4.0 (36%).** Fill ends 2.0 us, prelude 2.9-3.4, the walk paced by the stream at ~1 us per round.
- **Small HEIGHT kw-fold layers (res2 40%, res3.x 41%, center_head.1 43%).** First matmul ~1.3-1.6 us after kernel start; the walk (6.6-8 us, 1.5-1.9x the FPU floor) and the final pass (1.0-1.5 us) are now most of the gap. With all start-up DM stubbed the layers run 8.94 / 8.53 / 8.33 us.
- **res3.0 (31%).** Tap build 0.6-5.0, walk 4.4-12.4, writer weight_reserve then the last rounds.
- **stem.conv1 (8%).** Weights land 3.0-3.7 us; the slowest cores now wait on their slab forward (reader fill to 5.3 us).

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `dataflow_kernel_lib` `SenderPipe` / `McastArgs` (multicast channel) | capability | The channel is bound to one RISC's NoC (the writer's NoC1) and one sender core. It cannot issue a round from the reader on NoC0 while later rounds keep the writer's NoC1 channel, so round 0 needs its own ack count and ready flag (two extra semaphores) beside the channel's. | 10273 (stem.conv1, early-issued round 0 on the NoC1 channel: 10.24-10.26 us) | 9828 | conv2d_3x3_reader.cpp:121 (justification), :311 (`r0_try`) |
| `dataflow_kernel_lib` `SenderPipe::send` (multicast channel) | capability | No send without the CONSUMER_READY handshake (a round into slots known empty at kernel start), and none from a core that is not the channel's fixed sender (the round-0 reader is a model-picked row). The relay to the spare row writes into the second channel, whose sender is the forwarder. | 27090 (res5.0 on the host-fill tree, round 0 through the channel) | 26020 | conv2d_3x3_writer.cpp:93 (justification), :1596 / :1794 / :1908 (issue, send, relay) |
| `TensorAccessor` (sharded page -> NoC address) in the BLOCK fill | capability | No precomputed-run form (as in Perf 12's kw-fold fill): the device plan resolved owner and page runs per call, ~0.4 us before the first read and ~0.27 us of RISC time per 4-row batch on a fill-critical writer. The host passes (owner x / y, offset, slab row, rows) runs instead. | 12459 (res4.0, device-planned split) | 11266 | conv2d_3x3_reader.cpp:858 (`CONV2D_3X3_HOST_FILL`), conv2d_3x3_writer.cpp:822 (`hf_serve`) |

The kw-fold round-0 wait reuses the existing raw polling receive (Perf 9 bypass: `ReceiverPipe::receive` has no polling form) and the server protocol's raw semaphore flags (Perf 6); no new helper gap. The partials change uses the helper (`caller_owns_pack_target`).

## Perf 14
- Date: 2026-11-20
- Scope: perf tournament (run round 3 of 4) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores. No SUPPORTED change. Base `1a04504f992` (end of Perf 13). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated (cap 6), 5 graduated, 1 NULL.
  - All 3x3 layers of one inference: 1109 -> 1080 us (-2.6%; whole perf file, 3 alternating pairs).
  - res4.0 -10.6% (11.13 -> 9.94 us), res4.x -10.3% (10.61 -> 9.51), aspp.convs.1 -7.5% (47.85 -> 44.24), res2 / res3.x -5.7%, center_head.1 -3.3%, aspp.convs.3 -3.3%, res5.1 -2.8%.
  - **aspp.convs.1 crosses 70%** (66% -> 72%). aspp.convs.3 is now at 83%.
  - All 22 focus cases are bit-identical to the base.
- Instrumentation: the Perf 1-13 zones are kept; no zone was added or removed. The new paths run inside existing zones: the ring k-steps in `compute_matmul_walk`, the spread rounds in `writer_weight_fetch`, the partner server in the `server_*` zones, the balanced fill in `reader_fill` / `writer_fill`. One zone location was moved: after clang-format re-wrapped a writer line, `writer_weight_fetch`'s source location hashed onto the firmware's `PROFILER-NOC-FLUSH` (`22f444dd244`).
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r14/`.
  - `ablation_hooks.patch` is the round-13 patch re-cut for `1a04504f992`. It adds stubs that patch lacked: the host-planned BLOCK fill on both RISCs (`fill`), the N-slice round 0 and spare relay, and the kw-fold round-0 wait (`weights`).
  - Not covered: the slab-direct NoC0 round 0 (stem.conv1's `weights` stub leaves it running). For the next round, the `matmul` stub must also skip the new `ring_ksteps` branch.
  - `startup.py` and `start_spread.py` measure the kernel-start spread (below). The bit-exact gate (final vs base, all 22) is `merge_gates/test_focus_exact.py`.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel on `1a04504f992`: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only", "fill only" and "weights only" stub that one stage. All 22 cases are in `ablation_r14/peel_results.txt`; zones are in `zones_full.txt` (base), `zones_dm.txt` (all DM stubbed) and `zones_all.txt` (everything stubbed).

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only | fill only | weights only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem.conv2 | 27.6 | 27.6 | 27.4 | 26.4 | 26.3 | 26.3 | 26.3 | 19.5 | 19.6 | 27.5 | 27.0 |
| stem.conv3 | 48.1 | 48.0 | 48.1 | 45.8 | 45.8 | 45.8 | 45.8 | 36.6 | 39.2 | 48.0 | 47.2 |
| res2 | 10.2 | 10.5 | 10.3 | 9.4 | 9.4 | 9.5 | 9.4 | 7.1 | 7.6 | 10.2 | 9.9 |
| res3.0 | 13.2 | 13.3 | 12.1 | 11.8 | 11.9 | 11.8 | 9.1 | 3.8 | 9.6 | 12.0 | 13.1 |
| res3.x | 10.4 | 10.5 | 10.2 | 8.7 | 8.7 | 8.8 | 8.8 | 5.7 | 8.6 | 10.1 | 8.9 |
| res4.0 | 11.5 | 11.4 | 10.9 | 10.5 | 10.4 | 10.5 | 9.2 | 5.6 | 10.0 | 10.9 | 11.2 |
| res4.x | 11.0 | 10.9 | 10.8 | 9.8 | 9.8 | 9.8 | 9.5 | 6.2 | 9.8 | 10.7 | 10.3 |
| res5.0 | 26.0 | 26.1 | 25.7 | 25.0 | 24.9 | 25.0 | 23.9 | 8.0 | 18.4 | 25.7 | 26.2 |
| aspp.convs.1 | 48.0 | 48.2 | 45.3 | 45.0 | 44.9 | 44.9 | 40.5 | 8.6 | 31.4 | 45.4 | 48.1 |
| aspp.convs.2 | 42.2 | 42.3 | 39.8 | 39.0 | 38.8 | 38.8 | 35.9 | 9.0 | 30.9 | 39.8 | 42.4 |
| sem dec res3 fc0 | 35.1 | 34.8 | 34.6 | 32.4 | 32.4 | 32.4 | 32.6 | 12.6 | 19.3 | 34.8 | 33.4 |
| inst res2 fc1 | 24.2 | 24.1 | 24.2 | 22.8 | 22.8 | 22.8 | 22.8 | 10.9 | 10.8 | 24.2 | 23.6 |
| center_head.1 | 9.7 | 9.7 | 9.3 | 8.8 | 8.8 | 8.8 | 8.8 | 5.1 | 6.8 | 9.3 | 9.8 |

Ranked by gap to target x calls (Perf 13 final, us per inference): res4.x 24.3, res3.x 16.4, decoder res3 layers 16.0 (three layers), res2 12.4, stem.conv1 8.7, res5.x 8.3, res3.0 7.3, center_head.1 7.0, res4.0 5.5, aspp.convs.1 4.7.
1. **Weight delivery on the server-fed HEIGHT layers.** The fill + weights to weights-stubbed step removes 0.5-2.3 us from every kw-fold layer (res3.x -1.5, sem dec res3 fc0 -2.2, inst dec res3 fc0 -2.1, stem.conv3 -2.3, sem res2 fc1 -1.7, inst res2 fc1 -1.4, res2 -0.9). That is ~31 us per inference, the largest single lever.
   - On res3.x, compute waited ~1.4 us for round 0's three kernel-row chunks (`compute_kwf_chunk_wait` p50 1.41).
   - Server timeline: DRAM read of the 52 KB round 0.40-1.81 us, sends 1.18-2.21, flush to 2.72; receivers saw the chunks at ~2.2, 2.7 and 3.3 us.
2. **res4.x: walk co-paced by the column stream, then a ~1.1 us tail.** Critical core: prelude to 2.4-2.7, 8 steps at ~0.9 us to 9.9 (round fetches 0.8-1.2 us each), combine 9.9-10.4, store to 10.8, end 11.0. Weights stub -1.0 us, mm-only stub -1.2, everything stubbed 6.2 (ring + store skeleton). Every N-slice layer ends the same way: ~1 us from walk end to kernel end.
3. **kw-fold walk at narrow n-groups.** With all DM stubbed, the walk ran 7.0 us for 360 tile-MACs (~26 cycles each) on center_head.1 (gw 1) and res2 (gw 2), against ~18.5 on the wide layers (inst res2 fc1, sem dec res3 fc0). The packer alone takes 3.5 us on center_head.1 (not the bound) and 5.2 us on res2 (60 fp32 L1-acc packs at ~117 cycles: co-bound).
4. **BLOCK fill.** The fill-only stub saves aspp.convs.1 2.7 us, aspp.convs.2 2.4, res5.1 1.4, res5.2 1.3, aspp.convs.3 1.0. On aspp.convs.1's critical core the reader fill runs 1.2-6.6 us and the tap build (gated by it) to 11.1, before a 36 us walk at the 16-cycle FPU floor.
5. **Kernel-start spread (not op-controlled).** DEVICE KERNEL DURATION counts each critical core's start offset. It is 0.13-0.46 us per case, 7.2 us over the 22 cases (`start_spread.py`).
   - The pattern is fixed by core position, identical in every run: x = 1-2 cores start ~0.4 us after x = 11, and the launch-message arrival (FW start) shows the same gradient.
   - It is a clock offset or go-signal latency, not kernel work. Late launch messages for the last kernel groups add ~0.5-0.75 us of prep on a few cores (res4.x's spare rows), off the critical path.
   - No idea spent.

Roofline gate, as in Perf 8-13: aspp.convs.1/2's walk and the decoder walks are at the floor. Far from it: the start-up items above, res4.x's step, the narrow kw-fold walk.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Numbers are unzoned medians of 3 alternating runs unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `server_dual_mcast`: each kw-fold server rectangle fed by two servers on both NoCs | WIN | res3.x 9.753 -> 8.986 | Attribution (zoned): multicast stubbed -0.97 us, server DRAM reads stubbed -0.25. One source pushed round 0 at ~34 GB/s beside the fill. Change: a partner spare core per rectangle reads and multicasts half of every chunk (whole slot rows) on NoC0 while the first server sends the other half on NoC1. Each server writes its own byte lane of the flag word (0x0101 = both halves landed). Receivers ack the first server, which forwards the clearance to the partner. Both servers issue a round's chunks up front. Partners and servers are placed by NoC hops, with paths that cross no receiver rectangle. Rejected: nearest-core placement (11.9 us zoned vs 10.15: the 9-core rectangle's partner routed NoC0 through the 77-core rectangle); partner on the big rectangle only (the small one became critical). Domain: res2 -3.4%, decoder res3 layers -1.4% to -1.6%; center_head.1, sem / inst res2 fc1, decoder res2 layers flat (+0.9% to -0.2%). **Measured regression: res3.0** (non-kw-fold K-blocked walk) +2.0% to +2.7% over 3 options x 3 runs: the first matmul waits for the tilize there, round 0 is hidden, and the partner's NoC0 multicast only slows the reader's NoC0 fill (3.62 -> 3.93 us). No new semaphore (13 of 16 at most). |
| `nslice_stream_spread`: every BLOCK column weight round in its own slot, read and multicast by its own column member | WIN | res4.x 10.521 -> 10.101 | Attribution: the round period was a serial chain on one sender. A receiver acked round r + 1 only after r landed (~0.2 us); the sender waited for every ack (0.05 -> 0.42 us by round 7); the multicast landed ~0.35 us after its 0.12 us issue; each read's ND `TensorAccessor` arithmetic took ~0.15-0.2 us of RISC time. Change: round 0 from the round-0 reader, round 1 from the column sender, rounds 2+ from the other rows in turn; lookahead 2 (round 1 waits until round 0 landed on its sender); no handshake; one flag semaphore per round; the round's DRAM address from the shard index. Rejected: all 8 rounds read at kernel start (+21% zoned: 64 reads share the DRAM columns' NoC1 links); every round from one reader (+10%; +3.2% with lookahead 2); lookahead 1 (+14% to +18%); 8 slots with the handshake kept (-1.5%: the gain is mostly the handshake). Domain: res4.0 -2.1%, LoFi res4 -3.3%, HiFi4 BLOCK within +-0.3%, res5.1 / 5.2 / aspp.convs.3 flat. res5.0 +1.9% (+0.8% to +2.7% over every run of the variant), inside the band. Inexpressible (handshake stream stays): aspp.convs.1/.2 (8 rounds of 59 KB do not fit L1), one-round streams, ragged rows, the 16-semaphore budget. |
| `nslice_tail`: cut the N-slice tail (combine + output store) | NULL | best bit-identical option res4.x 10.576 -> 10.448 (-1.2%, flat) | Attribution: the combine (0.51 us on all 104 cores) is pack-bound: fp32 DEST -> bf8_b ~0.075 us per tile, plus 0.10-0.17 us of pack reconfig waiting for the walk's last fp32 pack. Pack-only, 4-tile batches, srcA copy and UnpackToDestFp32 copy measured 0.45-0.54 us. The store issue and flush after the last tile take 0.22-0.39 us, and the write ack at kernel end 0.16 us on spare-column cores. Ceilings (wrong output): no combine -2.4% to -3.0%, no store -3.2%, neither -6.2%. Options, all flat: early-resolved store addresses (-1.2%), combining the first row group under the last group's pack (+0.5%), 4-tile DEST batches (+1.4%), the reader storing on NoC0 (flat). Folding bias / ReLU / format into the owner's last step (reload the incoming partial into DEST) was modelled from these costs at <= 0.05 us net, not bit-identical; not built. Nothing graduated. |
| `kwf_narrow_walk`: the kw-fold walk at gw <= 2 | WIN (small) | center_head.1 9.096 -> 8.866 | Attribution (DM stubbed): with every pack removed the walk did not move (7.08 vs 7.10 us). The costs were matmul_block's runtime subblock dims (LLK loops and the ct >= rt operand choice resolved per call) and a `matmul_block_init` at every height change (4 -> 2 -> 4), although the BH MOPs depend on the dims only through ct >= rt. Change: compile-time dims per subblock, re-init only where ct >= rt flips, a pass's rows never split 4 + 1 (3 + 2 instead). Walk 7.10 -> 6.50 us. Side finding: here 4 x 1 (in0 streamed via srcB) runs ~20.9 cycles per tile-MAC and 1 x 3 / 1 x 4 ~18.4, against the Perf 3 lesson that orientation does not matter. Rejected: one 1 x 3 pass [Q_1 Q_0 Q_2] per row (+9.5% on the op: it waits for all three rounds, round 2 lands ~2 us after round 0's first chunk; -8% to -10% with the weights stubbed); 1 x 3 for the short rows only (flat to +4%); fused 1 x 2 side rounds at gw 1 (+24%). Domain: res2 -1.9%, stem.conv2 -2.1%, the other kw-fold focus cases -0.9% to +0.5%, HiFi4 HEIGHT kw-fold shapes flat. |
| `block_fill_balance`: shorten the slowest cores' BLOCK fill | WIN | aspp.convs.1 47.78 -> 45.64 (median over 5 pairs) | Attribution: every response in a column runs along it. The busiest NoC0 column link carried 268 KB (325 KB in columns whose weight-sender row read its higher shards the long way round), against at most 83 KB on NoC1. All batches were in flight at once, so the tap build's first kernel row (5.8 us on the critical core) landed with the rest. Change: (1) per column, the host moves single slab rows between the reader (NoC0) and the writer (NoC1), either way round the column, until link, owner-injection and reader-ejection loads stop improving (busiest link 268 -> ~183 KB against an LP optimum of ~182); spare members' reads count as fixed load and the weight-sender row now splits too; (2) each RISC issues the first kernel row's band alone, the rest once it landed; (3) the reader counts the writer's landings from NoC1's transaction-id counters, so a writer busy with weights never serves the fill. Balance alone -0.7 to -1.6 us, gate alone flat. Rejected: deferred zero block, one-at-a-time first-band batches, reader windows 2-4. Domain: aspp.convs.2 -5.0%, aspp.convs.3 -4.5%, HiFi4 aspp -1.1% to -1.5%; res5.1/5.2, res4.x flat. **Measured regression, excluded:** overlapping kernel-row bands (32x64x128 on 4 slices +7.3%, stride 2 on 4 slices +6.6%, median of 3): the moved rows only add writer hand-over latency. Host balancing costs ~0.2 s per aspp program build (cache miss only). |
| `nslice_short_row`: the ring step's 1 x 1 short group and per-step re-init | WIN | res4.x 10.426 -> 10.126 (weights stubbed 10.107 -> 8.889) | Attribution (per-group zones, weights stubbed): 4-row group 0.42 us (at the FPU floor), 1-row group 0.275 us (~41 cycles per k-step, unpack-issue-bound), one ~52 ns re-init per step (the 1 x 1 group runs in the other ct >= rt MOP mode). Odd-owner steps ended on the 4-row group, whose pack then waited for its partial. Change (raw LLK): 1-row groups run in the 4 x 1 group's stream-in0 MOP mode with no re-init, and their single in0 tile is unpacked directly. The short group goes last for every owner. Addresses step from one CB lookup per group, with the throttle checked once per group. res5.x's 1 x 2 short group keeps its reuse-in1 order without the MOP or re-init. The recommended option is within ~0.1 us of the no-short-row ceiling (8.855 stubbed). Domain (LoFi): res4.0 -3.8%, res5.0 -4.8%, res5.1 -4.6%, res5.2 -0.9%, aspp.convs.1 -3.2%, .2 -1.3%, .3 -4.6%; HiFi4 BLOCK -2.3% to +0.3%; non-Panoptic ring shapes -3.6% to +0.4%. |

Not floated (cap 6, reason each):
- **Kernel-start spread**: not op work (above).
- **kw-fold final pass**: Perf 11 NULL (pack-bound at ~47 cycles per output tile).
- **stem.conv1 matmul** (~27 cycles per tile-MAC after its weights land, priority 15) and **stride 2** (res3.0 / res4.0, priorities 21-22): lowest priority.
- **decoder res3 / res3.x on 110 cores**: Perf 12 `kwf_n_planes` NULL; at tile-row granularity 110 cores do not lower rows per core.

### Graduated (one path each, in commit order)
1. **kw-fold compile-time subblock dims** (`c6203497d5d`). Clean apply; the "not taken" list moved from the kernel comment to this entry.
2. **Ring step short row** (`f41a671fe6a`). Clean apply.
3. **Dual-source servers** (`cbdb1086108`). Clean apply. The descriptor comment's placement history was reworded as the reason for the hop metric.
4. **Spread column stream** (`d2d6573d562`). Applied directly onto (3); the optimizer's 4 bit-exact unit tests went into `test_conv2d_3x3_block_sharded.py`.
5. **Balanced, gated BLOCK fill** (`2f28f4d31ff`). Merged with `git merge-file` onto (3) and (4); `grad/` == base + patch was checked first.
   - One writer conflict: the host-fill words now follow the spread stream's 12 round-0 words (`round0_arg + 12` under `NSLICE_SPREAD`), with the patch's gate word and window `static_assert`.
   - Transaction ids are disjoint: spread reads 2-5, async bias 7, writer fill 8-15 on NoC1, which the reader polls.
   - 6 unit tests added (one doc-line conflict).
6. **Zone hash fix** (`22f444dd244`).
- Experiment files: run logs and profiler CSVs over 500 KB stay on disk, uncommitted (large-file hook). The hooks reformatted the experiment copies (black, clang-format, trailing whitespace) at commit.

Carve-outs, each with the measured reason that earned it:
- **Dual servers only under kw-fold**: res3.0 (non-kw-fold server walk) measured +2.0% to +2.7% (median of 3, 3 options). The shape is `partners = WEIGHT_SERVER_PARTNERS and kwf and chunk >= 2`; the non-kw-fold server path is the measured exception.
- **Fill balance + gate only where kernel-row bands are disjoint**: overlapping bands measured +6.6% / +7.3% (medians of 3). The landing-count protocol runs everywhere.
- **Inexpressible, not guards**: spread stream where its slots or per-round flags do not fit (aspp.convs.1/.2, one-round streams, ragged rows, the semaphore budget); a rectangle without a spare core keeps one server; the balance model skips COL_MAJOR columns (the gate still applies) and assumes the Blackhole NoC grid.
- **Not carved out** (inside the noise band, no measurement earns a guard):
  - res5.0's spread stream (+1.9% isolated; -2.0% end to end with the short row);
  - inst dec res2 fc0 +0.6% (ranges 28.84-28.89 vs 28.99-29.07; its optimizer read +0.5%);
  - stem.conv1 +1.2% (ranges overlap; none of this round's code runs on slab-direct);
  - HiFi4 inst_res2_fuse_conv0 +0.5%, stem_conv2 +0.4%, inst_res2_fuse_conv1 +0.3%.

### Precision
- **All 22 focus cases are bit-identical to `1a04504f992`** (`ablation_r14/merge_gates/test_focus_exact.py`, one process, 0 differing elements on every case).
- No change touches the precision contract.
  - The short-row k-steps issue the same tiles and MVMULs in the same order into the same DEST tiles; only the SrcA / SrcB clear order differs.
  - The kw-fold change keeps the same MACs and pack order.
  - The rest is data movement and scheduling.

### Whole-op result (unzoned, us; whole perf file, base `1a04504f992` and final `22f444dd244`, mean of 3 alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 19.83 | 19.18 | 1.03x | 22.7 | 80% -> 83% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 83.64 | 83.09 | 1.01x | 90.8 | 76% -> 76% |
| 3 | res4.{1..5}.conv2 | 10.61 | 9.51 | 1.12x | 5.7 | 37% -> 42% |
| 4 | res3.{1,2,3}.conv2 | 9.82 | 9.26 | 1.06x | 5.7 | 40% -> 43% |
| 5 | stem.conv2 | 27.15 | 26.64 | 1.02x | 22.7 | 59% -> 60% |
| 6 | aspp.convs.2 | 41.98 | 40.91 | 1.03x | 45.4 | 76% -> 78% |
| 7 | aspp.convs.1 | 47.85 | 44.24 | 1.08x | 45.4 | 66% -> **72%** |
| 8 | res2.{0,1,2}.conv2 | 9.76 | 9.20 | 1.06x | 5.7 | 41% -> 43% |
| 9 | center_head.1 + offset_head.1 | 9.20 | 8.89 | 1.03x | 5.7 | 43% -> 45% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 23.90 | 23.55 | 1.02x | 22.7 | 66% -> 67% |
| 11 | stem.conv3 | 47.74 | 47.80 | 1.00x | 45.4 | 67% -> 66% |
| 12 | sem dec res2.fuse_conv.0 | 92.73 | 92.77 | 1.00x | 102.1 | 77% -> 77% |
| 13 | sem dec res3.fuse_conv.0 | 34.05 | 33.43 | 1.02x | 28.4 | 58% -> 59% |
| 14 | sem dec res3.fuse_conv.1 | 28.17 | 27.67 | 1.02x | 22.7 | 56% -> 57% |
| 15 | stem.conv1 | 9.73 | 9.85 | 0.99x | 1.1 | 8% -> 8% |
| 16 | inst dec res3.fuse_conv.0 | 18.89 | 18.40 | 1.03x | 14.2 | 53% -> 54% |
| 17 | inst dec res2.fuse_conv.0 | 28.87 | 29.03 | 0.99x | 28.4 | 69% -> 68% |
| 18 | res5.2.conv2 | 24.27 | 24.03 | 1.01x | 22.7 | 65% -> 66% |
| 19 | res5.1.conv2 | 26.04 | 25.32 | 1.03x | 22.7 | 61% -> 63% |
| 20 | res5.0.conv2 | 26.03 | 25.50 | 1.02x | 22.7 | 61% -> 62% |
| 21 | res3.0.conv2 | 13.06 | 13.08 | 1.00x | 5.7 | 30% -> 30% |
| 22 | res4.0.conv2 | 11.13 | 9.94 | 1.12x | 5.7 | 36% -> 40% |

All 3x3 layers of one Panoptic inference: 1109 -> 1080 us (-2.6%). At or above 70%: aspp.convs.3, sem res2 fc1, sem dec res2 fc0, aspp.convs.2, aspp.convs.1 (new). Per-case ranges: `ablation_r14/ab_results.txt`.

### Guard set (whole perf file: 41 shapes plus the 22 focus cases, unzoned, 3 alternating pairs; `ablation_r14/ab_results.txt`)
- **BLOCK (HiFi4 unless noted).** res4 LoFi -10.3%, res4 -1.9%, res4_0 -1.8%, aspp_convs1 -1.8%, aspp_convs2 -1.2%; flat: res5_2 -0.3%, res5_1 -0.1%, res5_0 +0.1%, aspp_convs3 +0.2%.
- **HEIGHT (HiFi4).** res3 -0.7%, res2 / inst_res3_fuse_conv0 -0.6%, sem_res3_fuse_conv0 / conv1 -0.3%, center_head1 -0.2%, stem_conv1 -0.1%; flat-positive: stem_conv3 0.0%, sem_res2_fuse_conv0 0.0%, sem_res2_fuse_conv1 +0.1%, res3_0 +0.2%, inst_res2_fuse_conv1 +0.3%, stem_conv2 +0.4%, inst_res2_fuse_conv0 +0.5%.
- **Interleaved (18 shapes).** -0.9% to +0.5%. No change of this round runs on the interleaved path.
- **Golden.** `eval/golden_tests/conv2d_3x3/`: 1518 passed, 480 skipped (INVALID), 0 failed, on the final tree.
- **Unit dir.** 333 / 333 (10 new: 4 spread stream, 6 fill balance). Each graduation was gated on its own files first: kw-fold + HEIGHT 72/72 (1), BLOCK + stride-2 76/76 (2), kw-fold + HEIGHT + stride-2 80/80 (3), BLOCK 72/72 (4), BLOCK 78/78 (5).
- **`--dev`.** BLOCK, kw-fold, HEIGHT, stride-2 and layouts files 227 / 227 on the final tree.

### Bottleneck now (zoned final, `ablation_r14/zones_final.txt`, `focus.txt`)
- **res4.x (42%).** Prelude to 2.6-2.9 us; walk 2.6-9.1 still ~0.8 us per step on the critical spare member: the lookahead-2 spread rounds land in pairs, with waits of 1.0-1.6 us on every second round. Then combine 0.5 and store to 9.9; the tail is pack-bound (NULL this round).
- **Small kw-fold layers (res2 43%, res3.x 43%, center_head.1 45%).** Round 0 is now complete at ~2.2-2.4 us (was 2.6-3.2). res2's side rounds take 4.2 us (pack-bound: fp32 L1-acc Q packs). The final pass takes 0.8-1.3 us.
- **aspp.convs.1 (72%).** Fill ends at 4.2 us (was 6.6) and the tap build at 8.6-9.3 (was 11.1), before a 35 us walk at the FPU floor.
- **stem.conv1 (8%), res3.0 (30%), decoder res3 layers (54-59%):** unchanged this round.
- **Carried flag (upstream):** the kw-fold fill still self-reads the core's own shard rows into its slab (Perf 12-13 notes; skipping them measured no change). This is a completeness note for the sharding refinement, not a perf lever.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckernel::matmul_block` / `compute_kernel_lib::matmul_block` | capability | Both pick the MOP mode from ct >= rt, so a 1 x 1 (or 1 x N) group after a 4 x 1 group needs a full `matmul_block_init` every ring step (~52 ns: both threads drain and re-record). Neither can run a k-step in the other mode, keep the CB lookups out of the per-k-step call, or run a 1 x N reuse-in1 group without its MOP. The raw path uses `_llk_unpack_AB_matmul_` / `_llk_math_matmul_` with `reuse_a` fixed false, direct UNPACRs for the 1-row in0 tile, and replayed MVMULs for the wide group. | 10467 (res4.x; weights stubbed 10107) | 10105 (8889) | conv2d_3x3_compute.cpp:433 (justification), :454, :1584 |
| `dataflow_kernel_lib` `SenderPipe` / `ReceiverPipe` (multicast channel) | capability | One receiver set cannot be fed by two senders on two NoCs with a completion lane per sender, and slot-reuse acks cannot be forwarded to a second sender. The partner's byte-lane flag multicasts and the forwarded acks are plain NoC calls inside the existing raw server protocol (Perf 6 bypass). | 9753 (res3.x, one raw server per rectangle; no helper variant exists for the server protocol) | 8986 | conv2d_3x3_weight_server_writer.cpp:21 |
| `dataflow_kernel_lib` multicast channel (`SenderPipe::send`) | capability | The channel has one fixed sender and a per-round CONSUMER_READY handshake. The spread stream needs a different sender per round, no handshake, and a flag semaphore per round. | 10521 (res4.x, handshake stream through the channel) | 10101 | conv2d_3x3_writer.cpp:106 (justification), :1685 |
| `TensorAccessor` (ND-sharded page -> NoC address) for a weight round | capability | No precomputed shard -> bank form: each round's address arithmetic took ~0.15-0.2 us of RISC time on a sender whose rounds are serial. The writer computes bank and offset from the shard index (`get_noc_addr_from_bank_id`) where the descriptor confirms round-robin placement. | (included in the row above) | (included) | conv2d_3x3_writer.cpp:1701 |
| no helper (cross-RISC NoC read completion) | capability | No helper or API waits for another RISC's NoC reads. The reader polls the writer's NoC1 transaction-id counters (`ncrisc_noc_read_with_transaction_id_flushed(1 - noc_index, ...)`) after the writer publishes its issued count, so a writer busy with weights never has to serve the fill. | not separated (the whole balance + gate change: aspp.convs.1 47780) | 45640 | conv2d_3x3_reader.cpp:870 (justification), :893 |

The kw-fold compile-time dims change still calls `matmul_block` and `matmul_block_init`: no helper is bypassed. Its re-init skip depends on an undocumented property of the BH LLK: the matmul MOPs depend on the subblock dims only through ct >= rt. No API exposes "does this dims change need a re-init", so the skip is written against the LLK (`llk_math_matmul.h`, `llk_unpack_AB_matmul.h`) and noted in conv2d_3x3_kw_fold.hpp's file comment.

## Perf 15
- Date: 2026-11-20
- Scope: perf tournament (run round 4 of 4) on the 22 `[perf]` LOOSE cases (LoFi, fp32 DEST, bf8_b activation and weights except stem.conv1, bias + ReLU except aspp.convs.3, pinned shard grids). Goal: >= 70% compute utilization against an ideal on 110 cores. No SUPPORTED change. Base `ec6f75b0d28` (end of Perf 14). Board: Blackhole p150b, 11x10 = 110 cores, AICLK 1350 MHz. DEVICE KERNEL DURATION.
- Outcome: 6 ideas floated (cap 6). 2 WINs graduated, 1 off-focus WIN graduated, 3 NULL.
  - All 3x3 layers of one inference: 1083 -> 1069 us (-1.3%; whole perf file, 3 alternating pairs).
  - res3.x -5.2% (9.26 -> 8.78 us), inst dec res2 fc0 -3.1%, stem.conv3 -2.8%, stem.conv2 -2.7%, inst res2 fc1 -2.7%, sem dec res3 fc1 -2.1%.
  - **inst dec res2 fc0 crosses 70%** (68% -> 71%).
  - All 22 focus cases are bit-identical to the base.
- Instrumentation: the Perf 1-14 zones are kept; none added or removed (127 zone locations, 0 hash collisions on every graduated tree). The new paths run inside existing zones: the face-split final pass in `compute_kwf_final`, the face-split Q packs in `compute_kwf_matmul_shift` / `compute_kwf_side_rounds`, the partner's round-0 read in `server_wait_read`.
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r15/`.
  - `ablation_hooks.patch` is the round-14 patch re-cut for `ec6f75b0d28`, with three fixes:
    - the `matmul` stub now covers the ring k-steps (`ring_ksteps`);
    - the partner weight servers (Perf 14) did not get the ablation defines, so the `weights` stub hung sem dec res3 fc0 (servers waiting for acks the stubbed receivers never sent);
    - the `weights` stub left the BLOCK spread stream (Perf 14) running. The first BLOCK peel read 8.6-18 us "everything stubbed"; the corrected stub reads 5.4-8.3 (the rows below).
  - Not covered: the slab-direct NoC0 round 0 (stem.conv1, as in Perf 14).
  - `merge_gates/test_focus_exact.py` is the bit-exact gate (current tree vs `op_base/` = `ec6f75b0d28`); it needs `--import-mode=prepend` under `run_safe_pytest.sh`.

### Measured breakdown (before, zoned, us)
DM-first cumulative peel on `ec6f75b0d28`: full > gather > fill > weights > scatter > untilize > tilize > matmul (everything stubbed). "mm only", "fill only", "weights only" stub that stage alone. All in `ablation_r15/peel_results.txt` (the `*2` rows are the corrected weights stub); zones in `zones_full.txt`.

| case | full | -gather | -fill | -weights | -scatter | -untilize | -tilize | -matmul | mm only | fill only | weights only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stem.conv1 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.1 | 10.0 | 10.0 |
| stem.conv2 | 27.0 | 27.0 | 26.9 | 25.8 | 25.8 | 25.8 | 25.8 | 19.5 | 19.6 | 26.8 | 26.5 |
| stem.conv3 | 48.1 | 48.1 | 48.1 | 45.9 | 45.9 | 45.9 | 45.9 | 36.8 | 39.2 | 48.1 | 47.3 |
| res2 | 9.9 | 9.9 | 10.0 | 9.5 | 9.5 | 9.5 | 9.5 | 7.1 | 7.3 | 10.1 | 9.9 |
| res3.0 | 13.2 | 13.3 | 12.1 | 11.9 | 11.9 | 11.9 | 9.1 | 3.7 | 9.6 | 12.1 | 13.1 |
| res3.x | 9.6 | 9.4 | 9.3 | 8.8 | 8.8 | 8.8 | 8.8 | 5.7 | 8.7 | 9.4 | 8.9 |
| res4.0 | 10.3 | 10.2 | 9.9 | 9.3 | 9.3 | 9.3 | 8.2 | 5.4 | 9.4 | 9.9 | 10.3 |
| res4.x | 9.9 | 9.8 | 10.3 | 8.7 | 8.8 | 8.8 | 8.5 | 6.1 | 9.2 | 9.7 | 9.2 |
| res5.0 | 25.5 | 25.8 | 25.9 | 23.8 | 23.9 | 23.8 | 22.9 | 7.9 | 17.6 | 25.6 | 24.7 |
| aspp.convs.1 | 45.2 | 45.1 | 44.1 | 43.6 | 43.4 | 43.4 | 39.2 | 9.0 | 32.0 | 43.5 | 44.4 |
| aspp.convs.3 | 19.5 | 19.3 | 19.3 | 17.2 | 17.1 | 17.1 | 16.6 | 8.3 | 15.6 | 19.3 | 17.8 |
| sem dec res3 fc0 | 34.1 | 34.1 | 34.0 | 32.4 | 32.5 | 32.4 | 32.3 | 12.4 | 18.8 | 34.0 | 33.0 |
| inst dec res2 fc0 | 29.7 | 29.3 | 29.1 | 27.5 | 27.5 | 27.5 | 27.5 | 10.8 | 11.9 | 28.8 | 28.5 |
| inst res2 fc1 | 24.5 | 24.2 | 23.7 | 22.8 | 22.8 | 22.8 | 22.8 | 10.7 | 11.1 | 24.0 | 23.7 |
| center_head.1 | 9.5 | 9.5 | 8.7 | 8.4 | 8.4 | 8.4 | 8.4 | 4.9 | 6.0 | 8.6 | 9.4 |

Ranked by gap to target x calls (unzoned base of this round, us per inference): res4.x 18.8, decoder res3 layers 13.9 (three layers), res3.x 13.4, res2 11.3, stem.conv1 8.7, res3.0 7.3, center_head.1 6.4, res4.0 4.2, stem.conv2 3.8, the rest under 3.
1. **res4.x walk.** ~0.68 us per ring step with the weights stubbed, against ~0.53 us of FPU work. Everything stubbed, the ring alone costs ~0.34 us per hop, and Perf 12 had measured pack onto the partial ~0.3 + write ~0.33 us, so the step looked chain-bound. Weights-only stub -0.7 us (co-bound). `NSLICE_SPREAD_LOOKAHEAD` checked inline (unzoned, one pass): 3 flat on res4.x and +8% on res5.x, 4 +20% on res4.x. More reads in flight lengthen every read, so the stream is not lookahead-bound.
2. **kw-fold round 0 on large rounds.** Round 0 complete on the receivers at p50 3.0 us on inst res2 fc1 (52 KB), ~4.1 on inst dec res2 fc0 (65 KB), ~5.1 on sem res2 fc1 (104 KB). The partners' half is late: they start ~0.35 us after the first servers and read their 26 KB on NoC1 in 1.7-1.9 us (first servers 1.1-1.2).
3. **kw-fold final pass, serial after each n-group / M-block.** stem.conv2 3.0 us, stem.conv3 2 x 3.0, inst res2 fc1 2.1, res2 1.3, res3.x 1.2. Perf 11 found it pack-bound at ~47 cycles per tile on the BH tilize unswizzle.
4. **res3.0.** The writer waits 3.5 us for round 2 on the critical core; both servers sit in `server_wait_ack` 4.5 us p50 (slot reuse).
5. **stem.conv1.** The round-0 multicast starts at 2.12 us behind 108 kernel-start acks; weights land p50 3.0 / max 3.75 us.

Roofline gate, as in Perf 8-14: the decoder walks, aspp's walks and res3.x's walk (on 86 cores) are at the FPU floor (~18.5-19 cycles per tile-MAC at 4-tile fp32 subblocks). Far from it: every start-up above, the kw-fold packer (Q packs plus final pass), res4.x's step.

### Portfolio and verdicts
Each idea was built and measured in isolation by its own optimizer (`perf_experiments/<slug>/`). Numbers are unzoned medians of 3 alternating runs unless noted.

| idea | verdict | focus result (us, isolated) | domain / notes |
|---|---|---|---|
| `ring_tile_pipeline`: N-slice ring partial forwarded per tile (count, pack, token and write per 1 or 2 tile-rows) | NULL | res4.x 9.330 -> 9.233 (per 2 rows, -1.0%); per row 9.595 (+2.8%, regression: each token push drains the packer) | Attribution (zoned with `tensix_sync`-closed per-group zones, weights stubbed): the packer waits ~0.05 us per group for the incoming partial and math never waits for DEST, so the step is not chain-bound. A ceiling with no arrival wait at all (wrong sums) was flat (9.248 / 9.279 -> 9.170 / 9.276; stubbed 9.024 -> 9.084). The step is math plus the DEST-half cycle: the 1-row group is too short to hide the 4-row group's pack. 3 + 2 row groups cut the 5-row walk 5.44 -> 5.12 us (weights stubbed) but not the op (spare-column store tail and start-up are equally critical) and need the kernel-row masks reworked. Wins only on a chain-bound synthetic ring (11 columns, 5 owners: 10.68 -> 9.17). aspp.convs.2 with the patch +3.9% (bimodal, below). Not graduated. |
| `block_wstream_latency`: per-round latency of the BLOCK spread column stream | NULL (hidden win) | res4.x 9.274 -> 9.221 (-0.6%); matmul stubbed 8.20-8.26 -> 7.72 | Attribution: the stream paces only the 24 spare-column members, one relay multicast (~0.45-0.55 us after the round lands on the column sender) further from DRAM; the grid rows are ring-paced (stream stubbed: grid max 9.32 vs 9.17 zoned, spare members 9.93 -> 9.16). Arrival of every round j + (8 - j) x 0.7 us = the spare walk's end, so only moving all arrivals helps. Best option (ordered multicasts, link-model readers from round 2, lookahead 3) recovers ~480 of the stream's ~780 ns with the matmul stubbed; res5.x +1.2% to +1.5% (in band). Rejected: the reader multicasting to the spare row too (+117%), unicast relays (+8% / +10%), spare rows reading DRAM (+12%), relay core reading every round (+5.7% to +7.9%), early first reads (+1.2%). Not graduated: no ring change to unlock it (above). |
| `kwf_round0_sources`: land the large kw-fold round 0 sooner | WIN (small) | inst res2 fc1 23.94 -> 23.56 (-1.6%), inst dec res2 fc0 29.01 -> 28.42 (-2.0%), pooled sessions; res3.x flat | Ceiling (round 0's DRAM reads and multicasts skipped, wrong output): inst res2 fc1 -3.0%, res3.x -3.7%, inst dec res2 fc0 -2.8%, sem res2 fc1 -0.4%; the slowest cores then wait on their fill to ~2.9 us (round 0 and the fill share each receiver's NoC). Change: the partner's writer reads the partner's part of round 0 on NoC0 at kernel start, and the first server takes 7/12 of round 0's slot rows. Zoned on inst res2 fc1: round 0 complete p50 3.50 -> 2.54 us. The optimizer gated both by a host route model (partner's NoC1 responses crossing receivers). Each half alone hurt the 86-core layers (NoC0 read: res3.x +2.9%; share: +0.5% to +1.4%), both together flat (-0.7%). Rejected: 3:1 split (res3.x +1.4%), partner kernels first (flat), other spare cores on 86 cores (+1% to +3%). Bit-identical. |
| `kwf_final_facesplit`: kw-fold final pass off the BH tilize path | WIN | stem.conv2 26.57 -> 25.98, stem.conv3 47.73 -> 46.47 | The Perf 11 null's bound (pack at ~47 cycles per tile on the tilize unswizzle) removed by a new ACC layout: each fp32 plane is two 16-column half-planes, so a tile's faces 0 / 2 and 1 / 3 are 32 contiguous rows each. Q packs: two replayed streams of 4-row PACRs (16 per tile instead of 32; the packer is PACR-issue-bound). Isolated cycles per tile (overwrite / L1 acc): 4 x 1 73 / 73 -> 43 / 48, 1 x 4 82 / 82 -> 49 / 53, fused side row 86 / 86 -> 54 / 59; matmul stubbed, the stem.conv2 walk 15.5 -> 12.9 us. Final pass: two 2-face unpack runs per tile (one config context per run of a plane's tile-rows in one image row), ELWADD with DEST row immediates and srcB row-broadcast bias, Default pack: 3.0 -> 2.6 us on stem.conv2, now pack-bound at ~42 cycles per tile. Domain: inst res2 fc1 23.97 -> 23.44, sem dec res3 fc1 27.64 -> 27.01, inst dec res2 fc0 28.92 -> 28.58, center_head.1 8.83 -> 8.67 (medians of 3); single runs res2 -4.0%, sem dec res3 fc0 -2.1%, inst dec res3 fc0 -4.0%, sem dec res2 fc0 -1.0%, sem res2 fc1 -1.2%; res3.x flat; HiFi4 HEIGHT kw-fold flat to -1%. Rejected: per-tile unpack contexts (unpack-bound), 16-row bias rows (the reader's decode doubled: sem dec res3 fc0 +2.4 us), the right stream placed by the packer's ch1 counter (**incorrect**). Bit-identical. |
| `s2_weight_slots`: res3.0's walk waiting on weight-slot reuse | NULL | res3.0 12.950 -> 12.969 (all rounds resident, +0.15%) | Attribution: the stall is real on fast cores (round 2 lands 10.4 -> 5.5-6.2 us zoned when resident), but the slowest core starts its walk at 6.04 us, frees slot 0 at 8.64 and needs round 2 at 10.38, after it lands (10.31). The servers' round-2 delivery (~1.7-1.8 us) is shorter than one K-block (1.9 us). Weights-only stub ceiling 0.1 us. The bound is the slow cores' walk start: their fill ends 2.6-3.7 us (x = 1-2 cores also start ~0.4 us late), and the tap build runs ~2.7-3.0 us past it. Smaller rounds are inexpressible (a K-block under one kernel row turns off the DEST taps). No patch. |
| `stem1_r0_noack`: land stem.conv1's weights sooner | NULL on the focus; WIN off it | stem.conv1 9.79 -> 9.77-9.82 (flat); 64x128x8 s2 3.342 -> 2.984 (-10.7%, every candidate run below every base run) | Attribution: the critical core (3,11, last grid row) gets its weights at 3.6-3.7 us but its first slab chunk only at 3.9-4.2 us (its writer's NoC1 forward reads share the row's westward links), then runs a fixed 5.82 us matmul (304 tile-MACs with the fused bias step, ~25.8 cycles each). Receivers never waiting for weights (diagnostic): 9.69 us. On stem.conv1 the acks were in before the go; on small shapes up to 63 kernel-start NoC1 atomics delayed the sender's own DRAM read past its writer's first landing poll. Change: no ack semaphore, receivers only reserve their slot. Rejected: the last row's rectangle first (64x128 +2.3%), multicasting before the bias split (flat, bandwidth-bound at ~19 B/ns). Domain: HiFi4 64x128 -11.1% (one run), 256x512 s2 medians -8.4% (the base hit a slow mode in 2 of 4 runs: flat to win), the rest flat. Bit-identical. |

Not floated (cap 6, reason each):
- **gw = 2 Q_1 as 2 x 2 at compile-time dims**: Perf 7 measured 2 x 2 much slower (runtime dims then); lower expected value than the six.
- **decoder res3 / res3.x on 110 cores**: Perf 12 `kwf_n_planes` NULL; a K-split pair scheme is beyond one round.
- **res4.x ring tail**: Perf 14 NULL.
- **kw-fold first MAC on small rounds**: within ~0.3 us of the DRAM-latency floor.
- **stem.conv3 at one M-block** (ACC 674 KB): does not fit the `RESIDENT_SHARD_SETS = 2` budget, a model-chaining contract, not a lever.

### Graduated (one path each, in commit order)
1. **Face-split kw-fold ACC** (`4c61f988138`). Clean apply. Two lines of the file comment that narrated the replaced path were rewritten. The optimizer's exact bias-placement test went into `test_conv2d_3x3_kw_fold.py` (`test_conv2d_3x3_kw_fold_bias_layout`, 4 cases: zero weights, an integer bias per channel, exact compare).
2. **Partner reads round 0** (`d0188e3bd00`). Graduated as one unconditional path, not as delivered.
   - The patch gated the NoC0 read and the 7/12 share by a host route model (`_r0_partner_self_read`).
   - Coordinator A/B, unzoned, 3 alternating pairs, rule vs every partner self-reading with the share: within ±0.7% on all 22 (inst dec res3 fc0 +0.7%, res3.x +0.3%, inst res2 fc1 -0.2%, the BLOCK cases noise).
   - No measured regression earns the guard, so the predicate, its on/off constant, the always-1 runtime flags and the partner reader's round-0 path were deleted.
   - The unit test now checks that the 7/12 share and the halves give bit-identical output (`test_conv2d_3x3_kw_fold_round0_split`, 8 cases).
3. **Slab-direct round 0 without acks** (`1f38d5ef099`). Clean apply, plus the optimizer's 64x128 case in the whole-grid slab-direct test. The slab-direct and face-direct tests ran 3 times in non-dev mode (the Perf 13 deadlock only showed outside `--dev`) and once under `--dev`.
- Experiment files: global `import torch` in three experiment test files moved into the functions (repo import check). Run logs and profiler CSVs over 500 KB stay on disk, uncommitted (large-file hook).

Carve-outs, each with the measured reason that earned it:
- **None earned by a measured regression.** The round-0 route rule was removed (above).
- **Inexpressible:** face-split planes over 8191 rows (the unpacker's 16-bit x_dim; no kw-fold shape that fits L1 reaches it). The round-0 share needs a partner (a rectangle without a spare core keeps one server) and at least two slot rows per chunk.

### Precision
- **All 22 focus cases are bit-identical to `ec6f75b0d28`** (`ablation_r15/merge_gates/test_focus_exact.py`, run after each graduation, 0 differing elements on every case).
- No change touches the precision contract. The face-split final pass issues the same ELWADDs of the same tf32 operands as the tilize copy, and the Q packs write the same fp32 values (L1 accumulation in the same order). The other two changes are data movement.

### Whole-op result (unzoned, us; whole perf file, base `ec6f75b0d28` and final `1f38d5ef099`, mean of 3 alternating; target = t_ideal / 0.7; utilization = t_ideal / measured)

| prio | case | before | after | speedup | target | util before -> after |
|---|---|---|---|---|---|---|
| 1 | aspp.convs.3 | 19.55 | 19.19 | 1.02x | 22.7 | 81% -> 83% |
| 2 | sem res2.fuse_conv.1 + head.{0,1} | 83.24 | 82.53 | 1.01x | 90.8 | 76% -> 77% |
| 3 | res4.{1..5}.conv2 | 9.43 | 9.41 | 1.00x | 5.7 | 42% -> 42% |
| 4 | res3.{1,2,3}.conv2 | 9.26 | 8.78 | 1.05x | 5.7 | 43% -> 45% |
| 5 | stem.conv2 | 26.64 | 25.92 | 1.03x | 22.7 | 60% -> 61% |
| 6 | aspp.convs.2 | 41.22 | 40.76 | 1.01x | 45.4 | 77% -> 78% |
| 7 | aspp.convs.1 | 44.13 | 44.22 | 1.00x | 45.4 | 72% -> 72% |
| 8 | res2.{0,1,2}.conv2 | 9.26 | 9.16 | 1.01x | 5.7 | 43% -> 43% |
| 9 | center_head.1 + offset_head.1 | 8.88 | 8.77 | 1.01x | 5.7 | 45% -> 45% |
| 10 | inst res2.fuse_conv.1 + heads.0 | 23.88 | 23.24 | 1.03x | 22.7 | 67% -> 68% |
| 11 | stem.conv3 | 47.80 | 46.47 | 1.03x | 45.4 | 66% -> 68% |
| 12 | sem dec res2.fuse_conv.0 | 92.91 | 91.67 | 1.01x | 102.1 | 77% -> 78% |
| 13 | sem dec res3.fuse_conv.0 | 33.53 | 33.28 | 1.01x | 28.4 | 59% -> 60% |
| 14 | sem dec res3.fuse_conv.1 | 27.55 | 26.98 | 1.02x | 22.7 | 58% -> 59% |
| 15 | stem.conv1 | 9.74 | 9.72 | 1.00x | 1.1 | 8% -> 8% |
| 16 | inst dec res3.fuse_conv.0 | 18.39 | 18.20 | 1.01x | 14.2 | 54% -> 55% |
| 17 | inst dec res2.fuse_conv.0 | 29.07 | 28.16 | 1.03x | 28.4 | 68% -> **71%** |
| 18 | res5.2.conv2 | 24.44 | 24.30 | 1.01x | 22.7 | 65% -> 65% |
| 19 | res5.1.conv2 | 25.14 | 25.14 | 1.00x | 22.7 | 63% -> 63% |
| 20 | res5.0.conv2 | 25.25 | 25.26 | 1.00x | 22.7 | 63% -> 63% |
| 21 | res3.0.conv2 | 13.05 | 12.99 | 1.00x | 5.7 | 30% -> 31% |
| 22 | res4.0.conv2 | 9.89 | 9.90 | 1.00x | 5.7 | 40% -> 40% |

All 3x3 layers of one Panoptic inference: 1083 -> 1069 us (-1.3%). At or above 70%: aspp.convs.3, sem res2 fc1, sem dec res2 fc0, aspp.convs.2, aspp.convs.1, inst dec res2 fc0 (new). Base and final ranges separate on every kw-fold focus case (res3.x 9.22-9.29 vs 8.70-8.95); the BLOCK cases, stem.conv1 and res3.0 overlap (no change of this round runs there except res3.0's server refactor and stem.conv1's ack removal). Per-run values: `ablation_r15/ab_results.txt`, report `ab_report.txt`.

### Guard set (whole perf file: 41 shapes plus the 22 focus cases, unzoned, 3 alternating pairs; `ablation_r15/ab_report.txt`)
- **HEIGHT (HiFi4).** res2 -1.4%, stem_conv2 -1.0%, stem_conv3 -0.9%, stem_conv1 -0.6%, sem_res3_fuse_conv1 / inst_res2_fuse_conv1 -0.5%, sem_res3_fuse_conv0 -0.4%, res3 / sem_res2_fuse_conv0 / inst_res2_fuse_conv0 -0.3%, sem_res2_fuse_conv1 / center_head1 -0.2%, res3_0 -0.1%, inst_res3_fuse_conv0 0.0%.
- **BLOCK (HiFi4 unless noted).** -0.2% to +0.4% (res5_1 +0.4%, ranges overlap). No change of this round runs on BLOCK.
- **Interleaved (18 shapes).** -2.2% to +0.4%: H16_W16_Ci64 tile_bf16 -2.2% (the layout-noise cell of earlier rounds), H64_W128_Ci128 tile_bf16 +0.4% (51.31-51.55 vs 51.54-51.86), the rest within ±0.7%.
- **Golden.** `eval/golden_tests/conv2d_3x3/`: 1518 passed, 480 skipped (INVALID), 0 failed, on the final tree.
- **Unit dir.** 347 / 347 (13 new: 4 kw-fold bias layout, 8 round-0 split, 1 slab-direct whole grid). Each graduation was gated on its own files first: kw-fold + HEIGHT 76/76 (1); kw-fold + HEIGHT + stride-2 92/92 and `--dev` kw-fold 37/37 (2); HEIGHT + layouts + stride-2 126/126, slab / face-direct `--dev` 18/18 and non-dev 3 x 18/18 (3).
- **`--dev`.** BLOCK, kw-fold, HEIGHT, stride-2 and layouts files 241 / 241 on the final tree.

### Bottleneck now (zoned final, `ablation_r15/zones_final.txt`)
- **res4.x (42%).** Its ring step is not chain-bound (`ring_tile_pipeline`): math plus the DEST-half cycle of the 4 + 1 row groups. The spare-column members are one relay multicast further from DRAM. Prelude to 2.6 us; tail 8.6-9.9.
- **res3.x (45%).** First MAC ~2.3 us, walk to 8.2 at the 86-core FPU floor, final pass 8.2-9.3.
- **kw-fold final pass.** Now pack-bound at ~42 cycles per tile: stem.conv2 2.5 us, stem.conv3 2 x 2.5, inst res2 fc1 1.6 (zoned).
- **stem.conv1 (8%).** Its last grid row waits for its first slab chunk (~3.9 us), then runs a 5.8 us matmul at ~25.8 cycles per tile-MAC (vs ~18.5 on the wide walks): the two next levers.
- **res3.0 (31%).** Set by the slow cores' walk start (fill + tap-build tail; the x = 1-2 cores also start ~0.4 us late), not by the weights.
- **aspp.convs.2 is bimodal** (found by `ring_tile_pipeline`): on grid row y = 4, column 10 waits 1.3-3.9 us for its ring credit from column 0, whose slab dies late (credit issued at 8.4-9.6 us). In 4 of 12 base runs the op takes 41.2-42.6 us instead of 39.6-40.1. Not attributed further.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` and the pack-untilize helpers (`pack_untilize_dest`, `llk_pack_untilize`) | capability | No helper packs DEST tiles as rows at an arbitrary row offset into two half-planes (left / right 16 columns), unpacks a tile from two 2-face runs at computed addresses, or adds a srcB row-broadcast operand while doing so. On Blackhole the helpers' non-8-bit tilize leaves tile row r in srcA rows 2r / 2r + 1, so the pack must unswizzle with 32 PACRs per tile (~47 cycles per tile, the final pass's bound). The replaced path was already raw (Perf 9-11: a PACR untilize stream plus a patched tilize MOP); no helper form of either exists to measure. | 26570 (stem.conv2, the replaced raw path) | 25980 | conv2d_3x3_kw_fold.hpp:81 (justification), :212 (`pack_fst`), :264 (`fs_unpack_run`), :302 (`fs_math_tile`) |

The partner's round-0 read uses raw transaction-id reads and polls inside the existing raw weight-server protocol (Perf 6 / 14 bypass; its page-run logic moved into `kernels/conv2d_3x3_server_pages.hpp` so the server reader and the partner writer share it): no new helper gap. The slab-direct change removes code. The Perf 13 bypass row for that multicast ("its own ack count and ready flag") now holds only the ready flag and the local go flag.

## Refinement 7 — Per-side padding (padding_mode "asymmetric") for FLUX.2, interleaved path
- Date: 2026-11-20
- What was done:
  - `SUPPORTED["padding_mode"]` gains `"asymmetric"` (per-side `(top, bottom, left, right)`, each side 0 or dilation). Sharded x asymmetric is two EXCLUSIONS cells (HEIGHT, BLOCK): those paths' fast gathers are built on `padding == dilation` and no model runs per-side padding sharded.
  - `Conv3x3Geometry.padding` holds the canonical form (int when the four sides agree, else the 4-tuple), with `pad_top` / `pad_left` properties. Every coordinate site, host mirror and kernel, takes the matching side: rows `h_in = ho*s + kh*d - top`, columns `w_in = wo*s + kw*d - left`, flat first tap `f = (ho*s - top)*W + wo*s - left`. `bottom` / `right` only set `H_out` / `W_out` and the right-edge reach (slab-direct `over` / `gap`, strided-window slot sharing).
  - Kernels: reader CT 9 / writer CT 27 are now `pad_top`; `pad_left` is appended (reader CT 76, writer CT 74; the accessor args moved up one). `RowGather`, `TapDirect`, `TapDirectPlanned`, `TapDirectFifo`, `WindowGather`, `FaceGather`, `block_footprint`, `ring_keep_lo` and `slab_direct::Geom` take both; `tap_edge_canon` takes `pad_left`. `SD_PAD` became `SD_PAD_TOP` / `SD_PAD_LEFT`. The symmetric-only paths (S2_TAPS, center taps, N-slice center taps, padded slab, tap shift, kw-fold, the K split cost model) keep their int-padding gates, which a tuple never matches; their static_asserts check both sides.
  - Int padding passes the same value to both sides, so symmetric calls compile to the same code.
  - `test_flux_conv3x3_compare.py` runs the real per-side call (`FLUX_CMP_PROXY=1` keeps the symmetric proxy, `FLUX_CMP_SKIP_TTNN=1` skips ttnn.conv2d). `perf_experiments/asym_padding/`: `cmp_windows.py` (median device ns per signpost window of a Tracy CSV, dropping the precompile collect pass's empty windows) and `panoptic_hashes.py` (sha256 of every Panoptic LOOSE output, run under tt-probe).
- Accuracy achieved: PCC 0.999999 on all 17 bh2x4 `(0, 0, 1, 1)` cases and the 6 encoder downsamplers `(0, 1, 0, 1)` s2 (band PCC, test_flux_conv3x3_compare.py); golden tolerances on the 72 interleaved test_op cells and 20 FLUX.2 LOOSE_CASES; unit tests PCC >= 0.999 (bf16) / 0.99 (bf8_b) on 8 per-side shapes.
- Performance (Blackhole p150b, HiFi4, bf16, device kernel ns, median of 3, `--profile`):
  - The 17 FLUX.2 bh2x4 `[perf]` cases with the real `(0, 0, 1, 1)` call: -0.44% to +0.72% of `measured_ns_blackhole` (the symmetric proxy at 9029c93c187). Same output and MACs; the halo rows replace zero rows on the same tap-major / ring walk.
  - Panoptic: 45 LOOSE outputs bit-identical before / after (sha256). The 22 compare cases, before -> after: -1.68% (aspp.convs.1) to +1.75% (res5.1), with the same kernels, so this is noise.
  - Encoder downsamplers (no bar), real vs symmetric proxy on the same tree: 1024² x 128 2439589 vs 2397110 (+1.8%), 512² x 256 1497838 vs 1323753 (+13.2%), 256² x 512 1334104 vs 1325598 (+0.6%), 2048² x 128 8302450 vs 7086259 (+17.2%), 1024² x 256 5756954 vs 4946511 (+16.4%), 512² x 512 5147296 vs 5128505 (+0.4%). The proxy takes S2_TAPS (p == 1), and left pad 0 falls to the strided window.
- Golden test progress: targeted slices only. The 72 interleaved asymmetric test_op cells pass (24 INVALID skipped). The 20 asymmetric LOOSE_CASES pass, both dispatches, with a program-cache hit. The 144 sharded asymmetric cells xfail (EXCLUSIONS). 120 prior-passing interleaved test_op cells (`p1_s2`, `p0_s1`) pass. Expected full suite: 1604 + 92 = 1696 passed, 236 - 92 = 144 xfailed, 576 skipped.
- Issues encountered: none in correctness. S2_TAPS cannot take left pad 0 by coordinate algebra. Its kw = 2 tap needs the next input tile's pixel 0, a new DEST move / packer window in raw LLK, so the stride-2 downsamplers keep the strided window (finding above).
- Tests added: `tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_asymmetric_padding.py` (8 per-side shapes x {bf16 TILE, bf8_b TILE, bf16 RM}: each side alone and in pairs, stride 2 window and row gather, dilation 2, odd W, a multi-block (0, 0, 1, 1); plus the sharded refusal). Unit directory: 347 passed / 17 skipped before the new file, 25 / 25 in it.

## Perf 16
- Date: 2026-11-20
- Scope: perf tournament (this run's round 1 of 5) on the 17 FLUX.2 Blackhole (2, 4) per-device decoder convs (feature_spec block `attention: PERF FOCUS (FLUX.2)`): bf16 activations and weights, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1), DRAM-interleaved ROW_MAJOR input. All 17 run the interleaved K-blocked tap-direct walk (R0) on 110 cores (11x10), Blackhole p150b, 8 DRAM banks. Hard gates: Panoptic's 22 `[perf]` cases and the 17 single-device FLUX.2 LOOSE cases (TILE input) must not get slower.
- Outcome: 6 ideas floated (cap 6): 5 WIN, 1 NULL. 2 graduated (bank-run fill, weights resident). 1 WIN superseded by them (fill NoC split). 2 WINs not graduated this round (ROW_MAJOR ring slab, weight-stream stagger), with the reasons below.
  - FLUX.2 bh2x4, call-weighted over the 17 cases (calls per 1024² + 2048² decode): **209.6 -> 78.9 ms (2.66x)**. The 110-core HiFi4 ideal is 40.2 ms, so utilization is 19% -> 51%. Every case is faster, and every case is now faster than ttnn.conv2d on the same ROW_MAJOR input.
  - P1 1026x2048 32->128: 13.76 -> 2.61 ms (5.3x). P2 514x1024 64->256: 4.42 -> 1.62 ms. P3 conv_out 1026x2048 128->3: 13.78 -> 5.22 ms.
  - End to end, one BH (2, 4) 1024² VAE decode (`test_vae_flux2_conv2d_3x3.py -k "bh2x4 and 1024 and conv2d_3x3"`): 197 -> 176.9 ms, PCC 0.99919. The same decode on ttnn.conv2d takes 328.2 ms.
  - All outputs are bit-identical to the base: the 17 focus cases, 8 off-focus guard configs, 45 Panoptic LOOSE outputs.
- Instrumentation: no zone added or removed. The bank-run fill runs inside `reader_fill`. The resident-weight first-pass wait has its own zone, `compute_weight_wait`. The P1 profile still exhausts the 250-marker budget on BRISC (12+ M-blocks x 9 K-blocks), so per-stage numbers on it come from the ablation below, not the zone sums.
- Ablation hooks, scripts, raw results: `perf_experiments/ablation_r16/` (`ablation_hooks.patch` with tokens fill / weights / store / tilize / bias / matmul for this walk, `ab.sh`, `ab_results.txt`). Guard harness for arbitrary configs, with A/B sha1: `perf_experiments/graduation_r16/test_guard_cfg.py` (needs `--import-mode=prepend`).

### Measured breakdown (before, device kernel us, median of 3, cases idx35 / idx41 / idx42 / idx43 / idx47 of `test_flux_conv3x3_compare.py`)

| stubbed (cumulative) | conv_out 1026x2048 128->3 | P1 1026x2048 32->128 | P2 514x1024 64->256 | P7 258x512 128->512 | 514x1024 128->512 |
|---|---|---|---|---|---|
| none (full) | 13771 | 13728 | 4430 | 2049 | 7723 |
| fill (ROW_MAJOR DRAM reads) | 2156 | 3232 | 2106 | 1644 | 6341 |
| + weights (multicast rounds) | 1764 | 1827 | 1451 | 1334 | 4809 |
| + store | 1707 | 1560 | 1283 | 1235 | 4669 |
| + tilize | 1415 | 1461 | 1232 | 1210 | 4573 |
| + bias | 1383 | 1397 | 1199 | 1193 | 4510 |
| + matmul (everything) | 306 | 293 | 88 | 46 | 125 |
| solo: weights | 13051 | 13012 | 4017 | 1468 | 4952 |
| solo: tilize + bias + matmul | 12691 | 12783 | 3556 | 1280 | 4740 |
| solo: store | 13491 | 12704 | 3922 | 1789 | 6741 |

Ranked by measured headroom:
1. **The ROW_MAJOR fill.** Every input pixel is one 64/128/256 B DRAM page and one NoC read; with disjoint kernel-row bands (bm * 32 < W) each pixel is read ~3x per core. The reader issues the whole slab, then waits for compute to pop every tap-direct entry before refilling, so fill and compute serialize. Cost: -76% to -84% on C_in 32 and conv_out, -52% on C_in 64, -18% to -20% on C_in 128. Bound (fill_noc_split's bench): the rate at which one NoC's DRAM endpoints accept reads, ~540 M reads/s across the chip on NoC0 (~900 M on both NoCs). It is not the RISC issue loop (the same reads from local L1 run 10x faster), not the bytes (64 / 128 / 256 B pages take the same time), and not the outstanding window.
2. **The weight stream.** One fixed sender multicasts one round per K-block (cb_weights depth 2, every receiver acks every round): 108-153 lock-step rounds per call. Solo: -5% to -36% (514x1024 128->512); after the fill: -10% to -24%.
3. **The store.** It is DRAM-write-bound (P1 writes 537 MB) and sits on the same writer as the weight rounds. Solo: -7% to -13%.
4. **Matmul** runs at 93-108% of the 64-cycle/tile-MAC HiFi4 ideal (P1 matmul alone ~1.10 ms vs 1.02 ms ideal): no compute headroom. tilize and bias: < 5% each.

### Portfolio and verdicts (each idea measured end to end in its own copy of the op; base = this round's start, median of 3)

| idea | verdict | measured (base -> candidate) | precision | domain / notes |
|---|---|---|---|---|
| bank_run_fill: one DRAM read per bank per chunk (pages p, p + 8, ... sit back to back in bank p % 8), staged bank-major, de-interleaved into the slab by local NoC reads | **WIN, graduated** | P1 13.71 -> 3.64 ms, P3 13.76 -> 5.69, P2 4.44 -> 2.28, P4 2.81 -> 0.92, P5 14.81 -> 5.57, P6 17.47 -> 8.81, P7 2.05 -> 1.68, 514x1024 128->512 7.72 -> 6.39 | bit-identical (17 cases, sha1) | Bank runs reach ~410-420 GB/s aggregate (the fill becomes bandwidth-bound). NoC de-interleave ~24 NCRISC cycles per pixel, hidden except on P3 (0.23 ms). Rejected options: TRISC tilize + wide pack_untilize de-interleave (2 passes, the BH bf16 unpack-tilize leaves DEST row-swizzled; +0.41 ms TRISC on the compute-bound C_in 128 cases), RISC word copies (~60 cycles per 64 B), a dedicated 64 KB staging buffer (changed one shape's bk: not bit-identical). |
| rm_ring_slab: ROW_MAJOR ring slab across M-blocks (each pixel read once per core), the next block filled under compute | **WIN, not graduated** | per-pixel reads: P1 13.77 -> 6.55 ms, P3 13.80 -> 5.63, P2 4.42 -> 2.70, P5 14.82 -> 7.03, P6 17.47 -> 10.12, 514x1024 128->512 7.73 -> 6.69 | bit-identical | Cuts reads per core 1.6-2.5x (P1 57288 -> 23170). It conflicts with the graduated bank-run fill as built: the overlapped ring fill keeps cb_im2col_rm live (where the bank-run staging lives) and refills in 8-pixel steps, against the >= 2 KB bank reads the bank-run fill needs. The fusion (bank-run reads into ring ranges, dedicated staging in the solve, coarser overlapped steps) is the next round's top candidate. After the graduations, P3 (5.22 ms), P5 and P6 are bound by the 3x DRAM re-read the ring removes. |
| fill_noc_split: the per-pixel fill split between reader (NoC0) and writer (NoC1) | WIN, superseded | P1 13.72 -> 9.23 ms, P3 13.77 -> 8.82, P2 4.43 -> 3.42, 514x1024 128->512 7.73 -> 7.09 | bit-identical | Established the bound (row 1 above). The bank-run fill made the fill bandwidth-bound, so the split has nothing left to spread; not graduated. |
| weights_resident (op_design R5): the whole [K, N] multicast once at kernel start, compute re-reads it per M-block (matmul_block in1 NoWaitNoPop + In1BaseOffsetFn), the solve shrinks bm to fit it | **WIN, graduated** | before the bank-run fill: P1 13.73 -> 13.08 ms (-4.8%), P2 -9.3%, P5 -7.0%, P6 -8.1%. With the fill stubbed: P1 3.23 -> 1.82, P2 2.11 -> 1.42 (the full weights-ablation value) | bit-identical where bk is kept (every focus case) | Not taken where [K, N] does not fit at any bm (258x512 / 514x1024 / 130x256 128->512: 1.15 MB), or with one M-block or one K-block per core. Its one measured regression (258x512 64->256, bf8_b weights, no bias, ReLU: +6.1%) came from fills contending once the lock-step stopped pacing them; on the bank-run tree that config is -6.5% (below). |
| weight_stream_decouple: stagger each core's K-block order, poll weight rounds non-blocking from the store loop, stagger the first fill | WIN, not graduated | 514x1024 128->512 7.73 -> 5.00 ms, P7 2.05 -> 1.53, 514x1024 128->256 5.47 -> 4.57, P1 -7.4%, P2 -9.3% | not bit-identical: rotated K order, 0.0028% of outputs differ by 1 bf16 ulp, PCC equal | The cost it removes was not the stream: the lock-step made all 110 cores store and refill at the same M-block boundary, saturating DRAM (P7: 142 us fill + 100 us store per boundary). It was measured before the bank-run fill removed the fill half of that burst, so its value on the graduated tree is unmeasured. Its reader / writer changes conflict with the new fill code. Re-measure and integrate next round, for the cases whose weights stream (the three 128->512 shapes). Helper gap it hit: the multicast pipe's receive / send block; there is no ack-now / complete-later or non-blocking "all acks in" query. |
| store_offload: store the output from the reader, or from inside the writer's weight waits | NULL | writer-serve P1 13.76 -> 13.68 ms, 514x1024 128->512 flat; reader store P1 +7.9%, 514x1024 128->512 +14.7%, P2 +11.9% | bit-identical | The store costs output bytes / ~500 GB/s whoever issues it. DRAM writes on NoC0 cost ~2x NoC1 (P1 +1.88 vs +1.09 ms), so the writer-on-NoC1 store is right. |

### Graduated
1. **Bank-run fill** (`8bc88914e73`). Domain: every DRAM-interleaved ROW_MAJOR input with one pixel per page (C_in a multiple of 32) on the K-blocked tap-direct walk, at any shape or padding. Carve-outs, all feasibility (they keep the per-pixel fill, which is still their path): TILE and sharded inputs (no pixel pages), L1-interleaved input (pages stride by the L1 bank count), C_in not a multiple of 32 (padded lanes), and the other walks (face-direct, row gather, strided window, slab-direct: their cb_im2col_rm is live during the fill, and the staging needs it dead). The staging ring aliases cb_im2col_rm: up to 4 chunks of up to 16 KB, at least 8 KB each (bank reads >= 1 KB) or 2 chunks of what fits. No new L1, no blocking change. DRAM trids 8-11 on this path.
2. **Weights resident** (`cd0c2f659c3`). Domain: every interleaved K-blocked walk, ROW_MAJOR and TILE, with more than one K-block and more than one M-block per core, not face-direct or the split row gather, where the solve finds a block with the whole [K, N] resident. Otherwise the per-K-block stream, unchanged. Its experiment knobs ("leftover" / "trim" modes, the no-ack channel, an im2col cap) were deleted; the only path is the solve-counted one. No measured carve-out.

FLUX.2 bh2x4 before / after (device kernel us, median of 3; `test_flux_conv3x3_compare.py`, ROW_MAJOR input):

| prio | case (H x W, C_in->C_out) | calls | before | + bank-run | + resident | change | util (110-core HiFi4) | target | ttnn.conv2d |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1026x2048 32->128 | 5 | 13758.4 | 3637.8 | 2606.7 | -81.1% | 39% | 1453 | 9227 |
| 2 | 514x1024 64->256 | 6 | 4423.4 | 2282.7 | 1619.8 | -63.4% | 63% | 1453 | 50358 |
| 3 | 1026x2048 128->3 | 1 | 13778.6 | 5696.9 | 5219.8 | -62.1% | 2% | 136 | 11752 |
| 4 | 514x1024 32->128 | 5 | 2794.3 | 921.8 | 725.8 | -74.0% | 35% | 363 | 2189 |
| 5 | 1026x2048 64->128 | 1 | 14800.1 | 5568.1 | 3892.0 | -73.7% | 52% | 2905 | 11395 |
| 6 | 1026x2048 64->256 | 1 | 17483.2 | 8819.6 | 5881.2 | -66.4% | 69% | 5810 | 17224 |
| 7 | 258x512 128->512 | 8 | 2049.8 | 1681.0 | 1682.0 | -17.9% | 60% | 1453 | 14637 |
| 8 | 258x512 64->256 | 5 | 1078.8 | 591.8 | 510.8 | -52.7% | 50% | 363 | 10808 |
| 9 | 514x1024 128->3 | 1 | 3425.9 | 1528.5 | 1409.7 | -58.9% | 2% | 34 | 2249 |
| 10 | 130x256 128->512 | 17 | 544.2 | 445.5 | 444.9 | -18.2% | 57% | 363 | 3226 |
| 11 | 514x1024 64->128 | 1 | 3436.1 | 1359.8 | 1077.1 | -68.7% | 47% | 726 | 2638 |
| 12 | 514x1024 128->256 | 1 | 5469.8 | 3549.3 | 2762.5 | -49.5% | 74% | 2905 | 52918 |
| 13 | 514x1024 128->512 | 1 | 7734.8 | 6392.9 | 6386.9 | -17.4% | 64% | 5810 | 58826 |
| 14 | 258x512 128->256 | 1 | 1440.5 | 945.3 | 832.3 | -42.2% | 61% | 726 | 11439 |
| 15 | 66x128 128->512 | 10 | 144.1 | 139.9 | 139.2 | -3.4% | 46% | 91 | 824 |
| 16 | 130x256 32->128 | 1 | 191.4 | 73.9 | 73.9 | -61.4% | 22% | 23 | 135 |
| 17 | 66x128 32->128 | 1 | 59.3 | 25.7 | 25.6 | -56.8% | 16% | 6 | 453 |

Call-weighted 209.6 -> 95.8 -> 78.9 ms. 2 of 17 reach the 70% target (P6 at 69%, prio 12 at 74%). PCC vs torch 0.9999985 on all 17, unchanged.

Where the time is now:
- **P3 / prio 9 (conv_out, C_out 3 -> one N tile):** the 3x DRAM re-read of a 538 MB input. Its DRAM floor is ~1.7 ms, against 5.22 ms. Lever: the ring + bank-run fusion.
- **P1 / P4 (C_in 32):** fill-stubbed + resident is ~1.8 ms on P1. The rest is the store's 537 MB of DRAM writes overlapping compute, plus the 0.3 ms everything-stubbed floor.
- **The 128->512 shapes (P7, prio 10, 13):** the weight stream, whose weights do not fit. Lever: weight_stream_decouple, re-measured on this tree.

### Guard set (no regression)
- **Panoptic (22 compare cases, LoFi bf8_b sharded):** every new path is gated off on sharded input (the descriptor emits no new define, and the solve returns the same blocking), so the programs are unchanged. Outputs: 45 / 45 LOOSE outputs bit-identical to Refinement 7 (sha256, `panoptic_hashes.py`). Timing, before -> after, one run: 20 / 22 within +-1.5%; stem.conv1 +1.3%; res5.1 +3.1%. Re-measured on both trees (base = the Perf 16 code reverted in place), medians of 3-4 runs:
  - stem.conv1 9711 vs 9719;
  - res5.0 25295 vs 25407 (+0.4%);
  - res5.1 25455 vs 25786 (+1.3%; base runs span 25010-25700, after 25479-25787);
  - res5.2 24607 vs 24683.
  All four are within run-to-run noise.
- **Single-device FLUX.2 (the 17 LOOSE convs of a 1024² decode/encode, TILE input; resident on vs off on the bank-run tree):**
  - 1024² 128->3 1725 -> 1292 (-25%);
  - 1024² 128->128 5117 -> 3708 (-27.5%);
  - 1024² 256->128 9715 -> 7232 (-25.6%);
  - 512² 128->256 1708 -> 1561 (-8.6%);
  - the other 13 within -0.1% / +0.8% (the stride-2 downsamplers and the one-M-block shapes do not take the path).
  The bank-run fill does not touch TILE input; the 1024² 3->128 encoder conv_in (C_in 8, padded lanes) keeps the per-pixel fill: 9320 -> 9302.
- **Off-focus configs, resident on vs off on the bank-run tree (`graduation_r16/test_guard_cfg.py`; sha1 identical on all 8):**
  - ROW_MAJOR 258x512 64->256 bf8_b weights, no bias, ReLU: 490044 -> 458229, and 491008 -> 459897 with symmetric padding;
  - ROW_MAJOR 258x512 64->256 bias: 591894 -> 512341;
  - ROW_MAJOR 514x1024 32->128 no bias, ReLU: 914550 -> 730441;
  - ROW_MAJOR 130x256 64->64: 69076 -> 69110;
  - TILE 1026x2048 32->128: 3723575 -> 3112665;
  - TILE 514x1024 64->256: 2305807 -> 1633334;
  - TILE 258x512 64->256 bf8_b weights, no bias, ReLU: 471574 -> 432087.
- **Golden:** interleaved ROW_MAJOR slice 210 passed (after each graduation); interleaved TILE slice 567 passed (after weights resident). Sharded programs are unchanged (see Panoptic). Unit tests: `test_conv2d_3x3.py`, `test_conv2d_3x3_layouts.py`, `test_conv2d_3x3_asymmetric_padding.py`: 113 passed.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `dataflow_kernel_lib::read_with_state` (`local_copy_helpers_dataflow.hpp`) | capability | The stateful read fixes the source address and varies only the destination (or the reverse). The bank-run de-interleave moves one pixel per read from a staging row to a strided slab row, so both addresses change on every read. The raw `ncrisc_noc_read_set_state` / `ncrisc_noc_read_with_state` pair keeps the size in the command buffer and writes two addresses per read (~24 NCRISC cycles per pixel). The staging also borrows cb_im2col_rm's L1 through `get_local_cb_interface(...).fifo_limit - fifo_size`: there is no API for a CB's base address as scratch while the CB is idle. | n/a (inexpressible); a non-stateful `noc_async_read` per pixel was not measured | whole op P1 3637769 (vs 13758372 with the per-pixel page fill) | `kernels/conv2d_3x3_reader.cpp`, `#ifdef BANK_RUN_FILL` block in the ROW_MAJOR fill |

Weights resident needed no bypass: `ckl::matmul_block` with in1 `NoWaitNoPop` + `In1BaseOffsetFn` expresses it. The first-M-block wait for each round is a PreKBlockFn wrapper (`ResidentWeightsPreKBlock`). Feedback for the helper: there is no in1 policy for "wait once on the first pass, keep fronted across calls". The caller has to own both the cumulative wait and the final pop.
