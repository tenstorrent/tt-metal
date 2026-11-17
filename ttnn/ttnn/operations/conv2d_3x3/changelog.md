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
