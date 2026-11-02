# Changelog: conv2d_nhwc

## Phase 0 — Core Implementation
- **Date**: 2026-11-02
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime R0: an im2col gather (reader) → tilize → K-blocked `matmul_block` with packer-L1 accumulation → in-place bias → untilize → DRAM write.
  - 2-D (output-pixel × output-channel) split over the full grid.
  - One `ttnn.generic_op` dispatch per call.
- **Verifier changes**:
  - **R10 column-clipped input window.** A 2-D clip for single-output-row blocks, and a trimmed contiguous range for multi-row blocks. This fixed 9 SUPPORTED tt-forge cells that failed with "window exceeds L1".
  - **Two-tier L1 budget.** The 768 KiB preference first, then the hard cap `get_max_worker_l1_unreserved_size() − 128 KiB`. This fixed SDXL 32×32×1280 s2 at fp32.
  - **SUPPORTED `stride_mode += strided`.** The kernels were already stride-parametric; probed and golden-verified.
  - **DRY fixes:**
    - compute CB indices are now CT args;
    - `DST_TILES`, `ACC_DTYPE` and the compute config derive from `MATH_FIDELITY` / `FP32_DEST_ACC_EN`;
    - weight tile bytes come from the weight dtype.
  - **Writer:** `writes_flushed` per tile-row and one final write barrier.
  - **`validate()`:** checks the prepared weight's K rows against C_in / kernel_size / groups.
- **SUPPORTED at Phase 0**:
  - dtype = [float32], weight_dtype = [float32], layout = [ROW_MAJOR], memory_layout = [INTERLEAVED];
  - bias_mode = [with_bias, no_bias], activation = [none];
  - channel_alignment = [tile_aligned], stride_mode = [unit, strided], groups_mode = [dense], dilation_mode = [unit].
- **Accuracy achieved**: PCC ≥ 0.999997, max_abs_err ≤ 1.1e-2, rel_rms_err ≈ 1.1–1.2e-3, ULP p50 ≈ 1.3e4 (tf32 operand truncation). Measured on 5 shapes via `test_conv2d_nhwc_precision_baseline.py`.
- **Golden suite at Phase 0**: 172 supported_pass, 0 supported_fail / xpass_drift / xfail_wrong_mode, 2262 xfail_expected, 1104 invalid_skipped (per `verifier_report.json`). As handed over: 133 pass, 9 supported_fail.
- **Issues encountered**:
  - The 9 window-OOM `supported_fail` cells, fixed in place.
  - The planner's claim that no shape reaches R10 did not hold for the translated tt-forge sweep.
- **Tests added**:
  - `test_conv2d_nhwc.py`: strided and wide-row column-clip cases; the stride rejection check moved to dilation.
  - `test_conv2d_nhwc_precision_baseline.py`.

## Refinement 1 — BLOCK_SHARDED + WIDTH_SHARDED placement (regimes R4 / R5)
- **Date**: 2026-11-02
- **What was done**: `SUPPORTED["memory_layout"] += BLOCK_SHARDED, WIDTH_SHARDED`, implemented natively. One `generic_op` dispatch; the output keeps the input's TensorMemoryLayout in L1.
  - **Reused**: the R0 reader gather, the compute tilize → `matmul_block` (packer L1 acc) → bias → untilize chain, the writer's weight/bias K-block fill, and all R0 CBs.
  - **Input**: the shard is consumed in place through `cb_in_shard` (`cb_descriptor_from_sharded_tensor`). The gather takes a source policy (`WindowSource` for R0, `ShardSource` for R4/R5). Halo sticks outside the resident shard come from the same-column shards, one NoC read per owner run, into the reused `cb_act_window`. WIDTH has no halo.
  - **K exchange**: K order is `[slice][kh][kw × slice_width]` (technique 6, in `prepare_conv2d_weights` via `channel_slices(input_memory_config)`). Each core gathers and tilizes only its own slice into `cb_act_send`. The writer multicasts each K-block into `cb_act_tiles` of every core in the group with `ttnn.Mcast` + `McastArgs`: a rotating sender, phase = slice, double-buffered landing, one handshake per K-block. Group = one grid row (BLOCK) or all cores (WIDTH).
    - A WIDTH core set spanning grid rows multicasts over its bounding rectangle, with the extra cores as receive-only sinks: the helper's rotating mode needs rectangular groups.
    - With S = 1 there is no exchange.
  - **Technique 5**: column tiles at or beyond N_t (`S·ceil(N_t/S)` padding) are zero-filled by the writer for both weights and bias. Prepared tensors stay C_out wide, so `conv2d_nhwc` can still read C_out from the weight.
  - **Output**: untilized straight into the output shard (`cb_out_shard`, zero-copy, tile-sized pages).
    - BLOCK: `core_m_tiles·32` rows × `ceil(N_t/S)·32` columns per core.
    - WIDTH: ttnn requires the shard height to be exactly M, so a partial last tile-row is staged in `cb_out_rm` and the writer copies only its valid rows.
  - **Perf structure**:
    - BLOCK weights are read once per grid column and multicast down it (second channel, fixed sender at row 0).
    - Barriers are batched per K-block (weights) and per tile-row (gather).
    - `select_blocking_sharded` pins bn to the slice and walks bm (whole shard first) and bk (the largest divisor of the slice's K that fits).
    - Knob `OWN_PREFETCH`: tilize the whole own slice at m-block start. It is taken only when it fits at the whole-shard bm.
  - **Measured** (fp32, device per-core kernel max, AICLK 1350):
    - SDXL 64×64×640 3×3, pinned 10×10 BLOCK: 2,531 µs, against 8,810 µs for the same conv interleaved (R0).
    - Weight column multicast on vs off: 2,531 vs 2,544 µs (neutral).
    - 28×28×256 BLOCK with own prefetch: 80.1 → 77.1 µs.
    - Forcing prefetch on SDXL (bm → 1): 9.3 ms. That is why it is gated.
- **Accuracy achieved**: PCC ≥ 0.999 and rel-RMS ≤ 0.02 on every BLOCK/WIDTH case (fp32 tolerance), same tf32 behaviour as R0. Shapes: 7×7×512, 14×14×256 s2, 14×14×1024→2048 1×1 s2, 32×32×1280 s2 (WIDTH); 8×8×64, 2×14×14×128, 28×28×256 (s1, s2), 56×56×128, 64×64×640 s2, 32×32×256→320, and the pinned 10×10 64×64×640 (BLOCK).
- **Golden test progress**: all 18 BLOCK/WIDTH cells inside SUPPORTED pass (9 shapes × bias on/off). The slice also included 2 interleaved representatives (pass), 0 failures and 0 XPASS. The other BLOCK/WIDTH cells are xfail (dtype, dilation, groups, TILE, relu) or INVALID skips.
- **Issues encountered**:
  - The multicast channel must be configured on NOC_1, because `WriterConfigDescriptor` uses the preferred DRAM-write NoC.
  - Rotating multicast needs rectangular groups and S > 1, hence the WIDTH sinks and the S = 1 bypass.
  - WIDTH output shards cannot be taller than M, hence the tail staging.
  - The device profiler's aggregate `DEVICE KERNEL DURATION` was corrupted by start-time skew (~336 s). Per-core durations from `profile_log_device.csv` were used instead.
  - Remaining headroom for Refinement 3 on the SDXL shape (fp32): bk = 3 under the 768 KiB budget gives 60 K-blocks with subblock 1×2 (bm = 13 is prime). The just-in-time owner gather sits on the critical path when prefetch does not fit.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_sharded.py`: 15 cases (BLOCK and WIDTH, ragged rows, uneven C_out over slices, a WIDTH group spanning two grid rows with a ragged tail, strided, 1×1, pinned 10×10).
  - `test_conv2d_nhwc_perf_probe.py`: profiling probes.

## Refinement 2 — Numerical configurability (bf16 activations, bf16 / bf8b weights, mixed precision) + fused ReLU
- Date: 2026-11-02
- What was done:
  - `SUPPORTED["dtype"] = [fp32, bf16]`, `SUPPORTED["weight_dtype"] = [fp32, bf16, bf8b]` (every pair, mixed precision included), `SUPPORTED["activation"] = ["none", "relu"]`. No EXCLUSIONS were needed. bf8b activations stay with Refinement 7 (TILE).
  - Compute policy: `COMPUTE_POLICY` / `compute_policy(dtype, weight_dtype)` in the program descriptor is the single source for `math_fidelity`, `fp32_dest_acc_en`, the DEST tile count (subblock rule) and the partials page format (`cb_partials` / `cb_biased`). This replaces the module constants. Policy: fp32 → HiFi4 + fp32 DEST (unchanged); bf16 → HiFi2 + fp32 DEST.
  - Formats and geometry: the activation / output CBs already followed the tensor dtype, and the kernels were already parametric in `elem_bytes`, so no kernel change was needed. `prepare_*` and `validate` share `conv_geometry` with `elem_bytes` (bf16: granule 8, so `tap_width` / `band_tiles` change).
  - Fused ReLU (technique 4): `pack_relu_config(ReluConfig::zero())` wraps only the final output pack (the bias pack, or the untilize pack when there is no bias). It is reset to `none()` right after, so the next block's tilize and L1-accumulated partial packs never see it. It is a compile-time arg, so there is no cost when off. This covers the interleaved and BLOCK/WIDTH paths.
  - BLOCK_SHARDED with COL_MAJOR shard orientation (transposed grid) is now built. Bringing bf16 into SUPPORTED exposed 3 translated `test_conv_bs_grid_pre_sharded` cells that used it. The fix: `_block_core` maps (slice, row) to a core; K-exchange groups use `McastCoreOrder.ColumnMajor` and weight multicast uses `RowMajor`; halo sources move along x; the output keeps the input's orientation. WIDTH + COL_MAJOR is a typed refusal.
  - The descriptor's L1-fit and placement refusals (window exceeds L1 / R10 not built, sharded blocks do not fit, non-rectangular grids, shard width granule) now raise the typed `UnsupportedAxisValue` instead of a bare `NotImplementedError`.
  - Reused: every CB, helper call and descriptor branch. Added: the policy function, one compute CT arg (`fuse_relu`) with 4 `pack_relu_config` calls, and the COL_MAJOR coordinate mapping.
- Fidelity / accumulation policy (measured, bf16 activations; the gate is PCC ≥ 0.995 and rel-RMS ≤ 0.05):
  - SDXL level-1 3×3 BLOCK 10×10, bf16 × bf8b, rel-RMS: HiFi4/fp32acc 0.0080, HiFi2/fp32acc 0.0089, LoFi/fp32acc 0.0202, HiFi2/bf16acc 0.0139, LoFi/bf16acc 0.0220. PCC ≥ 0.9997 in every config.
  - Same shape, max per-core kernel time: HiFi2/fp32acc 884k cycles (~655 µs @1.35 GHz), LoFi/fp32acc 879k, HiFi2/bf16acc 630k (~467 µs), LoFi/bf16acc 624k. Fidelity is not on the critical path (the shape is dataflow-bound). fp32 DEST costs about 1.4× through half the DEST tiles (smaller subblocks) and double the partials bytes.
  - Full golden INPUTS sweep at HiFi2 with bf16 DEST (bf16 and bf8b weights, 174 cells): worst rel-RMS 0.044 (14×14×256→512 k3 p0, bf16 weights), 0.039 (bf8b). That is too close to the 0.05 gate, because bf16 packer-L1 accumulation rounds the partials every K-block. With fp32 DEST the same worst cells measure 0.0048 (bf16 weights) and 0.0091 (bf8b).
  - Chosen: bf16 → HiFi2 + fp32 DEST (precision first). Finding for Refinement 3: a bf16-DEST variant is ~29% faster on the perf shape and passes there (rel-RMS 0.014). It would need a gate on accumulation depth / shape class, not a blanket switch.
- Accuracy achieved:
  - bf16 activations: PCC ≥ 0.9995, rel-RMS ≤ 0.0091 on every probed cell (golden INPUTS worst cells, SDXL L1/L2 BLOCK, ResNet layer4 BLOCK, 32×32×1920→1280 BLOCK).
  - fp32 activations: unchanged (PCC ≥ 0.999, rel-RMS ≤ 0.02, including fp32 × bf8b).
- Golden test progress (slices run):
  - Interleaved + BLOCK + WIDTH ROW_MAJOR cells for every dtype pair: 156 passed. Relu cells (interleaved + BLOCK): 24 passed.
  - Loose BLOCK cases: 10 passed, including the SDXL level-1 3×3 perf case and the bf16 `_MODEL_PLACEMENT_CASES` with 40/240-channel slices and 10/20 out tiles. conv_out C_out = 4 stays xfail (`c_out_unaligned`).
  - Translated + regression suites: every group run, 0 failures after the fixes above. The bf16 `ttnn_pytorch_sweep` alone: 614 passed.
- Issues encountered:
  - Newly reachable bf16 translated cells exposed two pre-existing gaps.
    - COL_MAJOR BLOCK orientation: now built (see above).
    - `test_conv2d_with_fold` (C_in = 320, k = s = 16/32, 224–672 wide): even a single output row's input window is several MB. These need R10 DRAM slicing (a direct DRAM im2col gather), which is not built; they are now typed refusals (lenient xfail). Finding: R10 has no refinement in the queue yet.
- Tests added:
  - `test_conv2d_nhwc_sharded.py`: `run_sharded` takes dtype / weight_dtype / activation / orientation. Added `test_conv2d_nhwc_mixed_precision` (interleaved, BLOCK, WIDTH-tail × {bf16×bf16, bf16×bf8b, bf16×fp32, fp32×bf8b} × {none, relu}), `test_conv2d_nhwc_relu_no_bias_block`, and `test_conv2d_nhwc_block_col_major` (s1, s2 ragged).
  - `test_conv2d_nhwc_perf_probe.py`: `test_perf_probe_block_sdxl_l1_bf16_bf8b`, the perf-1 contract config.

## Refinement 3 — Speed up the perf-flagged SDXL level-1 3×3 (BLOCK)
- Date: 2026-11-02
- What was done (perf; no SUPPORTED change). Target config: SDXL level-1 3×3, (1, 64, 64, 640) → 640, bf16 ROW_MAJOR pre-sharded (410, 64) on 10×10 BLOCK, bf8b weights, bias.
  - **Baseline breakdown** (883k cycles/core): the whole-slice own prefetch (468 KB, over the L1 target by itself) tilized all 9 own K-blocks before the first matmul, so the reader's gather (~600k cycles) sat entirely on the critical path.
  - **Interleaved slice K order** (writer + compute only; the prepared weight layout is unchanged): K-block j is own K-block j div S of slice j mod S. The writer maps weight rows `slice·slice_k_tiles + (j div S)·bk`; the rotating sender order becomes 0, 1, …, S−1, 0, …
  - **Send-depth tilize schedule**: the owner tilizes own block kb + `ACT_SEND_DEPTH` right after own block kb was multicast. The old `OWN_PREFETCH` knob is subsumed (`ACT_SEND_DEPTH ≥ kblocks_per_slice` is the old prefetch). `cb_act_rm` is one own K-block deep (`ACT_RM_KBLOCKS`), so the reader gathers the next own block during the other S−1 slices' turns.
  - **Gather rewrite** (reader, shared by R0 and R4/R5):
    - Stateful local NoC reads: every gather source is core-local, so the read state is set once per m-block.
    - K-block → band segments are built once per K-block.
    - Interior pixels of one output row form a run with an affine source address: a tight `read; addr += stride·tap; dst += row` loop.
    - Edge pixels and K-padded blocks keep the general `gather_row`.
    - Result: interior tile-row gather 5,000 → 1,000 cycles. Before the rewrite, the per-row address arithmetic alone was ~107 cycles per row.
  - **Weight stream lead** (writer): `SHARDED_WEIGHT_DEPTH = 6`. Weight block j + 4 is issued before act block j and landed (barrier + column multicast) after it, so the weight DRAM latency and column handshake leave the row chain. Depth 2 reproduces the old one-ahead behaviour. R0 keeps `WEIGHT_DEPTH = 2`.
  - **L1 footprint target**: `SHARDED_CB_BUDGET_BYTES = 480 KiB` for R4/R5 CBs (the shards are extra), then the old budgets as fallbacks. The plan is bm = 13, bk = 3, 60 K-blocks, F = 484,608 B.
- Measured (device per-core kernel max from `profile_log_device.csv`, AICLK 1350 MHz; the aggregate CSV duration is corrupted, as in Refinement 1):
  - **SDXL L1 3×3 BLOCK bf16 × bf8b: 883,338 → 258,975 cycles (654 → 192 µs, 3.41×).** `achievable_ns` = 149,049 (201k cycles), so we are 1.29× off.
  - `device_l1_peak_bytes` (graph capture, as the harness): **590,336 B ≤ 618,992 B** (before: the prefetch footprint alone exceeded it).
  - Step by step:
    - interleaved order + send-depth schedule: 883k → 480k
    - stateful reads: → 464k
    - interior fast path: → 377k (still per pixel)
    - run-based gather: → 284k
    - writer weight/act overlap: → 266k
    - weight lead 4: → 258k
  - Guard set, before → after (cycles):
    - BLOCK 28×28×256 fp32 bias: 103.3k → 81.0k
    - BLOCK 28×28×256 bf16 × bf8b no bias: 66.9k → 40.2k
    - WIDTH 14×14×256→512 s2 fp32: 109.3k → 97.4k
    - interleaved 28×28×256 bf16 × bf8b relu: 101.9k → 101.1k
    - interleaved SDXL 64×64×640 fp32: 11.869M → 11.868M
  - No shape got slower.
- Bottleneck now (ablations on the target config, all at bk = 3):
  - Compute alone (all NoC stubbed): 194k.
    - LoFi would give 171k; bf16 DEST gives no change; tilize is ~10k.
    - The rest is unpack-bound matmul at the forced 1×2 subblock: bm = 13 is prime, so in1 is re-unpacked for every in0 row.
  - Data movement alone (matmul stubbed):
    - weights stubbed: 174k (145k with a 32 B act payload)
    - with weights: 238k before the lead
  - The row act exchange is transfer-bound at ~2.9k cycles per 78 KB block.
  - Combined: 258k. Compute and exchange overlap imperfectly. A deeper landing ring (depth 3) gave only −6k, and a stubbed tilize −2k. The likely cause is L1 contention between the 78 KB/block multicast landing, the unpacker and the fp32 packer-L1 accumulation.
- Next levers (not done):
  - bk = 6 (half the K-block passes and handshakes) needs ≈ 777 KB of CBs at bm = 13 — over the L1 target.
  - Padding bm to 14 for 2×2 subblocks costs +7.7% exchange bytes and also exceeds the budget.
  - LoFi / bf16 DEST for bf16 × bf8b would need a shape-class precision gate (Refinement 2 data).
  - Splitting the act exchange across both data-movement RISCs (two landing CBs by owner parity, matmul `In0SourceFn`) would keep two blocks in flight.
- Accuracy achieved: the SDXL loose case passes its precision gate (PCC ≥ 0.995, rel-RMS ≤ 0.05; bf16). Every unit test passes at its prior tolerance (fp32 PCC ≥ 0.999 / rel-RMS ≤ 0.02).
- Golden test progress: `test_golden.py -k "BLOCK_SHARDED or WIDTH_SHARDED"`: 130 passed, 99 xfailed, 72 skipped, 0 failed, 0 XPASS. This includes the SDXL level-1 3×3 BLOCK loose case and the other BLOCK loose cases.
- Issues encountered:
  - The profiler marker budget truncates per-K-block zones at ~30 per type, so the zone breakdowns cover the first half of the run.
  - The JIT farm's up-front precompile fails (an unrelated sfpi header error) and is skipped (best-effort).
- Tests added: `test_conv2d_nhwc_perf_probe.py`:
  - `test_l1_peak_block_sdxl_l1_bf16_bf8b` — the harness's graph-capture L1 peak
  - guard probes `test_guard_interleaved_bf16_relu`, `test_guard_block_bf16_nobias`, `test_guard_width_strided`

## Refinement 4 — Depthwise convolution (regime R11)
- Date: 2026-11-02
- What was done: `SUPPORTED["groups_mode"] += "depthwise"` (groups == C_in == C_out), on interleaved and BLOCK / WIDTH. It is a second compute structure, not a block-diagonal matmul.
  - **Gather** (reader, shared R0 / R4/R5 code): per n-block one K-block of `taps × bn` tiles. The im2col row is `[tap][bn·32 channels]`. New segment builder `build_segments_dw` and edge path `gather_row_dw`. The existing interior run loop (affine stateful reads) is reused unchanged. When bn covers the whole window stick, the kW taps of a kernel row are one read.
  - **R0 window**: holds only the core's own channel tiles (a DRAM read at channel offset `n_tile_start·32`, new reader RT arg). Each input byte is read by one n-core only.
  - **Compute**: per tile-row, tilize `taps × bn` per-tap tiles, then per output tile one `eltwise_chain`: `BinaryFpu<Mul>` with the tap weights row-broadcast, both operands on `Col` mapping + `Strided {ct, bn}` (tile `tap·bn + ct`), `DestAccumulation::WholeShape` over the taps, one pack into `cb_partials`. Then the same `bias_and_untilize` (bias, ReLU, untilize into DRAM rows or the output shard).
  - **Weights**: `prepare_conv2d_weights` lays out `[kH·kW·32, C_out]`, with row 0 of tap t's tile-row = `w[:, 0, kh, kw]`. This is placement-independent. The writer reads `weight_k_rows = taps` rows per block (new CT arg; dense passes bk).
  - **Sharded**: no K exchange (each channel slice is self-contained), no weight column multicast. `nt_s = shard width / 32`, and the output shard is the core's own slice. A shard width that is not a tile multiple is a typed refusal.
  - **Refusals**: a channel multiplier (C_out ≠ C_in) is a typed `UnsupportedAxisValue`.
  - `prepare_conv2d_bias`'s groups placeholder now reads groups > 1 as depthwise.
  - Reused: every CB, the gather run loop, tilize, bias / untilize / ReLU, the writer weight stream and the descriptor branches. Added: `_dw_plan_keys` (one source for the dense-vs-depthwise plan differences), knobs `DW_ACT_TILE_ROWS = 1`, `DW_WEIGHT_DEPTH = 2` and `DW_CORE_M_TILES_MIN = 1`, the depthwise footprint / walk in both selectors, and 3 CT args + 1 RT arg.
- Accuracy achieved:
  - fp32: PCC ≥ 0.999, rel-RMS ≤ 0.02.
  - bf16 (incl. × bf8b): PCC ≥ 0.995, rel-RMS ≤ 0.05.
  - Shapes: 14×14×32, 2×9×11×64, 28×28×96 k5, 56×56×128 s2, 7×7×1024, 112×112×32, 16×16×64 p0; WIDTH 7×7×1024 and 14×14×256 s2; BLOCK 28×28×256 and 2×14×14×128 k5; bias / no bias; relu.
- Golden test progress:
  - `test_golden.py` depthwise slice (`-k g16_ or g32_ …`): 157 passed, 0 failed, 0 XPASS. That includes every depthwise cell inside SUPPORTED: 103 interleaved / strided / 5×5 cells, 5 WIDTH 1024 cells and the loose bf16 × bf8b case.
  - Remaining depthwise xfails: unaligned channels (16, 48, 240), dilation and HEIGHT.
  - `test_translated.py`: all 164 depthwise sweep ids. 99 passed (newly reachable, incl. 2048-channel and 150×150 cells); 65 xfail are typed refusals.
- Perf (BH, AICLK 1350, bf16 × bf8b, per-core kernel max):
  - interleaved: 112×112×32 76.9 µs, 56×56×128 34.6 µs, 112×112×64 s2 55.6 µs
  - WIDTH 7×7×1024: 10.3 µs; BLOCK 28×28×256: 14.6 µs
  - Zone breakdown (112×112×32): the DRAM window fetch is 50k cycles mean (83k max) of 103k, for ~450 64 B sticks per core. That is DRAM-request / issue bound, made worse by the ~3× halo re-read at 1.1 output rows per core. Output write is 6k.
  - `DW_CORE_M_TILES_MIN` (taller per-core blocks) is kept at its no-op default. At 7 it gives 112×112×32 77 → 64 µs, but at 4 it gives 56×56×128 35 → 59 µs.
  - Next levers (not done): split the window fetch across both data-movement RISCs (the writer idles during it), or native HEIGHT-sharded input (Refinement 5), which removes the DRAM stick reads.
- Issues encountered:
  - A `static_assert` inside `if constexpr` in a non-template function fired on the dense path; it is now made conditional.
  - The aggregate profiler duration is still corrupted; per-core spans from `profile_log_device.csv` were used.
- Tests added:
  - `tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_depthwise.py`: 26 cases, including the multiplier refusal.
  - Depthwise perf probes in `test_conv2d_nhwc_perf_probe.py`.

## Refinement 5 — HEIGHT_SHARDED placement (regime R3) + weight multicast (R1)
- Date: 2026-11-02
- What was done: `SUPPORTED["memory_layout"] += HEIGHT_SHARDED`, natively, for the dense and depthwise paths. One `generic_op` dispatch; the output is HEIGHT-sharded in L1.
  - **Reused**: the whole R4 sharded path with one channel slice (S = 1: no K exchange, compute tilizes straight into in0). That covers the shard-backed input CB (`cb_in_shard`, zero-copy), the `ShardSource` gather with halo fetch (`fetch_sticks`, one NoC read per owner run), the whole-shard bm walk, the writer weight stream, and untilize into the shard-backed output CB. No kernel changed. The prepared K order is the interleaved one (= the S = 1 slice order), so `prepare_*` is unchanged.
  - **Added** (host only):
    - `sharded_slices`: HEIGHT gives one whole-C_in slice. COL_MAJOR orientation and a shard width ≠ C_in are typed refusals.
    - `height_shard_cores`: input shard cores in shard order. They are the halo sources, and the first `g_m` are the compute cores.
    - HEIGHT branches in `_sharded_core_map`, `output_memory_config` (first `g_m` input cores, `core_m_tiles·32` rows × C_out each) and the reader halo RT args.
    - Weight multicast (R1) on HEIGHT: one group of all compute cores, row-major, fixed sender = core 0, dense and depthwise. It reuses the BLOCK column-multicast channel (`w_mcast`) and the writer's `reads_weights` path. Guarded by `MCAST_MAX_RECTANGLES = 3` (the helper's limit) and uniform m-block counts. Otherwise each core reads its own weights.
  - Output split: `core_m_tiles = ceil(M_t / num_input_shards)`, whole tiles. This aligns with the input shards for stride-1 convs. For strided convs most of each core's window is halo (correct, but traffic left for Refinement 6).
- Accuracy achieved: fp32 PCC ≥ 0.999 / rel-RMS ≤ 0.02; bf16 (× bf8b, relu) PCC ≥ 0.995 / rel-RMS ≤ 0.05. Shapes: 56×56×64 (s1, s2), 2×28×28×64, 17×17×32, 8×8×32 (1-stick shards), 56×56×256 1×1, ResNet50 layer2 pinned (128, 98 cores); depthwise 112×112×32, 56×56×128 s2, 2×9×11×64, 28×28×64 bf16×bf8b relu.
- Golden test progress:
  - `test_golden.py -k HEIGHT`: 100 passed, 124 xfailed, 72 skipped, 0 failed, 0 XPASS. This includes all 4 ResNet50 HEIGHT perf cases (bf16 × bf8b).
  - `test_translated.py -k HEIGHT`: 137 passed, 131 xfailed, 0 failed.
  - Remaining HEIGHT xfails: unaligned channels, dilation, grouped, TILE.
- Issues encountered: None. Not measured here: device time for the ResNet HEIGHT perf cases. That is Refinement 6's baseline.
- Tests added: `test_conv2d_nhwc_sharded.py`:
  - `test_conv2d_nhwc_height` (6 shapes × bias / no bias)
  - `test_conv2d_nhwc_height_bf16_bf8b` (none / relu)
  - `test_conv2d_nhwc_height_resnet_layer2_pinned`
  - `_sharded_config` gained a HEIGHT branch

  `test_conv2d_nhwc_depthwise.py`: 3 HEIGHT cases plus HEIGHT bf16 × bf8b relu.

## Refinement 6 — Speed up the perf-flagged ResNet50 HEIGHT convs
- Date: 2026-11-02
- What was done (HEIGHT / R3 path; no SUPPORTED change):
  - **Tap-major gather** (`TAP_MAJOR_GATHER`; HEIGHT, stride 1, dilation 1, whole-tile taps). A K-block's RM data is laid out per tap as one contiguous 32-pixel × C_in block, and compute tilizes one tap wide (`tilize_w`). A tap's pixels along one output row then come from one NoC read, not one read per pixel per kernel-row band. bk is a whole number of taps. The prepared K order is unchanged.
  - **Resident weights** (`WEIGHTS_RESIDENT`). The whole weight slice stays in `cb_weight_tiles` for the kernel's life and is read and multicast once, before the first m-block. Compute reads it in place (matmul in1 `NoWaitNoPop` + per-K-block base offset, cumulative waits). The sender reads per DRAM bank with stateful reads, one address computation per bank (weight issue 7.4k → 3.2k cycles). The bias is loaded after the weights. If no resident blocking fits L1, the op falls back to the streamed path with the old placement.
  - **Single-K m-block pipeline** (`HEIGHT_SINGLE_K_PIPELINE`, `HEIGHT_MIN_M_BLOCKS = 2`). The whole K is one K-block (no packer-L1 partials passes), with the largest bm that fits. `cb_act_rm` is one whole m-block deep, so the gather of m-block i+1 overlaps compute of m-block i.
  - **Output placement** (`HEIGHT_ALIGN_OUTPUT`). Output shard r is computed on, and placed on, the input core whose shard best covers its window (monotone in r). For stride-1 this is identical to before. For strided convs most of the window becomes resident instead of far-away halo (layer2 downsample fetch was ~100k cycles of NoC0 wrap-around traffic). The weight multicast spans the bounding rectangle of the input grid (one rectangle); input cores holding no output shard are receive-only sinks.
  - **Fidelity**: `COMPUTE_POLICY_BY_WEIGHT` makes bf16 × bf8b run at LoFi with fp32 DEST (the reference fidelity). ResNet cases rel-RMS ≈ 0.020–0.025, PCC ≥ 0.9997.
  - Parked knobs, live at trivial defaults:
    - `WEIGHT_MCAST_CHUNKS = 1`: per-chunk multicast rounds pay a ~1.5k-cycle handshake each; 4 and 8 chunks measured slower.
    - `HEIGHT_RESIDENT_KBLOCKS = 1`: progressive K-blocked delivery measured flat on layer2 3×3 and slower elsewhere.
    - `SPLIT_GATHER = False`: the writer RISC gathers odd m-blocks into its own CBs (`CB_ACT_WINDOW2` / `CB_ACT_RM2`). It helped the strided cases (1×1 s2 23.4k → 20.3k) before output alignment; after alignment the reader is no longer the bound.
  - Reused: the HEIGHT sharded path, every CB, the halo fetch, the matmul / bias / untilize helpers and the Mcast channel. Added: `kernels/conv2d_nhwc_gather.hpp` (the reader's gather, shared with the writer for the split; the reader is now a thin wrapper), `gather_tile_row_taps`, the resident-weight writer block (with trid chunks), compute `ResidentIn1Offset`, and host `out_shards` / weight group / sinks.
- Perf (BH p150b, AICLK 1350 MHz, per-core max kernel cycles, bf16 × bf8b, bias). Before → after (achievable):
  - layer1 3×3 56×56×64: 116.0k → 38.8k (38.0k)
  - layer2 downsample s2: 165.3k → 41.0k (41.0k)
  - layer2 3×3 28×28×128: 81.3k → 35.7k (27.0k)
  - 1×1 s2 shortcut: 25.3k → 17.9k (20.3k)
  - `device_l1_peak_bytes` (graph capture) 581,120 / 534,272 / 397,824 / 264,320 B, vs achievable 723,456 / 670,576 / 450,928 / 317,040 B.
- Bottleneck now:
  - layer2 3×3 is compute-bound (matmul ≈ 10.6k, tilize ≈ 6.2k, bias / untilize over 4 m-blocks), queued behind the weight landing. Reading and multicasting 157 KB of bf8b weights takes ≈ 13k cycles (~18 B/cycle into each core, even with the chip idle).
  - layer1 is compute-bound (~38k).
  - Next lever: a dual-NoC weight multicast (half the slice from each data-movement RISC of the sender core). Not done: it needs a second multicast channel attached to the reader kernel.
- No-regression guard set (before → after, per-core max cycles): all within ±1–3% noise.
  - interleaved bf16 relu: 101.8k → 101.6k
  - BLOCK no-bias: 40.3k → 40.1k
  - WIDTH strided: 97.2k → 97.8k
  - SDXL L1 BLOCK: 258–260k → 257k
  - depthwise interleaved / WIDTH / BLOCK: 102.5k / 46.6k / 75–78k / 13.9k / 19.7k → 102.9k / 46.4k / 76–78k / 14.1k / 18.9k
  - BLOCK 28×28×256: 81.0k / 19.5k → 81.1k / 18.7k
- Accuracy achieved:
  - ResNet HEIGHT cases: PCC ≥ 0.9997, rel-RMS ≤ 0.025 (bf16 × bf8b, gate 0.995 / 0.05).
  - Every unit test at its prior tolerance.
- Golden test progress:
  - `test_golden.py -k "(BFLOAT16 and BFLOAT8_B) or HEIGHT"`: 308 passed, 292 xfailed, 416 skipped, 0 failed, 0 XPASS.
  - After the final `HEIGHT_MIN_M_BLOCKS` change, `-k HEIGHT`: 100 passed, 124 xfailed, 72 skipped, 0 failed.
- Issues encountered:
  - A non-uniform m-block count silently disabled the weight multicast (every core read the weights from DRAM). Resident weights are multicast once, so that gate no longer applies to them.
  - A NoC0 halo fetch from a "later" shard wraps the torus. Fixed by aligning the output placement, not by changing NoCs.
- Tests added:
  - `test_conv2d_nhwc_sharded.py::test_conv2d_nhwc_height_resnet_pinned`: layer2 downsample, 1×1 s2, layer1 at pinned geometry.
  - `test_conv2d_nhwc_sharded.py::test_conv2d_nhwc_height_weights_streamed`: resident fallback.
  - `test_conv2d_nhwc_perf_probe.py`: `test_perf_probe_resnet_*` and `test_l1_peak_resnet_height`.

## Refinement 7 — TILE activations (+ bf8b activations): techniques 1, 2, 3 / regime R9
- Date: 2026-11-02
- What was done:
  - SUPPORTED: `layout += TILE`, `dtype += bfloat8_b` (TILE only; bf8b × ROW_MAJOR is INVALID in feature_spec). EXCLUSIONS: TILE × depthwise (no golden cell; R11's per-channel-tile window is not wired to the TILE untilize).
  - Technique 1 (sharded TILE, HEIGHT / BLOCK / WIDTH): each core untilizes one contiguous range of input tile-rows once per kernel. The range is its own shard ∪ the window of its whole m range. Own tile-rows come in place from the zero-copy `cb_in_shard`; halo tile-rows are fetched from the owning shards (one NoC read per owner run) into the new `cb_win_tiles`. The untilized window lands in `cb_act_window` (now compute-produced; bf8b → bf16). The existing gather reads it as one contiguous stick source (`ShardSource` with no halo split), and the tilize / matmul / multicast run unchanged.
  - Technique 2 (DRAM-interleaved TILE): built on R0 (prepare's interleaved K order) rather than an in-op virtual HEIGHT placement. Per m-block, the reader reads the window's whole input tile-rows from DRAM, compute untilizes them, and the gather runs as above. One dispatch, no `to_layout`.
  - Output: TILE in → TILE out. The final pack (`add_bias_bcast_rows`, or a `copy` with no bias; packer ReLU around it) writes tiles straight into the output shard (TileRowMajor = the shard's tile order) or, on R0, into `cb_out_rm`, which the writer drains as tile pages.
  - Technique 3 / R9 (`TILE_DIRECT_IN0`): for 1×1, stride 1, p 0, one K-block per slice, and output tile-row r on the core holding input tile-row r, the shard's tiles are in0. HEIGHT matmuls `cb_in_shard` in place; BLOCK/WIDTH multicast shard K-blocks from `cb_in_shard`. No untilize, gather or tilize; the RM CBs shrink to one page.
  - A 4-D TILE input is accepted when W and W_out are tile multiples. Otherwise `ValueError` (pass it flattened with `input_spatial`).
  - Compute policy: bf8b activations use the bf16 policy (HiFi2, fp32 DEST); bf8b × bf8b runs at LoFi (the ResNet50 reference fidelity).
- Perf: R9 vs the general TILE path (BH, AICLK 1350, per-core max kernel cycles, bf8b TILE, bf8b weights, bias, relu): ResNet layer1 1×1 HEIGHT 28,187 → 21,975 (−22%); layer3 1×1 BLOCK 35,632 → 23,223 (−35%). One run each; the margins are far outside noise.
- Accuracy achieved: every TILE golden cell passes its dtype gate (fp32 PCC 0.999 / rel-RMS 0.02; bf16 0.995 / 0.05; bf8b 0.99 / 0.10), relu included.
- Golden test progress:
  - `test_golden.py -k _TILE`: 167 passed, 19 xfailed (`C_in = 4` DRAM TILE: `c_in_unaligned`, Refinement 8), 0 failed.
  - TILE model-placement loose cases: 5/5 in SUPPORTED pass (4 ResNet50 bf8b TILE + relu, SDXL DRAM-TILE downsample). SDXL conv_in waits on Refinement 8.
  - ROW_MAJOR regression slice (HEIGHT + BLOCK / WIDTH / interleaved representatives + ResNet / SDXL perf loose cases): 160 + 3 passed, 0 failed. Unit directory: 147 passed.
- Issues encountered: a `shard_h / 32` division by zero on ROW_MAJOR HEIGHT (shard_h < 32) broke the reader build; guarded.
- Tests added: `test_conv2d_nhwc_tile.py` (8 placements × {bf8b, bf16} with relu, plus the 4-D TILE accept / refuse case); `test_conv2d_nhwc_perf_probe.py::test_perf_probe_tile_1x1` (R9 A/B).
- Not done / findings: the DRAM TILE path does not overlap the next m-block's untilize with the current matmul (compute untilizes at m-block start). The R0 window is whole tile-rows, so ρ is a little above the RM column-clipped window. Neither was measured or optimized here.

## Refinement 8 — Channel alignment, dilation, grouped convs
- Date: 2026-11-02
- What was done:
  - SUPPORTED: `channel_alignment` += `c_in_unaligned`, `c_out_unaligned`, `both_unaligned`; `dilation_mode` += `dilated`; `groups_mode` += `grouped`. EXCLUSIONS unchanged (TILE × depthwise).
  - **C_in unaligned**: the K tap stays C_in rounded up to the L1 granule (prepare already zero-padded those weight rows). Pad lanes are made zero so a stale NaN never meets a zero weight:
    - R0 RM, tap not a DRAM-alignment multiple (e.g. C_in = 3 bf16, 16 B tap vs a 64 B DRAM read alignment on BH): each stick is read into a pre-zeroed, DRAM-aligned staging half (`cb_stick_stage`) and copied into the packed window. The two halves overlap the next chunk's DRAM reads with the copies.
    - R0 direct reads with a short stick: the window is zeroed once.
    - Sharded / TILE: the gather now has a source stick stride (`src_bytes`) separate from the K tap. A HEIGHT shard may be wider than C_in (SDXL conv_in shard width 8 for C_in = 4); a TILE window's sticks are whole tile columns (C_in = 4 → 32). Taps are then read one at a time.
  - **C_out unaligned**: the R0 RM writer writes only the valid columns of the last n-slice (output page = C_out·elem_bytes). Weight / bias column tile padding is zero in prepare. Sharded outputs untilize into a shard whose width is whole tiles (technique 5, as for the over-wide BLOCK slices).
  - **Dilation**: compile-time `DIL` in the gather. Kernel-row h offsets and tap stick offsets scale by DIL, the band splits into per-tap reads, and the interior fast path uses the effective kernel extent. Depthwise reads per tap. Window bounds already took d. New: an R0 block inside one output row fetches only its kH kernel rows (the d = 18 fp32 256-channel cell did not fit L1 with all d(kH−1)+1 rows).
  - **Grouped**:
    - Native per-group on R0 (`native_groups`: C_out/g and C_in/g whole tiles, group channel offset DRAM-aligned). The core split keeps each core's n range inside one group. The core reads only its group's C_in/g channels of each stick (`win_chan_offset`, as depthwise), and K is one group's. The prepared weight is `[K_g, C_out]` (column c's rows are its own group's), the same matrix the per-group conv needs: no extra weight bytes. This covers AlexNet conv4/5, g = 3 with 32 channels/group, asymmetric g = 2 and the strided g = 2 / 4 cells.
    - Everything else (HEIGHT / BLOCK / WIDTH, and groups that are not whole tiles: ResNeXt g = 32 with 4 ch/group, AlexNet conv2 with 48, g = 4 / 8 with 16) runs as a dense conv over a block-diagonal weight built in prepare. **Waste: g× the MACs and weight bytes** of a per-group conv (32× for ResNeXt g = 32; these are small convs: 28×28×128 runs in 37.6 µs).
  - Reused: the R0 / sharded gather, the window / halo fetch, every compute path, the writer's column-tile zero fill, depthwise's `win_chan_offset`. Added: `cb_stick_stage`, gather CT args (`DIL`, staging, `window_zero_bytes`, `src_bytes`, `in_row_tiles`), writer CT `C_out`, host `native_groups`, a dilated-row window.
- Accuracy achieved: every cell at its golden dtype gate (fp32 PCC ≥ 0.999 / rel-RMS ≤ 0.02; bf16 0.995 / 0.05; bf8b 0.99 / 0.10). This includes the atrous + grouped fp32 loose case, the half-tile / 1.5-tile depthwise cells, the C_in = 1 / 3 starters, and SDXL conv_in / conv_out (HEIGHT bf16 × bf8b, BLOCK bf16, DRAM TILE).
- Golden test progress:
  - Refinement 8 cells (`-k "c_in_unaligned or c_out_unaligned or both_unaligned or dilated or grouped"`): 936 passed, 0 failed (skips are INVALID bf8b × RM).
  - Regression slices: sharded tile-aligned + all loose cases 398 passed; interleaved tile-aligned (strided / depthwise / k5 / k7 / p1) 620 passed; 0 failed.
  - Unit directory: 143 passed + `test_conv2d_nhwc_r8.py` 22 passed.
- Perf (BH p150b, per-core max kernel cycles at 1350 MHz; one run each):
  - Guards are unchanged within noise: interleaved bf16 relu 101.6k → 103.3k; ResNet layer1 HEIGHT 38.8k → 38.8k; SDXL L1 BLOCK bf16 × bf8b 257k → 259k; depthwise interleaved 56×56×128 46.4k → 46.4k.
  - New cells:
    - SDXL conv_in HEIGHT: 45.3 µs (achievable 15.0 µs).
    - SDXL conv_out HEIGHT: 55.4 µs (achievable 43.6 µs).
    - SDXL conv_out BLOCK (8×8, 40-channel slices): 1.51 ms.
    - ResNet conv1 224×224×3 k7 s2: 382 µs interleaved, 83.5 µs HEIGHT.
    - AlexNet conv5 (native g = 2): 32.6 µs.
- Issues encountered:
  - BH DRAM reads must land at the source's 64 B alignment, which the 16 B-granule window cannot honour (hence the staging).
  - The native-group core split could exceed the grid (fixed: the m split shrinks).
  - The d = 18 window overflowed L1 (fixed with the dilated-row window).
  - The remote JIT compile server failed with an SFPI toolchain mismatch (`__xtt_vector8`); all runs compiled locally (`TT_METAL_JIT_SERVER_ENDPOINT` unset).
- Findings (not follow-ups; perf work for the BLOCK / perf refinements):
  - The SDXL conv_out BLOCK cell walks to bk = 1 (96 K-blocks per m-block). `cb_act_rm` is one whole m-block deep (`ACT_RM_KBLOCKS·bm` tile-rows) at bm = 32, so it is handshake-bound. A smaller bm with a larger bk would cut the rounds.
  - ResNet conv1 interleaved is DRAM-request bound: 98 cores × ~1.8k one-stick reads, ~3.6× halo re-read. Same finding as Refinement 4's depthwise; HEIGHT runs it 4.6× faster.
- Tests added:
  - `test_conv2d_nhwc_r8.py`: 22 cases, one per kernel path × axis value.
  - `test_conv2d_nhwc.py::test_conv2d_nhwc_rejects_unsupported_dilation` became `test_conv2d_nhwc_rejects_excluded_cell` (TILE × depthwise), since dilation is now supported.

## Refinement 9 — Speed up the perf-flagged ResNet50 BLOCK convs
- Date: 2026-11-02
- What was done (R4 BLOCK path; no SUPPORTED change):
  - **Baseline breakdown** (layer3 3×3, 115k cycles): the per-pixel band gather of a one-channel-tile slice (~1.3k cycles per tile-row, RISC-issue bound at ~46 cycles per local read), compute's eager tilize of the next own block (it waited ~11k on the reader's gather and stalled the row), and per round a full ack → data round trip on both the act and the weight channels.
  - **Tap-major gather on BLOCK / WIDTH** (`TAP_MAJOR_LAYOUTS`): stride-1 convs reuse the HEIGHT tap-major gather. It is kept only where it does not shrink the K-block: 2-tile taps force an even bk, so the band walk wins on SDXL (bk 3) and layer4 (bk 9).
  - **Just-in-time own-block tilize** (`ACT_TILIZE_LEAD = 2`, compute CT 37): own K-block kb is tilized 2 rounds before its multicast, not as soon as the send slot frees.
  - **Early act acks**: the receiver acks round r + 1 right after round r lands, before weight work.
    - kernel_lib: `ReceiverPipe::acknowledge()` / `wait_delivery()`, the two halves of `receive()`, additive.
  - **Pipelined weight stream** (writer):
    - The weight CB is a ring addressed ahead of its write pointer.
    - DRAM reads run up to depth − 3 blocks ahead of the landing, one read transaction id per block.
    - Receivers ack on slot free and land one block ahead of the matmul. The sender multicasts two blocks ahead, after its act round.
  - **Slice-major BLOCK weights** (`weight_slice_tiles`, prepare + writer): column slice c's [K_t, nt_s] tiles are a contiguous page run, so a K-block read spreads over every DRAM bank. Row-major put each column's tiles in 1–2 banks, and layer4's weight stream was bank-bound. The tensor shape is unchanged.
  - **Rotating column weight sender** (`WEIGHT_ROTATING_SENDER`): block w is read and multicast by row w mod g_m. It is gated on weight K-block bytes ≤ 0.25 × act K-block bytes (`WEIGHT_ROTATE_MAX_RATIO`). Layer4's heavier stream swung 81–95k rotating vs a steady 84k fixed.
  - Parked at a byte-identical default: `ACT_CHANNELS = 1`. At 2, rounds alternate between two act Mcast channels with a third landing slot, and the next round's owner sends before waiting. Concurrent multicasts in one row measured 116–131k cycles.
  - Removed after measuring: per-column stateful weight reads (R0 interleaved 101k → 107k).
  - Reused: the HEIGHT tap-major gather, every CB, the Mcast channels / helpers, matmul / tilize helpers. Added: the receiver ack/wait split (kernel_lib), the weight ring pipeline, a rotating weight channel, the optional second act channel, compute JIT tilize, and prepare's slice-major page order.
- Perf (BH p150b, AICLK 1350 MHz, per-core max kernel cycles, bf16 × bf8b, bias, 3-run medians). Before → after (achievable):
  - layer3 downsample: 95.2k → 82.7k (66.5k)
  - layer3 3×3: 115.4k → 77.1k (64.6k)
  - layer4 3×3: 90.4k → 84.0k (48.2k)
  - `device_l1_peak_bytes` unchanged: 413,696 / 352,384 / 392,064 B vs targets 370,416 / 238,640 / 161,296. The excess is the four own-K-block act copies. Halving bm to fit costs 77k → 100k on layer3 3×3 (twice the exchange rounds).
- Guard set (one run each, before → after, cycles):
  - interleaved bf16 relu: 101.7k → 102.1k
  - BLOCK 28×28×256 no bias: 40.7k → 31.1k
  - WIDTH strided: 97.2k → 94.6k
  - SDXL L1 BLOCK: 260.5k → 237–241k
  - HEIGHT layer1 / layer2: 38.8k / 35.7k → 38.9k / 35.7k
  - depthwise interleaved / WIDTH / BLOCK: 46.3k / 14.2k / 20.0k → 46.1k / 14.3k / 19.9k
- Bottleneck now:
  - Act exchange: 24 rounds × ~2.6k. About 1.5k is payload at ~40 B/cycle into each core; about 1.1k is the ack/flag floor (all payloads stubbed: 30k).
  - Reader gather: about 46 cycles per local read, 32 reads per tile-row at stride 2. About 15k passes before the first round can go out.
  - layer4: the weight DRAM stream (reads stubbed: −20k) and the act payload.
- Next lever: split exchange rounds across both NoCs (the reader RISC multicasting odd rounds on NOC_0), so two rounds are in flight on disjoint links. That would also let the act blocks shrink toward the L1 target. Not done here: it moves the multicast landing into the reader alongside the gather.
- Accuracy achieved: the three ResNet BLOCK cases have PCC ≥ 0.99988 and rel-RMS ≤ 0.0203 (bf16 × bf8b, gate 0.995 / 0.05). Every unit test passes at its prior tolerance.
- Golden test progress:
  - `test_golden.py -k "WIDTH or BLOCK or loose"`: 242 passed, 72 skipped, 0 failed. This includes the three ResNet BLOCK perf cases and the SDXL BLOCK case.
  - `test_translated.py -k BLOCK`: 147 passed.
  - `test_golden.py -k "INTERLEAVED and BFLOAT8_B and with_bias and ROW_MAJOR and tile_aligned"`: 228 passed, 342 skipped.
  - Unit directory: 165 passed.
- Issues encountered:
  - Passing absolute rounds to the act receiver hung WIDTH with receive-only sinks: the rotation length is the group size, not the slice count. Fixed by passing the owner phase.
  - The Counter data-ready signal cannot carry out-of-order deliveries, and its sender barriers made it slower (113k).
- Tests added: `test_conv2d_nhwc_perf_probe.py::test_perf_probe_resnet_block` (three cases; `R9_CASE` env filter) and `test_l1_peak_resnet_block`.

## Perf 1 — tournament round 1 (BLOCK act exchange / gather / weight stream)
- Date: 2026-11-02. No SUPPORTED change. BH p150b, AICLK 1350 MHz. Numbers are per-core max TRISC_1 KERNEL cycles from `profile_log_device.csv`; the ops-CSV durations are broken by cross-core clock skew on this box. ns = cycles / 1.35.
- Focus case: the first `attention: PERF FOCUS` loose case, SDXL level-1 3×3. Input (1,64,64,640), C_out 640, BLOCK 10×10, shard (410,64), bf16 × bf8b, bias. Achievable 149,049 ns (201.2k cycles). Precision contract unchanged: LoFi, fp32 DEST.
- Plan: S = 10 slices; per core bm 13, bn 2, bk 3. That gives 60 act rounds of 80 KB along each grid row, and a rotating weight sender down each column.

### Instrumentation (permanent)
- Added `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`. `MaybeDeviceZoneScope` is an alias of `DeviceZoneScopedN` and costs nothing with the profiler off. The header named in the brief did not exist in this tree.
- Zones:
  - reader: `reader_window_barrier`, `reader_gather_kblock`, `reader_rm_reserve` (wait split from work), `reader_wt_drain`
  - compute: `compute_matmul`, `compute_tilize_own` / `compute_tilize`, `compute_bias_untilize`
  - writer: `writer_act_reserve`, `writer_act_send`, `writer_act_wait`
- The focus case uses 244 of the 250 markers per RISC.
- Summary tool: `perf_experiments/tools/zones.py`. One-run wrapper: `prof.sh`.

### Measured breakdown (focus, round start: 237.9k cycles ≈ 176.2 µs)
Zones (mean per core):
| zone | cycles |
|---|---|
| compute_matmul (whole helper call) | 232k |
| writer_act_wait (54 rounds) | 137.6k |
| reader_gather_kblock | 187k, of which reader_rm_reserve 90k |
| compute_tilize_own (6 calls) | 27k |
| bias + untilize | 5k |

Cumulative payload ablation (sync kept):
| step | cycles |
|---|---|
| full | 237.3k |
| − act payload | 196.5k |
| − weights | 197.1k |
| − gather | 179.6k |
| − tilize | 167.3k |
| − matmul (everything stubbed) | 84.8k |

Solo ablations: − matmul 224.4k; − weights 224.0k; − gather 217.2k.

Reading:
- The matmul payload is about 82k cycles (≈ 17.6 cycles per tile, about 90% of LoFi peak). The whole-op compute floor at full utilisation is 73.7k.
- With every payload stubbed, an 85k handshake floor remains. It runs serially with the matmul instead of hiding it.
- Ranked headroom: the act exchange (payload plus serialised row multicasts), then the weight stream sharing the writer's loop and NOC_1, the gather startup bubble, and tilize.
- The matmul is roofline-gated and got no ideas.

### Portfolio and verdicts (artifacts in `perf_experiments/<slug>/`, brief in `perf_experiments/BRIEF.md`)
| idea | verdict | focus (cycles) | domain / notes |
|---|---|---|---|
| `gather_fast`: padding kernel rows stay on the fast run path (zero runs); interior runs alternate two NoC cmd buffers | **WIN** | 240k → 229.3k | Applies everywhere. ResNet layer3 ds 87.7k → 80.1k; layer4 84.0k → 81.2k; HEIGHT layer2 ds 41.3k → 35.4k; rest flat. Exceptions: none. |
| `weights_on_reader_risc`: the streamed weight ring (DRAM reads, column multicast / receive) runs on NCRISC (NOC_0) as a non-blocking step polled by the gather | **WIN** | 239–241k → 225.5k (alone) | layer4 −14%, BLOCK 28×28×256 fp32 −36%, layer3 3×3 −6%, no-bias −9%. Measured regression on layer3 ds (+4.5%). |
| `dual_noc_act_mcast`: split each act round across NOC_1 (writer) and NOC_0 (reader), two flags | WIN, marginal | 238.7k → 229.8k (−3.7%, inside the ±4% noise) | layer3 ds +18%. Raw multicast. Conflicts with `weights_on_reader_risc` (both load the reader and NOC_0). **Not graduated.** |
| `fewer_rounds_bigger_kblock`: bk 3 → 6/9/18 | **BLOCKED** (L1) | bk 6 tap-major 195.8k (−20%) | Costs about +330 KB per core (922,880 B peak vs the 618,992 B target). The only variant that fits (send buffer aliased into the landing ring) runs 252.9k, no win. bk 18 runs out of L1. layer3 3×3 bk 9 regresses +6.7%; layer4 bk 18 −17%; 28×28×256 −16%. |
| `act_pipeline_depth`: ack ahead / deeper landing ring | NULL | ring depth 3: 237.0k; 2 channels: 360–415k | A Flag channel can carry only one outstanding round. Two concurrent row multicasts regress heavily. |
| `handshake_floor`: cheaper act round protocol | NULL | best full-case variant 238.1k | Diagnosis: the ack round trip is not on the critical path. The floor is the next sender's loop work (the weight landing at about 600 per round) plus about 350 cycles of flag latency. Any overlap of a weight or second row multicast with the act multicast on NOC_1 costs +30–130%. Batching the weights 3 per handshake gives floor −17% but full case +7.6%. |
| `tilize_own_fast`: cheaper own-block tilize | NULL | ±0.3% | The 27k zone is mostly a 17k wait for the first gather. The real tilize is about 11k, and a free tilize would save only 5.7k. The BH fast-tilize path is already in use. Split pieces: +3.4%. |

### What graduated
- **`gather_fast`** (commit 18de8479fa3, `kernels/conv2d_nhwc_gather.hpp`)
  - One unqualified path. The interior/padding fast-path condition and the per-read loop it replaced are deleted.
  - Carve-outs:
    - The BRISC split-gather instance keeps one cmd buffer (its buffers serve the writer's writes; structural).
    - Under `WR_WEIGHTS` the gather uses one cmd buffer. Its transaction-id barrier does not cover reads issued from the second buffer: measured hang, readers stuck in `noc_async_read_barrier_with_trid`.
  - On the focus case only the padding part of this change remains active.
- **`weights_on_reader_risc`** (commit 624cc0c529c; descriptor `wr_on`, `kernels/conv2d_nhwc_reader.cpp`, gather `Poll` hook, writer `WR_WEIGHTS`)
  - Domain: every K-exchange core set with streamed multicast weights (BLOCK; any S, including one slice). Written as carve-outs, not an allow-list.
  - The writer-side ring stays as the path for R0 interleaved, WIDTH (no weight multicast) and HEIGHT resident weights.
  - Carve-outs:
    - HEIGHT streamed fallback: its receive-only sink cores carry no reader kernel to host the ring (cannot).
    - Measured regression kept on the unified path: ResNet layer3 ds 79.8k → 84.1k (+5.4%, 2 vs 3 tight runs). The other strided BLOCK perf case (SDXL level-0 ds) is flat at 106.4k → 106.7k, so stride is not the predicate. No principled host-side test for "reader-bound gather" was found, so no shape fence was added. Net for layer3 ds over the round: 87.2k → 84.1k.
  - Integration fix beyond the subagent's variant: the weight step now reports any NoC activity (send / land / ack), not only reads, so the gather always re-arms its read state.
  - `WEIGHTS_ON_READER` is an A/B module constant, default True.

### Whole-op before → after (cycles; one run each unless noted)
| case | round start | after Perf 1 | achievable |
|---|---|---|---|
| **SDXL L1 BLOCK (focus)** | 237.9k (176.2 µs) | **215.1k (159.3 µs), −9.6%** | 201.2k (149.0 µs) |
| ResNet layer3 ds BLOCK | 87.2k | 84.1k (3 runs) | 66.5k |
| ResNet layer3 3×3 BLOCK | 78.7k | 74.3k | 64.6k |
| ResNet layer4 3×3 BLOCK | 84.0k | 66.0k | 48.2k |
| SDXL L0 ds BLOCK | — | 106.7k (new probe `test_perf_probe_block_sdxl_l0_ds_bf16_bf8b`) | 94.2k |
| guard BLOCK bf16 no-bias | 30.8k | 27.9k | — |
| BLOCK 28×28×256 fp32 | 77.7k | 48.9k | — |
| guard WIDTH strided | 94.6k | 94.6k | — |
| guard interleaved bf16 relu | 100.0k | 99.3k | — |
| ResNet layer2 ds HEIGHT | 41.3k | 35.6k | 41.0k |
| ResNet layer1 3×3 HEIGHT | 38.8k | 38.5k | 38.0k |

- Precision is identical before and after on every case (focus PCC 0.999776, rel-RMS 0.02017).
- Focus L1 peak: 590,336 B (target 618,992 B), unchanged.

### No-regression / correctness
- Guard set (one per kernel path × layout × placement): no regression beyond noise except the recorded layer3 ds.
- `--dev` (watcher + asserts): 12/12 BLOCK perf probes pass.
- Unit directory: 197 passed.
- Golden `eval/golden_tests/conv2d_nhwc/`, full: 4,578 passed, 0 failed (rest skipped / xfailed). Interleaved 3,671 in 12 pytest-split groups, BLOCK 298, WIDTH / HEIGHT / loose 609.

### Bottleneck now (round 2)
- On the focus case the writer spends 169k of 207k waiting on act rounds. Matmul is 82k, and the gather has slack (rm reserve 61k).
- The act exchange is the bound: 60 serialized 80 KB row multicasts.
- Remaining levers:
  - Fewer rounds: bk 6 was −20%, but blocked at +330 KB L1. A row-chunked send that overlaps the owner's tilize with the transfer could make the aliased-send variant fit.
  - Act payload bandwidth: the dual-NoC split, re-measured now that the weights are on NOC_0.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `noc_async_read_with_state` (dataflow_api stateful read; no kernel_lib helper covers it) | capability | No API issues a run of fixed-length local reads over two cmd buffers. The stateful read is tied to `read_cmd_buf` and rewrites the length register on every read, so each issue is a poll plus 4 writes on one buffer, with no overlap of one buffer's poll with the other's request. Also, a second-buffer read is not counted by `noc_async_read_barrier_with_trid` even when tagged, and the trid API (`noc_async_read_set_trid`) only tags `read_cmd_buf`. | 171.2k (231.1k cyc, padding fix only, one buffer) | 169.9k (229.3k cyc) | `kernels/conv2d_nhwc_gather.hpp:181` (`run_issue` / `local_read_run`) |
| `dataflow_kernel_lib` Mcast pipes (`SenderPipe::send` / `ReceiverPipe::wait_delivery` / `acknowledge`) | capability | No non-blocking readiness query: no `acks_ready()` / `poll_delivery()` / `try_send()`. A weight ring polled from inside another stage's loop has to peek the channel's DATA_READY / CONSUMER_READY semaphores through `detail::make_mcast_semaphore` and the positional bindings, then call the blocking pipe with its condition already met. | 170.1k (229.7k cyc, ring on the writer with blocking pipes) | 159.5k (215.3k cyc) | `kernels/conv2d_nhwc_reader.cpp:90` (peeks at :137 / :176) |
