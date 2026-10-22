# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-21
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - **Algorithm**: regime R1 (row-parallel, row-resident). Tile-rows are split over the full Tensix grid
    (`split_work_to_cores`, `row_wise=True`). Each block of `block_rows` tile-rows is resident in L1 for
    both passes, so x crosses DRAM once.
  - **Compute** is all kernel-lib helpers: Σx² as one DEST-accumulated FPU multiply per tile-row; reduce
    with a `×1/W`, `+ε`, `rsqrt` post-op; then γ (Row broadcast) and `inv_rms` (Col broadcast) chains.
  - **`Layout` handling**: `tilize` / `untilize` in-kernel for `Layout::ROW_MAJOR`.
  - **Padding**: W-pad columns, pad tiles and RM tail bytes are zeroed by the reader.
  - **Dispatch**: one `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**:
  - dtype = [float32, bfloat16]
  - fp32_dest_acc_en = [True]
  - layout = [TILE, ROW_MAJOR]
  - alignment = [tile_aligned, w_non_aligned, h_non_aligned]
  - rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]
  - gamma_dtype = [float32, bfloat16, "none"]
  - gamma_layout = [ROW_MAJOR, "none"]
  - memory_layout = [INTERLEAVED]
  - EXCLUSIONS = [{float32, fp32_dest_acc_en=False}]
- **Accuracy achieved** (HiFi4, fp32 DEST, γ on; 4 shapes × 2 dtypes × 2 `Layout`s via
  `test_rms_norm_precision_baseline.py`):
  - bf16: PCC ≥ 0.999998, max_abs_err ≤ 0.048, rel_rms_err ≤ 0.0020, ≤ 1 bf16 ULP.
  - fp32: PCC 1.000000, max_abs_err ≤ 0.034, rel_rms_err ≤ 0.0020. tf32-limited: ~2·10⁴ fp32 ULP mean,
    with a systematic −0.14 % bias from tf32 operand truncation in the FPU, confirmed by host simulation;
    not a structural scale bug.
- **Golden suite at Phase 0**: 436 / 436 supported cells passing (0 supported_fail, 0 xpass_drift, 0
  xfail_wrong_mode); 6481 xfail_expected, all mapped to Refinements 1, 2, 4 and 5; 33900 invalid_skipped
  (per `verifier_report.json`, summary in `verifier_report_summary.json`).
- **Issues encountered / verifier fixes**:
  - γ row-0 expansion on the reader switched from RISC-V word-by-word `volatile` copies to local L1→L1 NoC
    reads: 1.15–1.28× on γ-bearing decode / 640-row shapes, no regression elsewhere (Blackhole,
    1350 MHz). The remaining reader serialization (it blocks the `x_depth = 2` prefetch) is carried by
    Refinement 1.
  - DRY: intermediate-CB tile bytes are single-sourced (`intermediate_format()` → `Geometry.T_interm`),
    used by both the CB declarations and the L1 `footprint()` solve. The reduce-scaler format is one
    constant.
  - `l1_ledger.md`:
    - currency fix (`T32` → `T_interm`);
    - added the missing sharing justification for `cb_sumsq_partial` ↔ `cb_x_gamma` / `cb_y_tiles`;
    - recorded the isolated `MIN_BLOCKS_PER_CORE = 2` measurement (1.02–1.05× on many-row γ shapes).
  - INVALID audit flagged 5 non-structural, author-scoped entries in `feature_spec.py`, reported to the
    user and not edited.
- **Tests added**:
  - `test_rms_norm.py` (acceptance, planner);
  - `test_rms_norm_perf.py` (implementer);
  - `test_rms_norm_precision_baseline.py` (verifier);
  - `test_rms_norm_perf_baseline.py` (verifier: interleaved `_perf_case` shapes at the nearest Phase 0
    config, ± γ, for profiler runs).

## Refinement 1 — TILE-layout gamma, with γ off the reader's critical path
- **Date**: 2026-10-21
- **What was done**:
  - `SUPPORTED["gamma_layout"]` gains `ttnn.TILE_LAYOUT` (native, both input `Layout`s, `TensorMemoryLayout::INTERLEAVED`).
  - **γ supply moved to the writer RISC-V** (perf lamp L4) for both γ `Layout`s. The reader now moves only x, so it can prefetch block k+1 while block k is scaled.
  - **TILE γ staging fill**: the two 16-element row-0 face segments of each γ tile (face 0 row 0, face 1 row 0) go into 64 B slots of `cb_gamma_row`. A slot is a multiple of every arch's DRAM read alignment, so source and destination keep matching low address bits. Rows ≥ 1 never leave DRAM. Padding needs no masking: pad columns of γ multiply zeroed x pad columns.
  - **Resident γ fast path** (`GAMMA_RESIDENT` knob, `choose_gamma_resident()`):
    - all `Wp` row-0 tiles are expanded once per kernel, in one reservation pushed C tiles at a time;
    - compute waits cumulatively and indexes with `TileAddressing::Offset` (base `c·C`);
    - chosen only when it fits on top of the streaming-γ blocking, so `_WIDE` shapes stay on the streaming path.
  - **Streaming γ** (fallback): one C-tile chunk per (block, chunk) step, pushed one step before the matching output chunk is drained (deadlock-free; see the writer header). The TILE staging fill is issued lazily, one chunk ahead.
  - Expansion uses stateful single-packet NoC reads.
  - **`x_depth = 1` when every core walks a single block** (`rows_g1 ≤ 1`). The second x buffer had nothing to prefetch; freeing it lets decode / 640-row shapes take the resident-γ path.
  - Reused: `cb_gamma_row` staging + row-0 expansion, the `scale_by_gamma` chain, the footprint solve. Added: TILE fill, writer-side supply, resident variant.
- **Accuracy achieved**: TILE γ at the golden thresholds (bf16 PCC ≥ 0.995 / rel-RMS ≤ 0.04; fp32 PCC ≥ 0.999 / rel-RMS ≤ 0.02).
  - Shapes: 1×1×32×32 … 1×1×32×7168, 2×3×33×1025, 1×1×2048×1024, 1×1×32×16384.
  - Covers mixed γ dtype and poisoned γ padding (1000).
- **Golden test progress** (interleaved, fp32 DEST, non-bf8b slices):
  - TILE γ: 282 / 283 pass. The one failure is `(1,1,32,32768)`: it is refused by the regime-R1 residency predicate and owned by Refinement 4, as expected.
  - RM γ: 280 / 280 pass.
  - no γ: 140 / 140 pass.
- **Perf** (`test_rms_norm_perf_baseline.py`, bf16 TILE input, HiFi2, fp32 DEST, Blackhole, device kernel ns):

  | shape | Phase 0 γ (RM) | R1 γ RM | R1 γ TILE | no γ | reader NCRISC γ / no γ |
  |---|---|---|---|---|---|
  | 32×1024 | 9,197 | 8,832 | 8,764 | 6,856 | 1,680 / 1,705 (P0: 3,439) |
  | 32×2304 | 15,768 | 15,310 | 15,417 | 11,101 | 2,878 / 2,919 (P0: 9,059) |
  | 32×5120 | 30,642 | 30,189 | 30,202 | 21,039 | 5,423 / 5,517 (P0: 22,027) |
  | 32×7168 | 41,412 | 40,892 | 40,957 | 27,653 | 7,422 / 7,410 (P0: 32,653) |
  | 640×6144 | 54,457 | 53,230 | 51,016 | 47,313 | — |
  | 640×7168 | 63,227 | 61,867 | 59,265 | 55,073 | — |

  - Many-row prefill RM γ is within run-to-run noise (±2 %) of a same-session Phase 0 run: 8192×7168 566 vs 574 µs, 8192×1024 87.6 vs 85.9 µs.
  - Prefill TILE γ is 3–10 % slower than RM γ. Ablation: stubbing only the TILE row-0 DRAM fill returns it to the no-γ time (8192×5120: 399 vs 398 µs), so the cost is 120 Tensix cores × 2·Wt tiny γ reads. The fix is G-MC (Refinement 6).
- **Issues encountered**:
  1. One device wedge (PCIe ID 1 read 0xffffffff at device open after a profiler run), cleared through the wrapper's dirty-flag reset. Not kernel-related.
  2. First cut pushed the resident window in one shot, and TILE γ at 32×5120 was 8 % slower than RM γ: compute stalled on γ. Fixed by the chunk-by-chunk push.
  3. On-demand per-chunk fill in streaming mode put DRAM latency in the output-drain loop. Replaced by the lazy one-chunk-ahead fill.
- **Tests added**:
  - `test_rms_norm_gamma_tile.py`: TILE γ × input `Layout` × dtype, mixed γ dtype, poisoned γ padding, forced-streaming pins for both γ `Layout`s, and a wide-W streaming pin.
  - `test_rms_norm_perf_baseline.py` gains a `gamma_tile` case.

## Refinement 2 — Numerical configurability: 16-bit DEST, bfloat8_b input and gamma
- Date: 2026-10-21
- What was done:
  - SUPPORTED gains `fp32_dest_acc_en: False`, `dtype: bfloat8_b` and `gamma_dtype: bfloat8_b`. `{float32, fp32_dest_acc_en=False}` stays the native EXCLUSION.
  - New **structural-capability EXCLUSION** `{dtype: bfloat8_b, alignment: w_non_aligned}`. The reader zeroes the W-pad columns of the last tile per element. Inside a Bfp8_b tile that would be exponent-section work (16 datums share one exponent), and it is not built. Pad *tiles* (`Wp > Wt`) are zeroed as raw bytes, which is a valid Bfp8_b zero tile, so they stay supported. `bfloat8_b × h_non_aligned` works natively: rows are independent and the shared exponent spans a face row.
  - Intermediate CB format follows the DEST width (`intermediate_format()`, single source for the CB declarations and `footprint()`): Float32 under fp32 DEST, Float16_b under 16-bit DEST. Before this change the 16-bit DEST case used the input dtype. The format is never Bfp8_b, even for a Bfp8_b input: raw Σx² lanes and x·γ products would lose their per-datum exponent.
  - **Format-aware Bfp8_b γ row-0 extraction** (writer γ supply, both the resident and the streaming path). Byte layout of a Bfp8_b tile: 64 B exponent section (face-major, one exponent per 16-datum face row), then 1024 B of 1 B mantissas (face-major). Row 0 of faces 0 and 1 needs three 32 B segments:
    - tile byte offsets 0 (both exponents), 64 and 64 + 256 (the mantissa rows);
    - every offset is a multiple of the DRAM read alignment.

    The segment list is single-sourced. The host side is `gamma_row_segments()` / `gamma_segment_bytes()`; the kernel side is `gamma_segment_offset()` / `gamma_seg_bytes`. Float32 / Float16_b keep their two `16·es_g` B segments. The staging page is `segs · GAMMA_SEG_SLOT` (192 B per γ tile for Bfp8_b, 128 B otherwise).
  - `Tensor.element_size()` raises for Bfp8_b. The host now uses `datum_bytes(dtype)`, which returns 1 for Bfp8_b; that value is only consumed on RM / W-pad paths, which Bfp8_b never takes. The reader has a `static_assert` that pins this.
  - Reused: the whole compute kernel (zero changes: the helpers reconfigure data formats), the γ staging + expansion machinery, the footprint solve. Added: the N-segment generalization of the γ fill/expand, `datum_bytes`, the EXCLUSION.
- Accuracy achieved (all at the golden thresholds; full table in `tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md`):
  - Precision matrix, 328 cases. Shapes 32×32 … 256×2048, 32×7168, and W/H/both non-aligned; dtype ∈ {bf16, fp32, bfp8}; DEST ∈ {fp32, 16-bit}; fidelity ∈ {HiFi4, HiFi3, HiFi2, LoFi}; distribution ∈ {uniform, normal}. Minimum PCC 0.99952 (LoFi); minimum PCC at HiFi2 or better: 0.99991.
  - bf16, 16-bit DEST, HiFi2, bf16 TILE γ (the Refinement 3 config), 32×{1024, 2304, 5120, 7168}: PCC ≥ 0.9995 (soft gate). Design lamp L8 (Σx² of up to 224 tiles in a 16-bit DEST accumulator) did not bite at these widths.
  - bfloat8_b input: PCC ≥ 0.9999 at HiFi2 or better; relative RMS error ≤ 1.4e-2.
- Golden test progress (`-k`-filtered slices, interleaved):
  - `BFLOAT8_B and INTERLEAVED`: 450 passed / 0 failed. Also 70 xfailed (the `{float32, fp32_dest_acc_en=False}` EXCLUSION) and 4520 skipped (INVALID).
  - 16-bit DEST non-bf8b (`INTERLEAVED and False and not BFLOAT8_B`, cartesian + loose, incl. all interleaved `_perf_case` and resilience cells): 454 passed / 0 failed / 350 xfailed (same EXCLUSION).
  - `test_translated.py`: 52 passed / 54 xfailed, all of them sharded placements.
- Perf at the exact interleaved `_perf_case` config (bf16, HiFi2, 16-bit DEST, bf16 TILE γ; `test_rms_norm_perf_baseline.py::test_rms_norm_perf_exact_config`, Blackhole, device kernel ns). This is the Refinement 3 starting point:

  | shape | device ns | Tensix cores |
  |---|---:|---:|
  | 32×1024 | 8,610 | 1 |
  | 32×2304 | 14,894 | 1 |
  | 32×5120 | 28,770 | 1 |
  | 32×7168 | 40,258 | 1 |
  | 640×6144 | 50,501 | 20 |
  | 640×7168 | 57,256 | 20 |
  | 7168×1024 | 80,850 | 120 |
  | 7168×2048 | 157,299 | 120 |
  | 8192×1024 | 88,550 | 120 |
  | 8192×2304 | 194,795 | 120 |
  | 8192×5120 | 435,122 | 120 |
  | 8192×7168 | 588,666 | 120 |

- Issues encountered:
  1. `Tensor.element_size()` raises `ValueError` for bfloat8_b. Fixed with `datum_bytes()`.
  2. A namespace-scope `static_assert` on the γ tile format fired on no-γ kernels, where `cb_gamma_tiles` is unconfigured. Fixed by gating the assertion on `has_gamma`.
- Tests added:
  - `test_rms_norm_numeric_config.py` (69 cases). It covers:
    - bf16 with 16-bit DEST × input `Layout` × γ `Layout` on aligned and non-aligned shapes, plus the Refinement 3 config at PCC 0.9995;
    - bfloat8_b input under both DEST widths, with no γ, bf16 γ and bf8b γ;
    - bfloat8_b TILE γ with bf16 / fp32 input, both input `Layout`s, and the forced-streaming path;
    - the `{bfloat8_b, w_non_aligned}` refusal.
  - `test_rms_norm_precision_matrix.py` and `precision_matrix_results.md`.
  - `test_rms_norm_perf_baseline.py::test_rms_norm_perf_exact_config`.

## Refinement 3 — Speed up the interleaved perf loose cases at their exact config
- Date: 2026-10-21
- What was done: Perf only; SUPPORTED is unchanged. All measurements on Blackhole at 1350 MHz.
  - **Classified the bottleneck first**: 32×7168, `CKL_ELTWISE_CHAIN_SKIP_COMPUTE` ablation plus per-phase device zones on all five RISC-Vs.
    - The baseline was **two co-binding stages**: compute (~50 k cycles) and the writer (~48 k).
    - With compute stubbed, the op still took 46 k cycles. The writer's resident-γ supply alone took 32.5 k cycles: 224 tiles × (2 DRAM reads + 2 L1→L1 expansion reads of 32 B), and it drained no output until it finished.
  - **Lever 1 — γ direct read (writer, lamp L4)**: TILE γ on the resident path reads each γ tile's row-0 byte prefix (`gamma_row0_prefix_bytes`: 544 B bf16 / 1088 B fp32 / 352 B Bfp8_b) from DRAM straight into its `cb_gamma_tiles` page. Source and destination are both tile-aligned.
    - One request per tile. No staging row (`cb_gamma_row` is no longer allocated on this path) and no expansion.
    - γ supply 32.5 k → 11 k cycles, now hidden behind `square_accumulate`.
    - Knob `GAMMA_DIRECT_READ` ∈ {PREFIX (default), SEGMENTS, OFF}. SEGMENTS (one read per row-0 segment, in place) is 1 % better at 640 rows and 2–6 % worse on prefill.
  - **Lever 2 — compute block for the scaling stages** (`compute_block_size`): `scale_by_gamma` and `scale_by_inv_rms` run `SCALE_BLOCK_TILES = 8` tiles per DEST window (chain-clamped to 4 under fp32 DEST) instead of one.
    - `cb_x_gamma` is waited Upfront and popped AtEnd (fully packed by the γ chain before the inv_rms chain starts).
    - Outputs are reserved and pushed PerOuter (C pages = the writer's / untilize's quantum).
    - The skip-compute ablation showed ~45 cycles/tile of pure dst-sync + CB handshake per chain. Blocking cut γ 67 → 37 and inv_rms 69 → 47 cycles/tile.
  - **Lever 3 — bank stagger** (lamp L9, `BANK_STAGGER`): a compile-time `RMS_BANK_STAGGER` define rotates each core's x-read, output-write and γ-read issue order by `core mod NUM_DRAM_BANKS`.
    - Measured flat on every shape (prefill and 640-row are DRAM-bandwidth-bound, not bank-lockstep-bound), and +1 µs on decode from the rotation arithmetic.
    - Ships **off**. Off compiles the plain loops (byte-identical issue order), so it remains a live knob.
  - **Probes measured and not built**:
    - L10 data-format reconfig off on every scaling-chain operand: 30.73 vs ~30.8 µs at 32×7168, noise;
    - NoC swap (reader NoC1, writer NoC0) on 8192×7168: 580.8 vs 580.9 µs;
    - L6 FPU dest-reuse fusion: not built (`DestReuseBinary` has no broadcast operand, and the catalog measures it as a loss);
    - L7 col-0 rsqrt scoping: not built (the finalize phase is 1.8 k of ~40 k cycles).
  - Reused: the γ supply structure (resident reservation pushed chunk by chunk), the scaling chains (same helpers and operands, only policies and shape blocking changed), the footprint solve (`gamma_row_pages()` feeds both the CB list and `footprint()`). Added: `gamma_row0_prefix_bytes()`, `gamma_direct()`, `gamma_row_pages()`, the knobs `GAMMA_DIRECT_READ` / `SCALE_BLOCK_TILES` / `BANK_STAGGER`, a compute-only CT arg (index 14), and shared-prefix CT index 13 (`gamma_direct`).
- Accuracy achieved: unchanged numerics.
  - Exact config (bf16, 16-bit DEST, HiFi2, bf16 TILE γ): PCC ≥ 0.9995 soft gate on all 12 interleaved perf shapes.
  - Golden thresholds (bf16 PCC ≥ 0.995 / rel-RMS ≤ 0.04; fp32 PCC ≥ 0.999 / rel-RMS ≤ 0.02) on every knob setting in `test_rms_norm_perf_knobs.py`.
  - Precision matrix 328/328 pass.
- Perf (device kernel ns, exact config, `test_rms_norm_perf_baseline.py::test_rms_norm_perf_exact_config`). Measured AICLK = 1350 MHz = reference clock, so scaled `achievable_ns` = `achievable_ns`:

  | shape | Tensix cores | R2 baseline | Refinement 3 | speedup | achievable_ns | vs achievable |
  |---|---:|---:|---:|---:|---:|---:|
  | 32×1024 | 1 | 8,610 | 7,635 | 1.13× | 9,149 | 0.83 ✓ |
  | 32×2304 | 1 | 14,894 | 12,355 | 1.21× | 17,003 | 0.73 ✓ |
  | 32×5120 | 1 | 28,770 | 23,003 | 1.25× | 75,825 | 0.30 ✓ |
  | 32×7168 | 1 | 40,258 | 30,866 | 1.30× | 14,894 | 2.07 ✗ (was 2.70) |
  | 640×6144 | 20 | 50,501 | 49,864 | 1.01× | 101,968 | 0.49 ✓ |
  | 640×7168 | 20 | 57,256 | 56,110 | 1.02× | 119,307 | 0.47 ✓ |
  | 7168×1024 | 120 | 80,850 | 76,776 | 1.05× | 89,392 | 0.86 ✓ |
  | 7168×2048 | 120 | 157,299 | 152,590 | 1.03× | 172,892 | 0.88 ✓ |
  | 8192×1024 | 120 | 88,550 | 85,535 | 1.04× | 96,744 | 0.88 ✓ |
  | 8192×2304 | 120 | 194,795 | 191,887 | 1.02× | 211,345 | 0.91 ✓ |
  | 8192×5120 | 120 | 435,122 | 430,506 | 1.01× | 738,307 | 0.58 ✓ |
  | 8192×7168 | 120 | 588,666 | 580,873 | 1.01× | 1,032,281 | 0.56 ✓ |

  - **Config-spanning guard set** (`test_rms_norm_perf_guard`): {32×7168, 640×6144, 8192×1024} × input `Layout` {TILE, ROW_MAJOR} × γ {TILE, ROW_MAJOR, none} × DEST {fp32, 16-bit}, all `TensorMemoryLayout::INTERLEAVED`, same-session before/after.
    - All 36 cases are ≥ 0.988× (worst: 8192×1024 TILE / RM-γ / 16-bit DEST, −1.2 %, inside noise).
    - Decode 1.13–1.30× on every path; ROW_MAJOR input at 640×6144 1.06–1.10×.
- Findings (remaining headroom):
  - **32×7168** is the only interleaved perf case above `achievable_ns`. It runs on one Tensix core: compute-bound at ~40 k cycles, with `square_accumulate` 84 cycles/tile (two full-tile unpacks), scale_by_gamma 37 and scale_by_inv_rms 47. That is ~20 % above the ~37-cycle-per-full-tile unpack floor of this 4-unpack schedule. Closing the gap needs occupancy (regime R3 W-split, Refinements 4 / 6), not per-core tuning.
  - **640-row** shapes are DRAM-bandwidth-bound in two phases: the whole grid reads (~500 GB/s), then the whole grid writes, because each Tensix core owns a single block.
  - **Prefill** is DRAM-bound at ~430 GB/s with NoC arbitration gradients: reads are served roughly by grid row, and the east-column writers finish last. Bank stagger and a NoC swap are both flat. The 120-core γ re-read (6 % of bytes with PREFIX) is the remaining avoidable traffic; G-MC belongs to Refinement 6.
- Golden test progress (`-k`-filtered slices):
  - `test_op_loose and INTERLEAVED`: 106/107. The one failure is the regime-refused `(1,1,32,32768)` (`NotImplementedError`, owned by Refinement 4, as before).
  - Cartesian `not loose and INTERLEAVED and BFLOAT8_B and rank and 4`: 354 run, 0 failures (3534 skipped as INVALID / EXCLUSION).
  - Unit dir:
    - `test_rms_norm.py` 150/150;
    - `test_rms_norm_gamma_tile.py` 70/70;
    - `test_rms_norm_numeric_config.py` 69/69;
    - `test_rms_norm_precision_matrix.py` 328/328;
    - `test_rms_norm_perf.py` 6/6;
    - `test_rms_norm_perf_knobs.py` 44/44.
- Issues encountered: none in the kernels. Two harness notes:
  1. `scripts/run_safe_pytest.sh` carries a run-harness wrapper diff (not committed).
  2. `-k` expressions containing `=` are rejected by pytest, so slices use plain substrings.
- Tests added:
  - `test_rms_norm_perf_knobs.py` (44 cases). It pins:
    - `GAMMA_DIRECT_READ` PREFIX / SEGMENTS / OFF × bf16 / fp32 / bf8b γ × input `Layout`, plus bf8b input;
    - `SCALE_BLOCK_TILES` 1 / 3 (ragged inside a chunk) / 8 × DEST width × {TILE γ, RM input + RM γ non-aligned, no γ};
    - `BANK_STAGGER` on;
    - the exact perf config at the 0.9995 gate.
  - `test_rms_norm_perf_baseline.py::test_rms_norm_perf_guard` (36-case config-spanning guard set for profiler runs).

## Refinement 4 — Cross-core Σx² combine (regime R3): WIDTH/BLOCK-sharded placement + interleaved W-split
- Date: 2026-10-21
- What was done: built the design's regime R3 (W-split + cross-core Σx² combine) as one topology serving three placements, and added `TensorMemoryLayout::WIDTH_SHARDED` / `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` (input and output; the output must carry the input's placement).
  - **Topology** (`select_plan()` + reader `root_block` / `contribute_block`): row-groups are consecutive (y, x)-ordered runs of G_w Tensix cores — exactly the partition `ttnn.Mcast(receivers, G_w, McastFixedSenderConfig(0), RowMajor)` builds — root = first core. Per block:
    1. every core runs `square_accumulate` on its W-slice and collapses it locally (`reduce<SUM, REDUCE_ROW>`, scaler 1.0, no post-op) into `cb_partial_col`;
    2. contributors unicast their partials into the root's `cb_gather` slots `r·G + index` (CB allocated on every core so the address matches), write-barrier, then bump `SEM_GATHER` on the root;
    3. the root copies its own partial in, waits `(block+1)·(G−1)` arrivals, sums with `reduce<SUM, REDUCE_ROW, AccumulateViaAdd, Skip>` + the finalize post-op (×1/W, +ε, rsqrt) into `cb_rms_root`, and multicasts it into every member's `cb_inv_rms` via `McastArgs` (receiver handshake; the root included through loopback);
    4. `scale_by_gamma` of chunk 0 runs before `cb_inv_rms` is waited (the stall shadow).
    A single-core group (G = 1) keeps the R1 local finalize.
  - **Sharded TILE (native, zero-copy)**: `cb_x_tiles` / `cb_output` are `ttnn.cb_descriptor_from_sharded_tensor` over the input / output shards. The reader pushes the shard's rows once (no NoC read), compute walks it strided (`TileAddressing::Strided`, row stride = shard width) and packs strided into the output shard; the writer only supplies γ. A ragged last shard's pad tiles are never read (the square stage runs over the slice's V valid tiles). **W-pad mask in compute** (lamp L5; the caller's shard is read-only): the core owning tile Wt−1 squares that tile separately and reduces it with `ReducePartialScaler::with_partial()` (the `[full, partial]` scaler pair from `calculate_and_prepare_partial_reduce_scalers`), accumulated (`Accumulate::at(cb_sumsq_acc, 1)`) onto the full tiles' collapse. Width shards spanning several grid rows (86 slices on 12×10) and ragged BLOCK grids are read off the shard spec.
  - **Sharded ROW_MAJOR**: shards are element-granular (8- / 24-element slices, 205-stick shards), so `tilize<C>` cannot read them in place. The reader re-lays the core's own shard sticks (local L1 NoC copies off the shard base address — no TensorAccessor, no DRAM) into the tilize staging at the C·32-element stride, zeroing the slice's pad bytes; the writer copies untilized sticks back into the output shard. RM γ slices start at any element: the DRAM read starts at the aligned-down byte (one extra staging page), with a 16-bit copy fallback in the γ expansion when the slice start is not 16 B-aligned.
  - **Interleaved TILE W-split**: each core reads its `[w0, w0+V)` slice of its tile-rows from DRAM once (pad tiles of the slice zeroed once, W-pad columns zeroed by the reader on the core holding tile Wt−1). Chosen when R1 refuses (row too wide: `(1,1,32,32768)`) or on occupancy. Layout knobs (lamp L2): `R3_ROWS_PER_GROUP = 2`, `R3_MAX_GROUP_CORES = 24`, `R3_MIN_SLICE_TILES = 4`; occupancy gate `R3_OCCUPANCY_GAIN = 2` × R1's cores and `R3_MIN_GROUP_CORES = 4` slices (a 2-way split loses: 32×256 4.02 → 4.23 µs). ROW_MAJOR interleaved stays R1.
  - Reused: every compute phase and helper (square_accumulate chain, reduce collapse, scaling chains, tilize/untilize), the R1 reader/writer loops (generalized from CT `Wt` to the runtime slice `[w0, w0+V)` — R1 is the `w0 = 0, V = Wt` case), the γ supply (offset by the slice), the footprint solve (new `Topology` terms). Added: `select_plan()` / `Plan` / `CoreWork` / `Topology`, 5 CBs (`cb_partial_col`, `cb_gather`, `cb_rms_root`, `cb_sumsq_last`, `cb_sumsq_acc`), 1 semaphore + the `ttnn.Mcast` channel, 8 shared CT args (prefix 14 → 22), the reader combine and RM-shard re-layout, the writer RM-shard write-back.
- Accuracy achieved: PCC ≥ 0.9995 (soft perf gate) on every interleaved and sharded `_perf_case` geometry at the exact config (bf16, 16-bit DEST, HiFi2, bf16 TILE γ); golden thresholds (bf16 PCC 0.995 / rel-RMS 0.04, fp32 0.999 / 0.02, bf8b 0.99 / 0.10) on every slice below, including the poisoned-padding WIDTH/BLOCK cases (poison 1000 in the W padding).
- Perf (device kernel ns, exact config, Blackhole p150 at 1350 MHz = reference clock; medians of 3 for the interleaved rows):

  | shape | placement | Tensix cores (R1 → R3) | Refinement 3 (R1) | Refinement 4 | speedup | achievable_ns | vs achievable |
  |---|---|---:|---:|---:|---:|---:|---:|
  | 32×1024 | interleaved | 1 → 8 | 7,635 | 4,570 | 1.67× | 9,149 | 0.50 ✓ |
  | 32×2304 | interleaved | 1 → 18 | 12,355 | 5,250 | 2.35× | 17,003 | 0.31 ✓ |
  | 32×5120 | interleaved | 1 → 23 | 23,003 | 6,340 | 3.63× | 75,825 | 0.08 ✓ |
  | 32×7168 | interleaved | 1 → 23 | 30,866 | 7,300 | 4.23× | 14,894 | 0.49 ✓ (was 2.07 ✗) |
  | 640×6144 | interleaved | 20 → 120 | 49,864 | 43,330 | 1.15× | 101,968 | 0.42 ✓ |
  | 640×7168 | interleaved | 20 → 120 | 56,110 | 49,000 | 1.15× | 119,307 | 0.41 ✓ |
  | 7168×1024 … 8192×7168 | interleaved | 120 (R1) | 76,776 … 580,873 | 77,900 … 582,500 | 1.00× (A/B same session: old 77.4/77.2 vs new 77.9/77.1 µs) | | ✓ unchanged |
  | 32×1024 | WIDTH [32,128] 8×1 | 8 | — | 3,550 | | 4,110 | 0.86 ✓ |
  | 32×2304 | WIDTH [32,256] 9×1 | 9 | — | 3,860 | | 4,617 | 0.84 ✓ |
  | 32×5120 | WIDTH [32,160] 8×4 | 32 | — | 4,820 | | 5,267 | 0.92 ✓ |
  | 32×7168 | WIDTH [32,256] 7×4 | 28 | — | 4,850 | | 5,481 | 0.88 ✓ |
  | 640×6144 | BLOCK [64,704] 9×10 (ragged) | 90 | — | 8,380 | | 19,039 | 0.44 ✓ |
  | 640×7168 | BLOCK [64,1024] 7×10 | 70 | — | 11,130 | | 21,669 | 0.51 ✓ |
  | 7168×1024 | BLOCK [736,128] 8×10 (ragged) | 80 | — | 40,930 | | 18,871 | 2.17 ✗ |
  | 7168×2048 | BLOCK [736,192] 11×10 (ragged) | 110 | — | 47,130 | | 25,947 | 1.82 ✗ |
  | 8192×1024 | BLOCK [1024,128] 8×8 | 64 | — | 56,430 | | 25,640 | 2.20 ✗ |

  - Interleaved slice-width sweep (lamp L2, `test_rms_norm_r3_perf.py`): decode wants ~20–30 slice cores (32×7168: S=4 / 56 cores 8.42 µs, S=8 / 28 cores 7.70, S=16 / 14 cores 7.68, S=32 / 7 cores 8.90); 640-row wants ~2 tile-rows per row-group (640×6144: G_w 48/24/12/6 → 70.3/46.0/43.5/44.9 µs). The shipped row-aware rule picks within ~2 % of the best measured point on every probe, and beats R1 on 128×4096 (19.2 → 9.9 µs), 96×6144 (26.9 → 12.8), 224×3072 (17.4 → 13.9), 1024×1024 (16.6 → 13.8), 64×512 (5.27 → 4.50).
  - Guard set (`test_rms_norm_perf_guard`, 36 configs): TILE 32×7168 30.9–44.6 → 7.2–7.9 µs on every γ / DEST variant; TILE 640×6144 → 42.9–44.5 µs; ROW_MAJOR paths (still R1) A/B-identical to Refinement 3 (32×7168 RM γ 41.9 / 41.9 old vs 41.7 / 41.8 new; 640×6144 TILE γ 56.3 / 54.9 vs 55.3 / 55.7).
- Findings (remaining headroom):
  - **Tall BLOCK-sharded geometries** (7168/8192 × 1024/2048, 23–32 tile-rows per core) are 1.8–2.2× over `achievable_ns`. The combine is a flat root reduce: the root sums G partials and runs the full-tile finalize (mul, add, rsqrt over 32×32 lanes) for all R rows of the block while the other G−1 cores wait. Next levers (Refinement 6): reduce-scatter the combine over the group's cores (catalog `tensix_all_reduce`: each member finalizes R/G rows), and column-0 SFPU scoping of the finalize (lamp L7, catalog `sfpu_tile_scope`).
  - **ROW_MAJOR interleaved decode** stays on one Tensix core (32×7168: 41.9 µs): interleaved RM R3 (DRAM stick-slice reads) is not built — no ROW_MAJOR perf case pins it.
- Golden test progress (filtered in-process slices; the harness's full run is the record):
  - WIDTH/BLOCK cartesian: TILE 1,740 pass / 0 fail (420 expected xfail: `{float32, fp32_dest_acc_en=False}`, `{bfloat8_b, w_non_aligned}`); ROW_MAJOR 630 pass / 0 fail (210 expected xfail).
  - WIDTH/BLOCK loose (perf geometries, resilience, poisoned padding): 192 pass / 0 fail / 3 `INFEASIBLE_L1` skips (harness shard creation).
  - INTERLEAVED TILE cartesian + loose: 934 pass / 0 fail (210 expected xfail), including `(1,1,32,32768)` (was `supported_fail`), `(1,1,32,16384)`, `(1,1,64,12288)`; INTERLEAVED ROW_MAJOR cartesian 630 pass / 0 fail; interleaved loose 107/107.
  - Unit dir: `test_rms_norm.py`, `test_rms_norm_gamma_tile.py`, `test_rms_norm_numeric_config.py`, `test_rms_norm_perf_knobs.py`, `test_rms_norm_r3.py`: 394 passed, 6 skipped.
- Issues encountered: none in the kernels (every R3 path passed its first device run). Notes: a non-template `if constexpr` cannot discard `McastArgs::sender()` (its inactive specialization static_asserts), so the role dispatch lives in the `run_blocks<COMBINE>` template; the shared device lock made one golden slice time out while waiting (re-run clean).
- Tests added:
  - `test_rms_norm_r3.py` (62 cases): regime-selection pins (R3 decode, R3 when R1 refuses, R1 for many rows, R1 for a 2-way split, R1 when the occupancy branch is off), ragged interleaved slices, both γ `Layout`s in fp32, no γ; WIDTH/BLOCK TILE shards (multi-row width groups, ragged grids, single-core groups), poisoned W padding, the pinned perf geometries, fp32 / bf8b shards; ROW_MAJOR WIDTH/BLOCK shards × bf16 / fp32 × γ / no γ.
  - `test_rms_norm_r3_perf.py` (profiler harness): R1 vs R3 and the group-size cap on the interleaved shapes, plus the sharded perf geometries.
  - `test_rms_norm_r3_ab_debug.py` (profiler A/B against an exported snapshot of the previous refinement; skipped unless `RMS_AB_OLD_DIR` is set).

## Refinement 5 — HEIGHT-sharded placement (local reduction)
- Date: 2026-10-21
- What was done:
  - `HEIGHT_SHARDED` added to `SUPPORTED["memory_layout"]` (input and output). It is regime R1-HS: the existing sharded path (Refinement 4) with one W-slice per Tensix core, so G = 1, `Topology.combine = False`, and there are no gather / root CBs, semaphore or multicast. `_sharded_work` gains a HEIGHT branch: shard i sits on the i-th core of `ttnn.corerange_to_cores(grid, n_shards, row_wise = orientation == ROW_MAJOR)`, the buffer's own shard order. A HEIGHT core needs no row-group, so COL_MAJOR orientation is accepted too. A ragged last shard carries fewer valid tile-rows (TILE) or sticks (ROW_MAJOR); grid cores past the last shard get no work.
  - Native placement, no `TensorAccessor` on the shard. TILE: `cb_x_tiles` / `cb_output` are zero-copy CBs (`ttnn.cb_descriptor_from_sharded_tensor`), and one block is the whole shard unless L1 forces a split. ROW_MAJOR: the R3 local-L1 stick re-layout. The W-pad mask is the compute-side partial reduce scaler from Refinement 4.
  - New L1 fallback (`Geometry.gamma_stream_direct`). Example: (1,1,160,11008) TILE with TILE γ, HEIGHT-sharded to 1 × 344 tiles per core. The input and output shards are 704 KB each and leave 52 KB of L1. The S-page TILE-γ staging row alone was 44 KB, so `choose_blocking` refused. It now retries without the staging row. The writer then reads each streaming γ chunk's row-0 prefix straight from DRAM into `cb_gamma_tiles` (the Refinement 3 direct read, factored into `read_gamma_chunk_direct` and shared with the resident path). It is taken only when the staging-row search refuses, so no shape that fit before changes.
  - Reused: the whole R3 sharded pipeline (reader / compute / writer untouched apart from the γ fallback), the footprint solve, the direct γ read. Added: the HEIGHT branch of `_sharded_work`, the `R1-HS` plan label, the `gamma_stream_direct` fallback (host + writer).
- Perf (device kernel ns, `_perf_case` config: bf16, HiFi2, 16-bit DEST, bf16 TILE γ; auto HEIGHT shard spec vs DRAM interleaved; one run each, no HEIGHT `_perf_case` exists):

  | shape | interleaved (Tensix cores) | HEIGHT_SHARDED (Tensix cores) |
  |---|---:|---:|
  | 1024×1024 | 13,650 (119) | 6,394 (32) |
  | 640×6144 | 43,337 (120) | 19,124 (20) |
  | 7168×1024 | 76,335 (120) | 14,298 (112) |
  | 8192×1024 | 85,800 (120) | 14,413 (86) |

  For comparison, the pinned BLOCK geometry for 7168×1024 is 40.9 µs (`achievable_ns` 18.9 µs). A height-sharded tall shape has no combine and no DRAM traffic for x or the output.
- Accuracy achieved: PCC ≥ 0.999 (unit tests, bf16 / fp32) and ≥ 0.9995 on the perf shapes and the W = 11008 fallback. Golden thresholds (bf16 PCC 0.995 / rel-RMS 0.04, fp32 0.999 / 0.02, bf8b 0.99 / 0.10) hold on every HEIGHT cell, including poisoned W padding.
- Golden test progress (filtered slices; the harness's full run is the record):
  - HEIGHT cartesian: 1,167 pass / 0 fail. That is bf16 TILE 420, bf16 ROW_MAJOR 210, fp32 TILE 192, fp32 ROW_MAJOR 105, bf8b 240. 315 are expected xfail (`{float32, fp32_dest_acc_en=False}`); the rest are INVALID skips.
  - HEIGHT loose: 93/93. The first run failed 2 cells, (1,1,160,11008) and (1,224,11008) TILE γ, on L1; the γ fallback fixed both.
  - Regression: non-HEIGHT wide loose cells (11008 / 32768 / 5119 / 3072): 31/31. Unit `test_rms_norm.py`, `test_rms_norm_numeric_config.py`, `test_rms_norm_gamma_tile.py`, `test_rms_norm_r3.py`: 350 passed, 6 skipped.
- Issues encountered: the 2 L1 refusals above (fixed). Every other HEIGHT path passed its first device run.
- Tests added: `test_rms_norm_height_sharded.py` (39 cases):
  - TILE shapes with and without γ: ragged H, tall rank-2 (99991, 64), multi-batch, a single wide shard;
  - poisoned W padding;
  - ROW_MAJOR × bf16 / fp32;
  - fp32 / bf8b;
  - ROW_MAJOR vs COL_MAJOR orientation on a 4×2 grid with 7 shards;
  - the W = 11008 γ-fallback shapes;
  - an interleaved-vs-HEIGHT perf record (`--profile`).

## Refinement 6 — Speed up the decode and sharded perf loose cases
- Date: 2026-10-21
- What was done:
  - **Classified first.** Device zones on 8192×1024 BLOCK [1024,128] 8×8 (64 Tensix cores) showed the op was compute-bound on the regime R3 root.
    - The root's combine took 51 k of ~76 k compute cycles while the other 7 members waited.
    - Ablating only the finalize post-op cut the combine to 12.8 k. So the full-tile finalize (mul_unary, add_unary, rsqrt: 3 × 32 SFPU vectors) cost ~1.8 k cycles per tile-row.
  - **Lever 1 — column-0 fused finalize** (`FINALIZE_COL0`, lamp L7, catalog `sfpu_tile_scope` c_skip).
    - `rsqrt(Σx²·(1/W) + ε)` runs as one SFPU pass over the 8 even-parity vectors of faces 0/2 (`VectorMode::C`, `dst_reg += 2`), instead of 96 vectors.
    - The product stays in an fp32 SFPU register. Every finalized tile is a REDUCE_ROW result and every consumer reads column 0 (`BroadcastDim::Col`).
    - Raw sfpi body, justified at the head of the compute kernel (no compute API exposes a column-scoped or fused SFPU op). Applies to R1, R1-HS and R3.
  - **Lever 2 — reduce-scatter combine** (`R3_SCATTER_ROWS_PER_OWNER`, lamp L2, catalog `tensix_all_reduce`). Each block's tile-rows are split over `owners = min(G, rows // knob)` members (trimmed to the block size).
    - Every member sends each owner its partials of that owner's rows as ONE write. The gather is now contributor-major: slot `j·q + r`.
    - The owner sums its rows with `reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>` over (G × owned) at row pitch q, with `BulkWaitBulkPop` on the uniform G·q quantum. It then finalizes and unicasts the finalized run into every member's `cb_inv_rms`; `SEM_INV` counts arrivals.
    - owners = 1 (decode, or the knob at 0) is the Refinement 4 root + multicast path, now also with the one-write gather.
    - `cb_gather` shrinks from G·R to G·⌈R/owners⌉ pages, so tall shards run one block (8192×1024: 21 + 11 → 32).
  - **γ(0) hoist**: with owners > 1, chunk 0's γ pass runs before the combine, filling the gather wait. It is gated off for a single root, where it delayed the multicast every member waits on (640-row interleaved measured ~1 % slower with it).
  - **Measured and dropped**: the all-gather as concurrent multicasts (rotating-sender `ttnn.Mcast`, Counter signal, no handshake). On 8192×1024 it took 26.4 µs vs 22.5 µs unicast, and the inv_rms wait went from 7.7 k to 7.9–19.3 k cycles, erratic across cores.
  - **Not attempted**: G-MC γ multicast (a second T3 restructure; γ supply finishes in ~1.1 µs on these shapes, off the critical path) and `trid_double_issue` (zero-copy shards issue no reads).
  - **Reused**: every compute phase and helper, the R4 gather/semaphore/multicast machinery (the owners = 1 path), the footprint solve, the R3 test harness (`_run`).
  - **Added**:
    - knobs `FINALIZE_COL0` and `R3_SCATTER_ROWS_PER_OWNER`;
    - `Topology.owners`, `scatter_rows()`, `gather_pages()`, `combine_bytes()`, and `_max_block_rows()` (monotone footprint search, identical to the old closed form without a combine);
    - reader `scatter_block`, compute-only CT arg 23/24, reader CT args after the accessor, packed group-member RT args, `SEM_INV`.
- Accuracy achieved:
  - Soft PCC gate 0.9995 holds on every interleaved and sharded `_perf_case` geometry.
  - Golden thresholds (bf16 PCC 0.995 / rel-RMS 0.04; fp32 0.999 / 0.02; bf8b 0.99 / 0.10) hold on every slice below.
  - The fused finalize keeps x·(1/W)+ε in fp32 (the stock path stored it to DEST between passes).
  - Precision matrix 328/328.
- Perf (device kernel ns, exact config: bf16, 16-bit DEST, HiFi2, bf16 TILE γ; Blackhole p150 at 1350 MHz = reference clock; same-session A/B vs the Refinement 5 kernels; medians of 3 where the delta was within the noise band):

  | shape | placement | Refinement 5 | Refinement 6 | speedup | achievable_ns | vs achievable |
  |---|---|---:|---:|---:|---:|---:|
  | 32×1024 | interleaved | 4,565 | 3,747 | 1.22× | 9,149 | 0.41 ✓ |
  | 32×2304 | interleaved | 5,241 | 4,436 | 1.18× | 17,003 | 0.26 ✓ |
  | 32×5120 | interleaved | 6,316 | 5,544 | 1.14× | 75,825 | 0.07 ✓ |
  | 32×7168 | interleaved | 7,322 | 6,466 | 1.13× | 14,894 | 0.43 ✓ |
  | 640×6144 / 640×7168 | interleaved | 43.2 k / 48.1 k | 43.2 k / 48.6 k | parity (±1 %, DRAM-bound) | | ✓ |
  | 7168×1024 … 8192×7168 | interleaved (R1) | 77.5 k … 586.8 k | 78.5 k … 590.7 k | 0.99–1.01× (noise) | | ✓ |
  | 32×1024 | WIDTH [32,128] 8×1 | 3,516 | 2,707 | 1.30× | 4,110 | 0.66 ✓ |
  | 32×2304 | WIDTH [32,256] 9×1 | 3,858 | 3,044 | 1.27× | 4,617 | 0.66 ✓ |
  | 32×5120 | WIDTH [32,160] 8×4 | 4,830 | 3,987 | 1.21× | 5,267 | 0.76 ✓ |
  | 32×7168 | WIDTH [32,256] 7×4 | 4,844 | 4,048 | 1.20× | 5,481 | 0.74 ✓ |
  | 640×6144 | BLOCK [64,704] 9×10 | 8,470 | 8,420 | parity (median of 3) | 19,039 | 0.44 ✓ |
  | 640×7168 | BLOCK [64,1024] 7×10 | 11,290 | 11,310 | parity (median of 3) | 21,669 | 0.52 ✓ |
  | 7168×1024 | BLOCK [736,128] 8×10 | 40,931 | 16,739 | 2.45× | 18,871 | 0.89 ✓ (was 2.17 ✗) |
  | 7168×2048 | BLOCK [736,192] 11×10 | 47,096 | 22,307 | 2.11× | 25,947 | 0.86 ✓ (was 1.82 ✗) |
  | 8192×1024 | BLOCK [1024,128] 8×8 | 56,352 | 22,647 | 2.49× | 25,640 | 0.88 ✓ (was 2.20 ✗) |

  - Lever split (8192×1024 / 7168×1024 / 7168×2048 BLOCK):
    - column-0 finalize alone: 56.3 / 40.9 / 47.1 → 31.0 / 22.7 / 28.8 µs;
    - plus reduce-scatter: → 22.5 / 16.7 / 22.2 µs;
    - reduce-scatter with the full-tile finalize: 25.8 / 18.9 / 24.3 µs (both levers carry weight).
  - Guard set (`test_rms_norm_perf_guard`, 36 configs, all `TensorMemoryLayout::INTERLEAVED`): 0.98–1.15× (TILE 32×7168 1.12–1.15×; every other case within ±2 %, noise). HEIGHT_SHARDED perf record: parity (0.99–1.01×).
- Findings (remaining headroom):
  - Every decode and sharded `_perf_case` is now under `achievable_ns`.
  - What binds on the tall BLOCK shapes (8192×1024, ~30 k cycles per Tensix core):
    - square+collapse 7.1 k;
    - γ pass 5 k;
    - the two NoC exchanges: gather wait 2.9 k after the γ pass, plus inv_rms wait ~7.7 k including the owners' combine;
    - inv pass ~7.5 k.
  - The exchanges are all-to-all inside one grid row on the NoC0 torus (~64 KB per Tensix core each way). One write per owner measured the same as one per tile, so they are link-bandwidth-bound, not issue-bound.
  - Next to try:
    - halve the bytes: column-0 partials need only faces 0/2, or a 32×16 half-tile intermediate format;
    - route west-bound flows on NoC1.
- Golden test progress (`-k`-filtered slices; the harness's full run is the record):
  - loose, all placements: 392/392 (3 harness `INFEASIBLE_L1` skips). The first run failed 4 tall WIDTH/BLOCK cells; fixed, see Issues.
  - WIDTH/BLOCK cartesian, bf16 input, both input `Layout`s, all γ modes: 1,260 pass / 0 fail;
  - bf8b WIDTH/BLOCK TILE with TILE γ: 240 pass / 0 fail.
  - Unit dir: `test_rms_norm.py`, `test_rms_norm_r3.py`, `test_rms_norm_height_sharded.py`, `test_rms_norm_gamma_tile.py`, `test_rms_norm_numeric_config.py`, `test_rms_norm_perf_knobs.py`, `test_rms_norm_precision_matrix.py`, `test_rms_norm_r6.py`: 780 passed, 110 skipped.
- Issues encountered:
  1. The fused finalize body first ran out of SFPU LREGs (the two scalars were held live across the loop). They are now reloaded per vector.
  2. **Ragged-owner ring drift.**
     - Symptom: the golden tall shards (3232×96 and 4064×160 WIDTH, 99991×64 BLOCK) had a few tile-rows with garbage inv_rms.
     - DEVICE_PRINT showed it: the last owner of a ragged split pushes fewer than q rows per full block into its q-page `cb_rms_root`. Its 7-page pops in an 8-page ring never wrapped (CB pointers wrap only on exact equality with the ring end), so the read pointer reached slot 14 and the one-run all-gather write read past the CB.
     - Fix: the reader splits a run at the ring end and pops `cb_rms_root` one page at a time.
- Tests added:
  - `test_rms_norm_r6.py` (19 cases):
    - reduce-scatter topology pins (ragged ownership, fewer owners than members, one tall WIDTH row-group);
    - several blocks with a ragged last block;
    - masked TILE W-pad with poisoned padding;
    - ROW_MAJOR shards ± γ;
    - fp32 DEST;
    - interleaved R3 owner trim;
    - the knobs (root topology, full-tile finalize);
    - the four golden tall-shard geometries under both topologies.
  - `test_rms_norm_r3_perf.py`: a combine-knob axis on the sharded perf cases (scatter / root / full finalize) and a knob dict on the interleaved ones (+ `fullfin`).

## Perf 1
- Date: 2026-10-21
- Scope: perf tournament, round 1 of 2. SUPPORTED is unchanged. Precision contract fixed (bf16, HiFi2, `fp32_dest_acc_en`, `math_approx_mode` as given). Blackhole p150b, AICLK 1350 MHz = the `achievable_ns` reference clock. Device kernel ns throughout.
- **Focus selection.** No `LOOSE_CASES` entry carries an `attention:` flag, so focus shapes were free-selected by measured headroom, at their exact `_perf_case` config (bf16, HiFi2, `fp32_dest_acc_en=False`, bf16 `Layout::TILE` γ):
  - (1,1,8192,5120) `TensorMemoryLayout::INTERLEAVED` (prefill, the largest absolute time);
  - (1,1,8192,1024) `BLOCK_SHARDED` [1024,128] on 8×8 Tensix cores (the tall-BLOCK family at 0.86–0.89 of `achievable_ns`).
- **Instrumentation (permanent, commit `75bdffbd8d4`).**
  - Restored `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`, opt-in `KERNEL_PERF_ZONES`, free when off).
  - Stage zones in all three kernels, with every wait split from its payload: `rd_reserve`/`rd_issue`/`rd_barrier`; the R3 combine's wait / send / gather / root / all-gather / inv; `cmp_wait_*` versus each pass; `wr_wait_out` versus `wr_issue`.
  - Measurement-only host hooks: `RMS_NORM_PERF_ZONES=1` and `RMS_NORM_ABLATE=reads,writes,gamma,combine,compute` (payload stubbed, CB / semaphore sync kept).
  - Bench `test_rms_norm_perf_bench.py`, plus `_perf_report.py` (per-zone max/mean per Tensix core, marker-cap and coverage checks).
  - The new paths below carry their own zones: `wr_wait_root` / `wr_allgather` on the writer-side all-gather, per-wave `rd_send` / `rd_allgather` / `cmp_*`, and the γ multicast inside `wr_gamma`.

### Measured breakdown (before)
- **8192×5120 interleaved: 431,679 ns, DM-bound.** Ablations, each alone, CB sync kept:

  | stubbed | ns |
  |---|---:|
  | compute | 431,356 (flat) |
  | x reads | 310,441 |
  | output writes | 250,483 |
  | **γ reads** | **389,241 (−42 µs)** |
  | reads + writes | 81,704 |
  | reads + writes + γ | 45,787 |
  | everything | 8,100 |

  - With no x traffic at all, γ still costs about 36 µs: 120 Tensix cores × 160 tiles × 544 B, every core walking the same γ tiles in lockstep.
  - Zones: `wr_gamma` per core max 174 µs / mean 60 µs. The writer finishes the whole resident γ window before it drains any output, so on the east-column cores γ sits serially ahead of 250–330 µs of output writes. Per-core end times spread from 78 to 431 µs.
  - Roofline: 168 MB moved at 390 GB/s against 512 GB/s DRAM peak. The `noc_estimate` CLI target is not built in this checkout, so the gate is first-principles.
- **8192×1024 BLOCK: 22,580 ns, compute and exchange balanced.**
  - Ablations: compute 16,101; combine data writes 15,327 (−7.3 µs); both 6,281 (−16.3 µs). Each hides the other, so neither looked expensive on its own. Reads / writes / γ are flat (zero-copy shards).
  - Zones on TRISC0: square 3.7 → collapse 1.4 → γ0 3.8 (shadows the send) → combine 3.3 → wait_inv 5.9 → inv 3.8 µs.
  - Zones on the reader: rd_send 4.5 µs (64 KB per Tensix core), rd_allgather 4.2 µs (64 KB per owner).
  - Link bound: about 2.3 µs per exchange phase at 2 KB tiles, against 4.2–4.5 µs measured.
- **Siblings:**
  - 8192×1024 interleaved, 84,987 ns: reads −19.6 µs, writes −32.7 µs, γ −0.9 µs, compute flat.
  - 640×6144 interleaved, 42,884 ns: DM-bound, γ and compute flat.
  - 7168×2048 BLOCK, 22,208 ns: compute −6.6, combine −7.0, both −17.1 µs.
  - 32×7168 WIDTH, 3,986 ns: compute −0.7, combine −0.7 µs, floor 2.55 µs.
- **Ranked bottlenecks:**
  1. prefill γ supply (redundant per-core DRAM reads, serialized ahead of the output drain);
  2. the R3 cross-core exchange (link-bound, serialized with compute);
  3. the tall-BLOCK compute passes.

### Portfolio (3 experiments, the cap) and verdicts — all three WIN
1. **`gamma_mcast`** (standing reuse-shared-broadcast idea): every set of Tensix cores that needs the same γ window reads it once — the group's first core (writer RISC-V) — and multicasts it into every member's landing CB through `ttnn.Mcast` + `McastArgs` (NoC1, Counter data-ready, no handshake). Focus 8192×5120: 432,303 → 388,696 ns (1.112×), at the γ-ablation floor (391,744). Output bit-exact.

   | variant | ns |
   |---|---:|
   | one group per shared window (**graduated**) | 388,696 |
   | per grid row | 390,513 |
   | per grid column | 390,402 |
   | + receiver handshake | 394,322 |
   | prefix-only multicast per tile | 423,025 (and slower than baseline on 8192×1024) |

2. **`combine_bytes`**: R3 combine tiles (column-0 partials and finalized inv_rms) cross the NoC as faces 0 and 2 only, half the bytes, as one-packet stateful writes, with a rotated destination order. Focus 8192×1024 BLOCK one-shot combine: 22,723 → 18,181 ns.

   | option | ns |
   |---|---:|
   | faces 0/2 | 18,615 |
   | rotation alone | 21,333 |
   | faces 0/2 + rotation (**graduated**) | 18,181 |
   | concurrent multicast all-gather | 48,168 (regression) |

   - Column-0 compaction was not built: the remaining all-gather ceiling was 3.2 µs, while the re-expansion on every member would cost an estimated 1.5–3 µs.
   - Faces 1/3 never matter: combine = elementwise `add_tiles`, finalize = lane-wise, and the srcB unpack for the `BroadcastDim::Col` multiply loads faces 0 and 2 only. Proven bit-exact with NaN-poisoned faces 1/3; a face-0-only negative control fails, as it should.
   - Found a latent race: the reader exited with its last semaphore increments still in flight, which hangs under `--dev`. Fixed with `noc_async_atomic_barrier()` at kernel end.

3. **`combine_pipeline`**: the TILE-sharded reduce-scatter combine (owners > 1) runs in waves.
   - Wave k's partials travel while wave k+1 is squared, and wave k is scaled as soon as its inv_rms lands.
   - The reader is a non-blocking poll loop with one semaphore pair per wave.
   - The all-gather runs on the writer RISC-V (its own NoC) when γ is resident or absent. The writer hands a token to the reader, which issues the NoC atomics.
   - Focus 8192×1024 BLOCK: 22,673 → 16,716 ns. TRISC0 now has no gaps: inv_rms wait 5.9 µs → 30 ns.

   | variant | ns |
   |---|---:|
   | 1 wave (code-path control) | 22,846 |
   | 2 waves | 19,235 |
   | 4 waves | 16,814 |
   | auto (**graduated**: 4 waves, halved below 8 tiles of square work per wave; γ0 half-way; writer-side all-gather) | 16,738 |

   Forcing 8 waves on 99991×64 measured +23 %, hence the auto rule.

### Aggregation and graduation
- **The three are non-conflicting and fastest together**, measured cumulatively in one session (ns):

  | shape | pipeline + γ multicast | + faces/rotation |
  |---|---:|---:|
  | 8192×1024 BLOCK | 16,716 | 15,733 |
  | 7168×2048 BLOCK | 18,702 | 15,390 |
  | 7168×1024 BLOCK | 13,085 | 12,039 |
  | 32×7168 WIDTH | 4,043 | 3,634 |

  All three graduated as one combination.
- **Faces 0/2 + rotation.** The op's only combine transport now, at every R3 unicast site: reduce-scatter partials, root-path partials, and the reader and writer all-gathers. The whole-tile writes are deleted.
  - Inexpressible: the owners == 1 inv_rms multicast. `McastArgs::send()` handshakes on every call, so it keeps one whole-tile send per block.
- **Atomic barrier at reader kernel end.** Unconditional.
- **Wave pipeline.** The path for every TILE-sharded (`WIDTH_SHARDED` / `BLOCK_SHARDED`) reduce-scatter combine. Knobs are module constants like the op's other knobs: `R3_WAVES`, `R3_WAVE_MIN_TILES`, `R3_WAVE_GAMMA0_POS`, `R3_WAVE_INV_GROUP`, `R3_ALLGATHER_ON_WRITER`. One wave needs exactly the one-shot CB pages, so the L1 delta is 0 B plus a 32 B token page per wave.
  - Inexpressible, so they keep the one-shot `scatter_block`: interleaved R3 with owners > 1 and `ROW_MAJOR` shards (their compute takes the non-sharded path), and owners == 1 (root + multicast).
  - Streaming γ keeps the all-gather in the reader's poll loop (correctness: a γ push must never wait behind it).
- **γ multicast.** On wherever a shared window can be grouped into equal consecutive `ttnn.Mcast` groups: R1 and HEIGHT use the whole grid as one group; R3 interleaved and BLOCK use one group per grid column.
  - Inexpressible, so they keep per-core reads: no γ; every window distinct (WIDTH, decode); strided sharing (R3 interleaved with G_w ≠ grid width); the streaming-direct L1 fallback.
  - **Carve-out 1 (measured regression)**: `GAMMA_MCAST_SHARDED_DIRECT_MAX_CORES = 32`. A sharded input on ≤ 32 Tensix cores keeps per-core reads. With no x DRAM traffic the per-core reads are uncontended, while the multicast moves whole γ pages (about 4× the row-0 prefix bytes) through one injector. HEIGHT, 1 tile-row per Tensix core, per-core vs multicast ns:

    | Tensix cores | W | per-core | multicast |
    |---:|---:|---:|---:|
    | 8 | 512 | 2,934 | 3,802 |
    | 20 | 1024 | 5,184 | 6,642 |
    | 20 | 3072 | 9,677 | 14,712 |
    | 20 | 6144 | 18,035 | 27,092 |
    | 32 | 3072 | 13,650 | 15,374 |
    | 32 | 6144 | 24,044 | 28,569 |
    | 48 | 1024 | 7,474 | 6,706 (win) |
    | 48 | 3072 | 17,216 | 14,978 (win) |
    | 64 | 6144 | 38,575 | 26,953 (win) |

    Interleaved inputs (x traffic contends with the γ reads) win even at 20 cores (640×6144 ROW_MAJOR: 0.97×).
  - **Carve-out 2 (measured regression)**: `GAMMA_MCAST_RM_SHARD_DIRECT_MAX_BYTES = 1024`. A `ROW_MAJOR`-sharded input with a ROW_MAJOR γ window ≤ 1 KB keeps the per-core read. That read is one small DRAM request per core, so the multicast hop is pure latency.

    | HEIGHT shape (γ window) | per-core | multicast |
    |---|---:|---:|
    | 256×512 (1 KB) | 4,798 | 5,396 |
    | 512×512 (1 KB) | 5,164 | 5,441 |
    | 1024×512 (1 KB) | 5,379 | 5,547 |
    | 2048×1024 (2 KB) | 9,684 | 7,787 (win) |
    | 256×2048 (4 KB) | 14,781 | 11,448 (win) |

  - Both carve-outs are written as `if (cannot profit) { per-core reads } else { multicast }`. The 33–47-core range is untested and takes the multicast.
- **Writer-side atomics on BRISC** trip the `--dev` assert `noc_assert_targ_addr_mid_clear`; the firmware leaves a PCIe routing value in that command buffer's MID register. That is why the writer-side all-gather hands its arrival increments to the reader (NCRISC).

### Whole op, before → after (same-session A/B against the pre-graduation snapshot; medians of 3; `test_rms_norm_perf_guard_ab.py`)
| cell | before | after | ratio | `achievable_ns` |
|---|---:|---:|---:|---:|
| 8192×5120 interleaved (focus) | 431,567 | 391,759 | 0.908 | 738,307 |
| 8192×7168 interleaved | 587,020 | 546,521 | 0.931 | 1,032,281 |
| 7168×2048 interleaved | 153,806 | 139,742 | 0.909 | 172,892 |
| 8192×1024 interleaved | 85,172 | 84,090 | 0.987 | 96,744 |
| 640×6144 interleaved | 42,936 | 43,828 | 1.021 → re-measured medians of 5: 43,496 vs 43,576 (noise) | 101,968 |
| 32×7168 interleaved | 6,513 | 6,433 | 0.988 | 14,894 |
| 8192×1024 BLOCK [1024,128] 8×8 (focus) | 22,728 | 15,659 | **0.689** | 25,640 |
| 7168×2048 BLOCK [736,192] 11×10 | 22,188 | 15,384 | 0.693 | 25,947 |
| 7168×1024 BLOCK [736,128] 8×10 | 16,796 | 12,053 | 0.718 | 18,871 |
| 640×7168 BLOCK [64,1024] 7×10 | 11,274 | 9,401 | 0.834 | 21,669 |
| 32×7168 WIDTH [32,256] 7×4 | 4,033 | 3,612 | 0.896 | 5,481 |
| 32×1024 WIDTH [32,128] 8×1 | 2,685 | 2,618 | 0.975 | 4,110 |

- **Config-spanning guard set (40 cells)**: one representative per kernel path (R1 / R1-HS / R3 interleaved / WIDTH / BLOCK) × input `Layout` (TILE / ROW_MAJOR) × γ (TILE / ROW_MAJOR / none) × DEST (16-bit / fp32) × bf8b. Every cell is 0.551–1.008×, apart from 640×6144 interleaved (noise, above).
  - Highlights: BLOCK 8192×1024 fp32 DEST 0.607, no γ 0.551; HEIGHT 2048×6144 0.720, 8192×1024 0.892; WIDTH 3232×96 0.849; 99991×64 BLOCK 0.917; 8192×5120 ROW_MAJOR γ 0.952, fp32 DEST 0.910; 8192×1024 bf8b 0.968.
  - Flat: every no-γ interleaved cell, ROW_MAJOR 32×7168, and the carve-out cells (0.99–1.01).
- **Golden**: green throughout.
  - `test_golden.py` 5,429 passed / 0 failed (1,365 expected xfail), run in 12 `--splits` groups on the final code; `test_regression.py` 15/15; `test_translated.py` 104 passed + 2 expected xfail.
  - Unit dir: 452 passed, plus the new `test_rms_norm_perf1.py` 44/44. Bit-exact pins: γ multicast on/off including both carve-outs and the staging path; waves 1/2/4 × writer/reader all-gather against the one-shot combine on 8 geometries (fp32 DEST, no γ, masked W pad, ragged owners, 99991×64, 2 W-chunks, streaming γ).
  - `--dev` (watcher, NoC sanitizer, asserts): `test_rms_norm_r3.py`, `test_rms_norm_r6.py`, `test_rms_norm_height_sharded.py`, `test_rms_norm_perf1.py` and the bench, all pass.
- **What binds now (round 2 starting point)**:
  - 8192×1024 BLOCK is compute-bound: combine ablation flat, compute ablation 15.6 → 8.7 µs; per wave, square 3.9, γ 4.0, inv 4.6 µs.
  - Prefill is pure x DRAM traffic: the γ ablation is now flat. The east-column output writers still finish last (`wr_issue` max 347 µs / mean 75 µs on 8192×5120).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (kernel_lib dataflow transport: face-subset tile write) | capability | No helper writes a face-strided subset of a tile (faces 0/2 of a column-0 collapsed tile, as 2 × 512 B packets per tile to one destination). Callers moving REDUCE_ROW / REDUCE_COL results must pay full tile bytes. The combine transport was already raw NoC. | n/a (whole-tile raw writes: 22,723) | 18,615 (faces 0/2) / 18,181 (+ rotation) | `kernels/rms_norm_combine.hpp:28` |
| `Semaphore<>` / kernel_lib dataflow wait helpers | capability | Only blocking waits exist (`Semaphore::wait_min`, `wait_front`). A poll loop over four event sources needs a non-blocking `try_wait` / value read on `Semaphore<>`. The site uses a raw `get_semaphore()` pointer read, plus `CircularBuffer::pages_available_at_front`. | 22,676 (one-shot schedule, blocking waits) | 16,738 (wave poll loop) | `kernels/rms_norm_reader.cpp:445` |

Helper friction that did NOT end in a bypass:
- `McastArgs`: an absent channel's `sender()` / `receiver()` still `static_assert` inside `if constexpr` in a non-template `kernel_main`, so the writer wraps each multicast body in a generic lambda. That is an ergonomics gap: the reader needs a `run_blocks` template for the same reason.
- `SenderPipe::send()` sends one region and one signal per call, so it cannot batch N strided regions under one signal. This only blocked the losing prefix-only variant.
- `ttnn.Mcast` builds only equal, consecutive RowMajor / ColumnMajor groups. That is the source of the strided-sharing fallback.

- **Tests added**:
  - `test_rms_norm_perf1.py` (44 bit-exact / PCC pins);
  - `test_rms_norm_perf_guard_ab.py` (same-session A/B guard set against a snapshot in `RMS_AB_OLD_DIR`; `RMS_GUARD_IMPLS=gmoff,...` for the in-session γ-multicast A/B);
  - `test_rms_norm_perf_bench.py` (`prod` / `gmoff` / `zones` / ablation modes; HEIGHT rows-per-core × W sweep);
  - `_ab_report.py`.
- **Experiment artifacts**: `perf_experiments/gamma_mcast/`, `perf_experiments/combine_bytes/`, `perf_experiments/combine_pipeline/` (benches, variant kernels, `--dev` triage captures).

## Perf 2
- Date: 2026-10-21
- Scope: perf tournament, round 2 of 2. SUPPORTED is unchanged. The precision contract is fixed (bf16, HiFi2, `fp32_dest_acc_en`, `math_approx_mode` as given); no option traded it. Blackhole p150b, AICLK 1350 MHz (= the `achievable_ns` reference clock). All times are device kernel ns.
- **Focus selection.** No `LOOSE_CASES` entry carries an `attention:` flag, so the focus shapes were picked by measured headroom, at their exact `_perf_case` config (bf16, HiFi2, `fp32_dest_acc_en=False`, bf16 `Layout::TILE` γ):
  - (1,1,8192,1024) `BLOCK_SHARDED` [1024,128] on 8×8 Tensix cores — compute-bound after Perf 1;
  - (1,1,8192,5120) `TensorMemoryLayout::INTERLEAVED` — the largest absolute time, DRAM/NoC-bound.

### Measured breakdown (before; the Perf 1 code, re-measured)
- **8192×1024 BLOCK: 15,584 ns, compute-bound.**
  - Ablations: compute chains stubbed 8,656; combine data writes stubbed 15,597 (flat); both 8,130; everything stubbed 8,065.
  - Per-Tensix-core TRISC0 zone totals: square 3.9 µs, collapse 1.77, combine 0.94, γ pass 4.0, inv pass 4.59. The compute thread is busy for the whole kernel (zone sum ≈ kernel span), and the exchanges are hidden (`cmp_wait_inv` 0.1 µs).
  - Each FPU pass runs at 41–48 cycles per tile.
  - Siblings: 7168×2048 BLOCK 15,353 (compute stubbed 8,470); 7168×1024 BLOCK 12,090 (7,370).
- **8192×5120 interleaved: 391,628 ns, DRAM/NoC-bound.** 168 MB at 430 GB/s, i.e. 84 % of the 512 GB/s DRAM peak.
  - Ablations: reads stubbed 301,986 (writes alone ≈ 278 GB/s); writes stubbed 236,695 (reads alone ≈ 356 GB/s); reads + writes stubbed 57,405; compute stubbed flat (388,387).
  - Per-core BRISC end times form a positional gradient: the bottom grid row ends at ~60 µs, the top at ~360 µs, and the east columns (x 13–15) at ~385 µs, with `wr_issue` totals of 250–290 µs against 15–40 µs on the west.
  - The aggregate write rate is ~constant over the span (19 / 44 / 69 / 89 % of write quanta done at 25 / 50 / 75 / 90 % of it). So a win has to raise aggregate throughput; rescheduling the tail alone would not help.
  - Siblings: 8192×1024 84,986 (reads-only 50.8 µs, writes-only 63.6 µs); 640×6144 (R3 interleaved) 43,419; 32×7168 WIDTH 3,622 (everything stubbed 2,531).
- **Ranked bottlenecks:**
  1. the tall-BLOCK compute passes (three FPU passes, near the per-pass hardware rate, so the lever is per-tile work, not overhead);
  2. the prefill DRAM/NoC aggregate rate;
  3. collapse + combine (2.7 µs).

### Portfolio (3 experiments, the cap) and verdicts
1. **`scale_fusion` — REGRESSION.** Fuse `x·γ` (Row bcast) and `·inv_rms` (Col bcast) into one DEST window, so x·γ never round-trips L1. Focus 8192×1024 BLOCK:

   | option | ns | precision |
   |---|---:|---|
   | baseline (two chains) | 15,691 | — |
   | γ first, stock dest-reuse ELWMUL (DEST→srcA) × inv, `BroadcastType::COL` | 21,250 | bit-exact |
   | inv first, dest-reuse × γ, `BroadcastType::ROW` | 21,368 | not bit-exact (max 0.125) |
   | γ first, custom raw-LLK phase 2 (16 MOVD2A per tile, no MOP) | 20,430 | bit-exact |
   | + γ srcB row-0-only unpack | 20,497 | bit-exact |

   - Why it loses (measured with diagnostic variants): the stage is bound by the FPU math thread (HiFi2 ELWMUL ≈ 48–51 cycles per tile in-op). The x·γ pack and unpack run free on TRISC2 / TRISC0, so fusion removes free work and adds MOVD2A (~40 cycles per tile) to the busiest thread. A multiply-only variant without the move is still slower (18.4 µs). Fusion also deletes the γ(0) gather shadow.
   - Under fp32 DEST it also loses precision (DEST→srcA truncates fp32).
   - Domain: a measured regression on every γ-bearing compute-bound regime (+10.8 to +43.9 %), flat when DRAM-bound. This confirms the `compute_fusion` catalog entry on Blackhole with broadcast operands.
2. **`bcast_srcb_reuse` — WIN.** Hold the broadcast operand in srcB across the tiles that share it: γ tile c for a column of tile-rows (column-major walk), inv_rms tile r for a row's tiles (row-major).
   - srcA streams one whole-tile UNPACR per tile in one unpacker context.
   - The math issues the HiFi phases itself and clears srcB only at the end of the run.
   - Same ELWMUL, broadcast mode and fidelity phases: bit-exact.
   - Focus: 15,685 → 14,801 ns (medians of 3). Options:

   | option | ns |
   |---|---:|
   | γ as stock per-tile LLK (no chain) | 15,732 (the win is the hold, not chain overhead) |
   | γ hold only | 15,494 |
   | inv hold only | 15,136 |
   | **both, with the two guards below** | **14,801** |
   | both, no inv width guard | 14,780 |

   - Why it wins in-op although the isolated bench is FPU-bound: inside the op the unpack thread runs longest during these passes (γ 3,959 vs math 3,540 ns), and the hold halves its per-tile work. Bench: unpack ~30 → ~17.5 cycles per tile; HiFi2 FPU floor ~34 cycles per tile.
   - Domain: bit-exact on every placement, input `Layout`, γ / no γ, fp32 DEST, LoFi..HiFi4, bfp8_b x (block-float srcA needs per-tile unpacker addresses: the W-counter advance skips the exponent section), masked W, multi-chunk.
3. **`dram_noc_balance` — WIN.** Interleaved inputs: the Tensix cores of the three westmost grid columns (virtual NoC x 1..3, right next to DRAM column 0) run their x reads on NoC1 and output writes on NoC0, through a second reader / writer kernel descriptor in DM_DEDICATED_NOC mode. Every other core keeps NoC0 reads / NoC1 writes. Focus 8192×5120: 390,093 → 374,972 (medians of 3), bit-exact.

   | option | ns |
   |---|---:|
   | base | 391,195 |
   | swap every core | 578,416 |
   | swap east half / west half / checkerboard | 400,679 / 412,408 / 413,081 |
   | odd banks on the other NoC (both / reads / writes / per-core parity) | 421,111 / 436,362 / 401,795 / 402,452 |
   | shortest-hop NoC per (core, bank) | 462,570 |
   | per-transaction NoC choice in DM_DYNAMIC_NOC (best searched masks) | 370,833–374,756, but dynamic mode alone costs 0.92× on 32×7168 |
   | **swap virtual x 1..3, dedicated mode** | **374,972** |
   | swap virtual x 1..2 | 372,487 |

   - Any swap that routes a stream along the DRAM column (reads on NoC1, writes on NoC0) concentrates traffic on the column links. That is why the default pairing beats wholesale re-routing, and only the columns adjacent to DRAM column 0 profit.
   - Domain: every `TensorMemoryLayout::INTERLEAVED` input, both `Layout`s (the RM stick helpers use the kernel's NoC).
   - Inexpressible: multicast senders (a sender's kernel NoC must equal its channel's multicast NoC).
   - No meaning: sharded inputs (no x / output DRAM traffic).

### Aggregation and graduation
- `scale_fusion` is discarded; it was mutually exclusive with `bcast_srcb_reuse` anyway. The two wins touch disjoint code (compute passes vs. DM NoC assignment) and are non-conflicting, so both graduated together.
- **srcB hold (`kernels/rms_norm_srcb_hold.hpp`, `hold_mul_row` / `hold_mul_col` in `rms_norm_compute.cpp`).** It is the γ pass on every path (resident and streaming γ) and the inv pass on every shard-packed path (wave-pipelined reduce-scatter, one-shot R3 sharded, R1-HS). The chain stays only for these carve-outs:
  - **γ, fewer tile-rows than one DEST window** (`HOLD_GAMMA_MIN_ROWS`; 8, or 4 with fp32 DEST). Measured regression: 256×3072 BLOCK (2 W-chunks) 18,170 → 18,416 ns; 32×7168 interleaved +0.6 %. Windows spanning columns were slower still (19,470).
  - **inv, chunks narrower than 4 tiles** (`HOLD_INV_MIN_CW`). Measured regression: 99991×64 BLOCK 120,262 → 121,847 ns; 8192×32 HEIGHT 4,135 → 4,245; 8192×64 HEIGHT 4,852 → 4,913.
  - **inv on the per-row-pushed path** (interleaved and ROW_MAJOR shards: one tile-row per CB push). I built it at graduation and measured it in-session against the chain: 32×7168 interleaved 0.995×, 32×7168 RM 1.001×, 2048×1024 RM BLOCK 0.999×, 256×2048 RM HEIGHT 1.002×, 1024×512 RM HEIGHT 5,339 → 5,408 ns (5 runs each, every new run slower). No win anywhere, so a raw-LLK helper bypass is not justified there; the helper chain stays and the hold code for that path was removed.
  - **Non-Blackhole builds** keep the chain (feasibility: these are raw Blackhole LLK sequences).
  - Every carve-out is written as "the measured-regression predicate → chain, else hold" (`gamma_keeps_chain`, `inv_keeps_chain`).
- **NoC balance (`NOC_SWAP_COLUMNS = 3`, `noc_swap_cores`, `_split_noc_swapped` in `rms_norm_program_descriptor.py`).** It applies to every interleaved input. The rule is expressed as logical x < 3, which equals virtual x 1..3 on this p150b. The only exclusion is the multicast-sender cores (inexpressible).
  - When every active core swaps (a single-core interleaved case such as 32×7168 ROW_MAJOR), only the swapped descriptors are emitted. An empty-range descriptor carries no CB formats and failed to compile; the guard set caught this.
  - The kernels build multicast role pipes through `kernels/rms_norm_mcast_roles.hpp`. On the default-NoC kernel these expand to exactly `mc.sender(noc)` / `mc.receiver(noc)`; a kernel whose cores hold no such role gets an inert stand-in.
- **Instrumentation.**
  - The hold passes sit inside the existing `cmp_gamma` / `cmp_inv` zones.
  - The compute ablation (`RMS_NORM_ABLATE=compute`) now also stubs the hold passes. Before this fix the ablation missed them (8,407 vs 12,952 ns).
  - New measurement-only switch `RMS_NORM_SRCB_HOLD=0` (define `RMS_SRCB_HOLD_OFF`) for in-session hold A/Bs; the guard test has a `holdoff` impl for it.
  - The swapped DM descriptors use the same kernel sources, so the same zones.

### Whole op, before → after (same-session A/B against the pre-Perf-2 snapshot, medians of 3, `test_rms_norm_perf_guard_ab.py`, 40 configs)
| cell | before | after | ratio | `achievable_ns` |
|---|---:|---:|---:|---:|
| 8192×5120 interleaved (focus) | 391,632 | 372,967 | **0.952** | 738,307 |
| 8192×7168 interleaved | 545,281 | 527,754 | 0.968 | 1,032,281 |
| 8192×1024 interleaved | 84,188 | 75,474 | 0.896 | 96,744 |
| 7168×2048 interleaved | 140,685 | 129,120 | 0.918 | 172,892 |
| 640×6144 interleaved (R3) | 43,884 | 38,514 | 0.878 | 101,968 |
| 32×7168 interleaved (R3) | 6,436 | 5,923 | 0.920 | 14,894 |
| 8192×1024 BLOCK [1024,128] 8×8 (focus) | 15,707 | 14,734 | **0.938** | 25,640 |
| 7168×1024 BLOCK [736,128] 8×10 | 12,091 | 11,581 | 0.958 | 18,871 |
| 7168×2048 BLOCK [736,192] 11×10 | 15,390 | 15,090 | 0.981 | 25,947 |
| 640×7168 BLOCK [64,1024] 7×10 | 9,511 | 9,331 | 0.981 | 21,669 |
| 32×1024 WIDTH [32,128] 8×1 | 2,601 | 2,544 | 0.978 | 4,110 |
| 32×7168 WIDTH [32,256] 7×4 | 3,626 | 3,595 | 0.991 | 5,481 |

- Lever split on the focus shapes: 8192×1024 BLOCK is all srcB hold (in-session hold on/off 15,747 → 14,764). 8192×5120 interleaved is all NoC balance: γ there has fewer than 8 tile-rows per core, so it keeps the chain, and its inv pass is per-row-pushed.
- Roofline after: 8192×5120 moves 168 MB at 450 GB/s (88 % of peak, was 84 %); 8192×1024 interleaved 446 GB/s (was 400).
- **Rest of the config-spanning guard set** (R1 / R1-HS / R3 interleaved / WIDTH / BLOCK × TILE / ROW_MAJOR × γ TILE / RM / none × 16-bit / fp32 DEST × bf8b):
  - Wins: no-γ 8192×5120 0.930; ROW_MAJOR γ 0.943; fp32 DEST 0.957; bf8b 8192×1024 0.962; RM 640×6144 0.856; 2048×1000 masked 0.889; 3232×96 WIDTH 0.944; 99991×64 BLOCK 0.961; BLOCK fp32 DEST 0.968; BLOCK no γ 0.979; 256×512 RM BLOCK 0.971; HEIGHT 8192×1024 0.983.
  - All other cells are 0.98–1.02.
  - Four cells sit at 1.010–1.018 against the snapshot: 256×512 HEIGHT TILE/RM-γ, 256×2048 RM WIDTH, 2048×1024 RM BLOCK, and 2048×6144 HEIGHT (noisy, 26.8–28.3 µs).
  - The in-session hold on/off A/B (same binary otherwise) put them at 1.006 (5 runs), 1.002, 0.999 and 0.987 (5 runs). That is no measured regression attributable to either graduated pattern, and sharded cells don't take the NoC swap, so no carve-out was added.
- **Golden: green.**
  - `test_golden.py` (12 `--splits` groups, final code): 5,429 passed / 0 failed (1,365 expected xfail).
  - `test_regression.py` 15/15; `test_translated.py` 104 passed + 2 expected xfail.
  - Unit dir: `test_rms_norm.py` 150, `_r3` 61, `_r6` 19, `_height_sharded` 39, `_gamma_tile` 70, `_numeric_config` 69 (+6 skip), `_precision_matrix` 328 (+104 skip), `_perf_knobs` 44, `_perf1` 44, `_r3_perf` 87 — all pass.
  - New `test_rms_norm_perf2.py`, 43/43, also under `--dev` (watcher, NoC sanitizer, LLK asserts):
    - 31 geometries bit-exact against the pre-Perf-2 snapshot, covering every call site, both carve-outs, bf8b, fp32 DEST, LoFi / HiFi4, no γ, masked W, ROW_MAJOR, and the all-cores-swap case;
    - PCC pins;
    - swap-set pins.
- **What binds now (for a later round):**
  - 8192×1024 BLOCK: compute, now on the math thread (TRISC1 14.5 of 14.7 µs). TRISC0 zones: square 3.9, inv 3.6, γ 3.3, collapse 1.7, combine 1.3 µs. Compute stubbed: 8.4 µs. The HiFi2 FPU floor (~34 cycles per tile per pass, 3 passes) is close.
  - Prefill: 88 % of DRAM peak. The per-core finish gradient remains (88–374 µs on 8192×5120), with physical columns 14–15 last; what is left looks like DRAM read/write mixing.
  - Pre-existing, unchanged: on the BLOCK reduce-scatter path the reader (NCRISC) has no zone over its last ~6 µs (`short coverage` in `_perf_report`).

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckl::eltwise_chain` + `ckl::BinaryFpu<Mul, x, BroadcastDim::Row \| Col>` (stock `llk_unpack_AB<BType>` + `llk_math_eltwise_binary<ELWMUL, BType>` per tile) | capability | BinaryFpu re-unpacks the broadcast operand into srcB for every tile and hands both source registers over one 16×16 face at a time: 4 srcA + 4 srcB UNPACRs per tile for Row, 4 + 2 for Col, one unpacker config-context round trip per tile. No `InputSpec` / policy can (a) keep srcB valid across the consecutive tiles that share a broadcast tile, (b) stream whole-tile srcA under one unpacker context with a strided walk, or (c) address per-face srcB rows inside a resident broadcast tile. A "hold the broadcast operand for N tiles" option on BinaryFpu (Row: column-major walk; Col: row-major walk) would cover it. | 15,685 (8192×1024 BLOCK, whole op) | 14,801 | `kernels/rms_norm_compute.cpp:118` (`hold_mul_row`), `:185` (`hold_mul_col`); LLK sequences in `kernels/rms_norm_srcb_hold.hpp` |

Helper friction that did NOT end in a bypass:
- `McastArgs::sender()` / `receiver()` (ergonomics): they `static_assert` `sender_available` and `sender_noc == noc_index` even inside a runtime branch a kernel's cores never take. One channel attached to two kernel descriptors on different NoCs (the NoC-swapped reader / writer) therefore cannot compile. `kernels/rms_norm_mcast_roles.hpp` wraps them with inert stand-ins.
- `Noc::async_writes_flushed<TXN_ID>` (capability): it `static_assert`s under `DM_DYNAMIC_NOC`, so the multicast Guard fence cannot compile in dynamic mode. This only affected the losing dynamic-NoC options.

- **Experiment artifacts** (benches, variant kernels, search logs): `perf_experiments/scale_fusion/`, `perf_experiments/bcast_srcb_reuse/`, `perf_experiments/dram_noc_balance/`.
- Result: all 3 ideas were measured. 2 graduated, 1 regressed (recorded, not graduated). 8192×1024 BLOCK is 0.938× and 8192×5120 interleaved 0.952×. There is no measured regression attributable to the graduated patterns.
