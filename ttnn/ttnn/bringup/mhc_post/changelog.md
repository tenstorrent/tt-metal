# Changelog: mhc_post

## Phase 0 — Core Implementation
- **Date**: 2026-09-30
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - **Design:** one `ttnn.generic_op` dispatch on regime `flat_stream`: flattened (token-tile row, column tile) units split over the full grid (`row_wise=True`), segments per token row, blocks of `block_col_tiles` columns.
  - **Reader:** loads F and all n X tiles for each block with one barrier, and expands the per-row post / comb coefficients into column-broadcast fp32 tiles in L1.
  - **Compute:** mixes in fp32 DEST on the SFPU (`MulBinary` + n× `Addcmul`, comb transposed).
  - **Writer:** stores n·B X' tiles per block with one barrier.
  - One perf pass by the implementer halved the expansion's L1 stores by duplicating faces through a local NoC copy.
- **SUPPORTED at Phase 0**: dtype=[float32], sublayer_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved** (fp32/fp32, 4 shapes via `test_mhc_post_precision_baseline.py`):
  - PCC = 1.0000000000
  - max_abs_err ≤ 1.07e-06, mean_abs_err ≈ 4.1e-08
  - relative RMS ≈ 5.3e-08
  - ULP of term magnitude ≤ 2.4
  - signed bias ≤ 3e-10 (no shrink)
  - got/true ratio median 1.000000000
- **Golden suite at Phase 0**: 84 / 208 passing: 84 supported_pass, 123 xfail_expected (bf16 cells), 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`)
- **Perf at Phase 0**: fp32 loose sweep at 1.22–2.39× the DRAM roofline. T640 C7168 fp32 runs in 537 µs, vs 3781 µs for the 81-op composite.
- **Issues encountered** (verifier fixes):
  - DRY: `TILE_HW` was duplicated in the op file and the program descriptor; it is now defined once.
  - `MAX_STREAMS` is now derived from `TILE_HW` (`isqrt`) instead of being a literal.
  - No kernel changes were needed.
- **Tests added**: `test_mhc_post.py` (acceptance, planner), `test_mhc_post_perf.py` (implementer), `test_mhc_post_precision_baseline.py` (verifier)

## Refinement 1 — bfloat16 residual streams and sublayer output
- Date: 2026-09-30
- What was done: added `ttnn.bfloat16` to `SUPPORTED["dtype"]` (X, X') and `SUPPORTED["sublayer_dtype"]` (F). All four dtype combinations now run on the unchanged kernels.
  - Reused: the program descriptor already derives each CB's format and its `UnpackToDestFp32` tag from the tensor dtype, so a bf16 CB stays `Default` and enters DEST through srcA exactly. `_block_col_tiles_fit` also rederives B_fit per combo (fp32/fp32 11, bf16/bf16 23, bf16 F / fp32 X 12, fp32 F / bf16 X 21). The perf structure is unchanged: full grid (110 cores), coarsest-fit B, and one barrier per block on reader and writer.
  - Added: nothing to the kernels. The packer's fp32 DEST → bf16 page rounding measured as RNE (no systematic shrink), so the SFPU RNE fallback was not needed.
- Accuracy achieved (precision baseline, 4 shapes: T32 C32, T128 C1024, T100 C1024 non-aligned, T640 C7168):
  - bf16 X': PCC ≈ 0.999998, rel-RMS ≈ 1.65e-3 (one bf16 rounding), max_abs ≤ 3.0e-2, signed bias ≤ 1.3e-5 (within 1e-5 + 6σ), ratio median within 3.4e-5 of 1, p5/p95 0.99737/1.00263.
  - fp32 X' with bf16 F: identical to fp32/fp32 (rel-RMS 5.3e-8, ULP of term scale ≤ 2.4, bias ≤ 1.4e-9).
  - The bf16 bias is slightly negative (−2.4e-6 ± 5e-7 at T640). That is about 1000× below a truncation bias (≈ 1e-3).
- Golden test progress: 123/123 bf16 cells pass (75 `test_op` + 48 `test_op_loose`), which is all 123 previous xfail_expected. The fp32 cells are unaffected (op-file-only change). `test_regression.py` is 12/12, including `test_depth_chain[bf16]` (122 wraps).
- Perf (device kernel ns, 110 cores, bf16 vs fp32): T640 C7168 388 µs vs 538 µs; T640 C1792 151 µs vs 187 µs; T1280 C4096 435 µs vs 628 µs. These are still well above the DRAM roofline (~202 / ~50 / ~231 µs) because the SFPU mix is the bound. Refinements 2 and 3 target this.
- Issues encountered: None.
- Tests added:
  - `test_mhc_post.py`: DTYPES now spans all 4 combos; `test_mhc_post_deterministic` is parametrized over fp32 / bf16.
  - `test_mhc_post_precision_baseline.py`: 4 dtype combos; the bf16-output gates are PCC 0.9999, rel-RMS < 4e-3, bias ≤ 1e-5 + 6σ, and ratio median within 1e-3.
  - `test_mhc_post_perf.py`: a dtype axis and a T1280 C4096 shape.

## Refinement 2 — Speed up the compute-bound SFPU mix on the perf-flagged bf16 profiles
- Date: 2026-09-30
- What was done: restructured the compute mix. The design's levers (a) and (b) were both applied in modified form. Lever (c), the FPU split, was not needed.
  - **Diagnosis** (ablation, T640 C7168 bf16, 388 µs): stubbing the SFPU gave 346 µs; stubbing the coefficient copies too gave 236 µs (≈ the 234 µs DM floor). Coefficient copies alone, with no switching, were cheap (242 µs). So the cost was the per-element switch between the fp32 UnpackToDestFp32 path and the bf16 srcA path: the old per-tile chain did 2(n+1) switches per output tile, each a stalled reconfig plus re-init.
  - **SyncFull DEST window** (lever a): the host sets `dst_full_sync_en` via the new `DST_FULL_SYNC` knob, giving 8 fp32 slots. Per output stream j, each window is one `eltwise_chain` iteration. It copies stream j's coefficient tiles once, then the n+1 data tiles of each window column, grouped by source CB. Coefficients cannot stay resident across windows, because every pack release does a full ZEROACC.
  - **`WeightedSum`** (lever b, as a fused block op): a custom DEST-only chain element (raw SFPI, UnaryOp CRTP). It computes `Σ_t d_t·c_t` in one SFPU pass, replacing `MulBinary` + n× `Addcmul`. The arithmetic is the same (`d_0·c_0`, then fused MADs in term order). The accumulator stays in LREGs, and each coefficient vector is read once for the two data faces of the same rows. The helper-substitution justification is at the top of the compute kernel.
  - **Half-packed coefficient layout** (`mhc_post_common.hpp`): term t of stream j occupies faces 0/2 (t even) or 1/3 (t odd) of tile j·P + t/2, with P = ceil((n+1)/2). The reader writes each half directly; the NoC face-duplication reads are gone. `cb_coef_bcast` shrinks from 20 to 12 tiles at n=4, which raises B_fit by 1–2 (ledger). With this layout, 3 coefficient + 5 data tiles fit the 8 slots at n=4.
  - **Reconfig trimming**: consecutive windows alternate coef-first / data-first. Each window's first copy starts on the CB the previous window ended on, so its unconditional reconfig is disabled. The per-window pack reconfig is disabled too, since the output CB is the only pack target.
  - **Regimes by n**: n ≤ 2 fits K > 1 columns per window; n = 3, 4 fit one; n = 5 runs a grouped path (terms accumulated in groups). K and the regime derive from `DEST_AUTO_LIMIT` in the kernel.
  - Reused: SegmentWalker, all CB lifecycles, the reader/writer block structure, and CopyTile / PackTile chain elements. Added: `WeightedSum`, the window generators, the half-packed expansion, and CT arg `coef_tiles_per_stream`.
- Perf (device kernel ns, 110 cores, Blackhole; before → after):
  - T640 C7168 bf16: 388.3 → 305.5 µs. Target ≈ 202 µs; DM floor with compute stubbed 224 µs.
  - T1280 C4096 bf16: 429.3 → 368.0 µs. Target ≈ 231 µs; floor 269 µs.
  - T640 C1792 bf16: 158.6 → 132.4 µs. Floor 116 µs.
  - fp32/fp32: T640 C7168 526 → 478 µs, T640 C1792 187 → 154 µs, T1280 C4096 625 → 547 µs.
  - Guard set ({fp32/fp32, bf16/bf16, bf16 F / fp32 X} × {T640, T1000 non-aligned} × {C1792, C7168}; `test_mhc_post_perf_guard`): every cell is faster, by 2–19%. No regression.
    - C1792 T640: 175 → 171 / 151 → 133 / 168 → 153 µs.
    - C1792 T1000: 248 → 224 / 189 → 167 / 243 → 216 µs.
    - C7168 T640: 532 → 469 / 388 → 304 / 527 → 482 µs.
    - C7168 T1000: 885 → 788 / 566 → 460 / 875 → 760 µs.
- Accuracy achieved: the arithmetic is unchanged. The precision baseline passes unchanged: bf16 PCC ≈ 0.999998, rel-RMS ≈ 1.65e-3, signed bias within the 1e-5 + 6σ gate; fp32 rel-RMS ≈ 5e-8. `test_depth_chain[bf16]` (122 wraps) passes, and bitwise determinism holds (`test_mhc_post_deterministic`). fp32 n ∈ {1,2,3,5} matches the reference at rtol = atol = 1e-5.
- Golden test progress: `test_op_loose` 96/96 (all perf-sweep cells, fp32 + bf16); `test_regression.py` 12/12; acceptance 42/42. `test_op` was not re-run in full: it is covered by the acceptance shapes and all four dtype combos.
- Issues encountered:
  - The SFPU face walk: `_llk_math_eltwise_sfpu_inc_dst_face_addr_` rebases on the carry (face-start) register and discards in-loop `dst_reg++` steps, so reaching face 2 from face 0 takes two calls. Found with a structured probe (face 0 was correct, faces 1–3 were wrong).
  - `dst_reg += 8` does not compile (the INCRWC immediate range is [-8, 7] DEST rows).
- Remaining headroom (finding, not a follow-up): the compute excess over the DM floor fell from ~155 to ~80 µs at T640 C7168 bf16. What remains is the serialized SyncFull window: fp32 unpack-to-DEST of 3 coefficient tiles, 5 data copies, one SFPU pass (~27 instr per SFPU row-pair) and one pack per output tile, with no pack/math overlap. Next levers:
  - a coefficient form that survives a DEST release, or a denser coefficient layout that frees slots for K > 1;
  - reusing data tiles across output streams, which needs n·P + n + 1 slots.
  The DM floor itself (224 vs 202 µs target) is Refinement 3.
- Tests added: `test_mhc_post_streams.py` (n ∈ {1,2,3,5} × aligned / non-aligned × fp32 / bf16 — the n-dependent window regimes); `test_mhc_post_perf.py::test_mhc_post_perf_guard` (the no-regression guard set).

## Refinement 3 — Reader / writer overlap and coefficient-expansion cost on the perf-flagged bf16 profiles
- Date: 2026-09-30
- What was done: data-movement levers. No SUPPORTED change.
  - **Diagnosis** (ablation on the R2 kernel; T640 C7168 / T640 C1792 / T1280 C4096 bf16):
    - Full: 306.6 / 131.5 / 370.0 µs.
    - Expansion stubbed: 278 / 78 / 321 µs.
    - Compute + expansion stubbed (DM floor): 182 / 48.5 / 209.5 µs.
    - The coefficient expansion was the largest single DM cost: 54 µs of 131 at C1792. It ran at ~3.5 cycles per store.
    - Per-core maps showed a physical-row gradient: grid row 2 is slowest, row 11 fastest, independent of unit order (`row_wise` False measured the same).
  - **Tight expansion loop** (`mhc_post_coef_expand.hpp`): one raw load per row feeds 16 unrolled word stores, over the two contiguous faces of a half-tile. It replaces per-element face-offset math plus a `fill_l1_range` call per row. Bit-exact fp32 word copies. This alone took C1792 from 131.5 to 86 µs.
  - **`COEF_EXPANDER` knob** (default `"writer"`, `"reader"` kept live):
    - load_coefficients is a shared `CoefExpander` that exactly one DM kernel runs, so `cb_coef_raw` and `cb_coef_bcast` keep a single producer.
    - The writer (BRISC) is idle until the first output block. It loads segment 0's set up front and segment s+1's right after writing segment s's first block, using a look-ahead `SegmentWalker` (same derivation). This needs `COEF_DEPTH ≥ 2`, which the host asserts.
    - On the reader path, the raw read rides block 0's barrier and the expansion runs in block 1's read shadow.
    - Writer vs reader, measured back to back: 66 vs 76 µs (C1792), 238 vs 246 µs (C7168), 267 vs 273 µs (T1280).
  - **Per-stream coefficient pushes:** the expander pushes P tiles per output stream. Compute waits cumulatively for `(j+1)·P` before stream j, so it starts mixing stream 0 while later streams are still being expanded.
  - **Block-size policy:** `B = min(B_fit, longest segment, MAX_BLOCK_COL_TILES = 8, ceil(max units per core / MIN_BLOCKS_PER_CORE = 3))`. The coarsest fit left 1–2 blocks per core, so there was no read / mix / write overlap and every core burst its full prefetch at DRAM at once.
    - Swept B ∈ {1, 2, 3, 4, 6, 8, 10, 12, 16}: B = 8 was best at C7168 (≈ 221 µs vs 263–320 µs), B = 4 at C1792, and B = 8–12 at T1280.
    - `DEPTH_IN` = 3 measured slower (B = 11: 252 → 293 µs) and was not adopted.
  - Reused: SegmentWalker, all CB lifecycles, the compute mix, the reader's block reads. Added: `mhc_post_coef_expand.hpp`, the reader/writer CT flag `expand_here`, the writer's post / comb accessors and RT args, and the knobs `COEF_EXPANDER` and `MIN_BLOCKS_PER_CORE` (`MAX_BLOCK_COL_TILES` changed from None to 8).
- Perf (device kernel ns, 110 cores, Blackhole; R2 → R3):
  - bf16 flagged: T640 C1792 131.5 → 66–71 µs (target ≈ 50); T640 C7168 306.6 → 238 µs (target ≈ 202); T1280 C4096 370.0 → 267 µs (target ≈ 231).
  - fp32/fp32: T640 C7168 478 → 450 µs; T640 C1792 154 → 134 µs; T1280 C4096 547 → 531 µs.
  - Guard set, fp32 / bf16 / mixed. Every cell is faster; no regression:
    - C1792 T640: 171 / 133 / 153 → 138 / 67 / 128 µs.
    - C1792 T1000: 224 / 167 / 216 → 197 / 113 / 192 µs.
    - C7168 T640: 469 / 304 / 482 → 434 / 220 / 425 µs.
    - C7168 T1000: 788 / 460 / 760 → 706 / 398 / 677 µs.
- Accuracy achieved: the arithmetic is unchanged (the coefficients are the same fp32 words). The precision baseline passes unchanged: bf16 PCC ≈ 0.999998 and rel-RMS ≈ 1.65e-3; fp32 rtol = atol = 1e-5 (streams tests). Determinism tests pass.
- Golden test progress: `test_regression.py` 12/12 (incl. the 122-wrap depth chain) and `test_golden.py -k loose` 96/96 (108 passed). The unit dir passes 116/116. The full `test_op` set was not re-run; it goes through the same kernels and is covered by the acceptance and knob tests.
- Issues encountered: none. Ablation hooks were temporary and have been removed.
- Remaining headroom (finding, not a follow-up):
  - The DM floor (compute stubbed) is now 187 / 50 / 209 µs, at or below target, and compute alone (DM stubbed) is 184 / 62 / 207 µs. The stages are balanced.
  - The wall exceeds both because per-core DRAM service is uneven by core position: about 170–215 µs per core at C7168. The slowest core sets the kernel time.
  - Next levers: a position-weighted work split; a compute cut (R2's serialized DEST window); splitting the expansion across both DM RISCs by segment parity (two coefficient CBs and a second compute instantiation — TRISC code-size risk, not attempted).
- Tests added:
  - `test_mhc_post_dataflow_knobs.py`: 24 cases covering `COEF_EXPANDER` writer / reader × {default policy, coarsest fit, B = 1, B = 3} × {T640 C1792 bf16 (row-straddling cores), T100 C224 fp32 non-aligned, n = 5}.
  - `test_mhc_post_perf.py`: `MHC_POST_SWEEP_MAX_BLOCK` env sweep; `max_block = None` now keeps the descriptor's policy.

## Perf 1 — perf tournament round 1 (2 experiments: 1 graduated, 1 regression)
- Date: 2026-09-30. Box: Blackhole p150, 110 cores. All numbers are DEVICE KERNEL DURATION.
- Focus (feature_spec `_PERF_FOCUS`, all bf16 streams, fp32_dest_acc_en = True, TILE, DRAM interleaved; all in SUPPORTED): T640 C7168, T640 C1792, T1280 C4096.
- **Instrumentation (permanent)**: `MaybeDeviceZoneScope` on every stage boundary, from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`.
  - Compute: `compute_wait_in` / `compute_wait_coef` (the unpack starved on reader / expander), `compute_reserve_out` (writer back-pressure), `compute_mix` (occupancy).
  - Reader: `reader_reserve` / `reader_issue` / `reader_barrier` / `reader_wait_help`.
  - Writer: `writer_help_read` / `writer_issue` / `writer_barrier` / `writer_coef_start` / `writer_coef_expand`.
  - The waits have their own zones, so each zone isolates either starvation or work.
  - With read help on, the per-term coefficient steps carry no zone: one step per event-loop pass would exhaust the 250-marker budget.

### Measured breakdown (op at Refinement 3; focus T640 C7168 / T640 C1792 / T1280 C4096)
| cut | us |
|---|---|
| full | 219.5 / 67.5 / 267 |
| DM floor (compute + expansion stubbed) | 188.7 / 52.3 / 214.6 |
| compute only (NoC stubbed, CB handshakes kept) | 173.5 / 50.5 / 197.7 |
| compute only, SFPU also stubbed | 124.8 / 38.2 / 141.6 |
| compute stubbed, expansion kept | 196.8 / 65.2 / 219.1 |
| SFPU stubbed, rest full | 246.5 / 73.6 / 286.9 (faster compute alone regresses: DRAM fairness) |
| sync floor (everything stubbed at once) | 7.5 / 6.2 / 7.9 |
| DRAM target (0.8 × 512 GB/s) | 201.7 / 50.4 / 230.5 |
- Ranked bottleneck:
  - The DM floor is DRAM-bound at ~85% of DRAM peak (~441 GB/s at C7168 / T1280), so it is at roofline in aggregate. Compute is below the DM floor.
  - The wall exceeds the floor for two reasons:
    - (a) the pipeline tail after each core's last read (median 37 µs at C7168, 25 µs at C1792), from block-granular (B = 8) CB hand-off;
    - (b) uneven per-core DRAM service by grid position: the slowest core sets the wall.
  - Speeding up compute alone does not help, because the stages are balanced.

### Portfolio (cap: 2 experiments)
1. **column_stream**: a column-granular pipeline. The reader pushes per column with trid-tracked reads (same bytes in flight); compute is column-outer; the writer writes per column and flushes before pop. Target: (a), the tail.
2. **split_noc_reads**: reader/writer NoC balance. BRISC (idle ~80% of the kernel) issues a share of the input reads alongside its writes. Target: (b), the slow cores.

### Verdicts
- **column_stream — REGRESSION, not graduated.** 3 repeats, focus bf16:
  - baseline 223.1 / 67.4 / 269.9 µs;
  - pure column stream (G = 1): 299.8 / 76.6 / 335.5;
  - DM-coarse + column hand-off: 229.4 / 74.8 / 294.7;
  - G = B: 219.4 / 67.3 / 268.1, identical to the baseline.
  - The column trickle makes the slow cores' reader-issue stalls worse (up to 165 µs at core (3,3)), and the per-column writer hurts too.
  - Its one win was fp32-X T640 C1792 (1.10–1.20×), but fp32 T1000 C1792 was −10% and fp32 T640 C4096 −5%. Artifacts: `perf_experiments/column_stream/`.
- **split_noc_reads — WIN over a measured domain, graduated with two measured carve-outs.**
  - Measured variants: BRISC reads F / X streams on NoC1 or NoC0, NCRISC writes output streams, dynamic-NoC mode, split-CB compute, event-loop and "helper DMA" schedules.
  - Nulls and regressions:
    - Any read on NoC1: DM floor +4–5%, full +6–20%.
    - NCRISC write help: +3–9%.
    - Writes on NoC0: +10–17%.
    - Split-CB compute: compute alone +4–8%.
    - Help from block 0: T1280 +5%.
  - Winner, `hF_n0_hf1_inc`:
    - BRISC reads the F block of every block ≥ 1 straight into the reader's reserved window, on NoC0 (dynamic-NoC mode), with a two-semaphore hand-off.
    - The op's CBs and compute kernel are unchanged, and the coefficient expansion is incremental (one term per event-loop pass).
    - X-stream help (`hX1_n0_hf1_inc`) measured within 2% of it. F was chosen because its share is independent of n.
  - Bit-exact against the op on 17 shapes: n = 1/2/3/5, fp32, mixed dtype, non-aligned, row-straddling, batch.
  - Help on, bf16, `orig` → help (median of 3 where repeated):

    | shape | blocks / core | orig µs | help µs | Δ |
    |---|---|---|---|---|
    | T1280 C4096 (focus) | 6 | 268 | 246 | −8% |
    | T640 C7168 (focus) | 6 | 218 | 213 | −2.5% |
    | T640 C1792 (focus) | 3 | 67 | 80 | +18% |
    | T1024 C5120 | 6 | 275 | 245 | −11% |
    | T1280 C6144 | 9 | 422 | 401 | −5% |
    | T1024 C7168 | 9 | 406 | 385 | −5% |
    | T4096 C2560 | 10 | 613 | 582 | −5% |
    | T2048 C4096 | 10 | 451 | 433 | −4% |
    | T2048 C7168 | 17 | 845 | 818 | −3% |
    | T4096 C1792 | 9 | 412 | 408 | −1% |
    | T1024 C4096 | 5 | 207 | 205 | −1% |
    | T2048 C1792 | 5 | 182 | 189 | +4% |
    | T512 C5120 | 3 | 150 | 156 | +4% |
    | T256 C1792 | 3 | 37.4 | 39.8 | +7% |
    | T640 C4096 | 3 | 148 | 151 | +3% |
    | T256 C7168 | 3 | 112 | 115 | +2% |
    | T1024 C2560 | 3 | 150 | 153 | +2% |
    | T640 C2560 | 3 | 99.6 | 96.5 | −3% |
  - fp32-X, help on: T1000 C7168 X fp32 / F bf16 678 → 700 (+3.2%, 3 runs, spread < 1%); X fp32 / F fp32 713 → 725 (+1.7%); T640 C7168 fp32 and T1280 C4096 fp32 −1…−2%; T640 C7168 mixed −0.5%.
  - Help off (the same event-loop kernel): within ±2% of `orig` on 13 shapes. Tiny single-block shapes (T32 C32, T17 C128) +0.6 µs.
  - Precision: no change. The help moves bytes only, and the arithmetic is the same kernel (bit-exact).

### What graduated (`mhc_post_dm.cpp`, one DM source for both RISCs; the old `mhc_post_reader.cpp` / `mhc_post_writer.cpp` were deleted)
- The DM path is now one kernel everywhere:
  - role 0 (NCRISC) is the reader; its blocking loop reads F unless the block is helped.
  - role 1 (BRISC) runs an event-loop writer: output blocks, coefficient loads, and read-help requests.
- Read help is compiled in unless a carve-out applies. Both carve-outs have the right polarity (help off where it cannot pay):
  1. The busiest core has < `HELP_MIN_BLOCKS` = 6 blocks. Measured +4…+18% at 3–5 blocks (table above). This keeps focus T640 C1792 on the unhelped schedule.
  2. X is float32 (`HELP_FP32_STREAMS = False`). That datapath is compute-bound; measured +3.2% at T1000 C7168 X fp32 / F bf16.
- With help off, the coefficient set is expanded in one go, as the old writer did. The incremental expansion measured T640 C1792 bf16 at a median of 68.9 vs 67.0 µs; in one go it is 67.35 µs.
- Deleted:
  - The `COEF_EXPANDER` knob and the reader-side expansion. The writer (the R3 default, measured faster) is now the only expander.
  - `CoefExpander::expand()` / `load()`, replaced by a resumable `start()` / `step()` job.
- New host knobs: `HELP_MIN_BLOCKS`, `HELP_FROM_BLOCK` = 1, `HELP_FP32_STREAMS`, `SEM_RD_GO` / `SEM_RD_DONE`. The NoC mode is `DM_DYNAMIC_NOC` only when help is on.
- Whole op after graduation (`test_mhc_post_perf.py`; `perf_experiments/split_noc_reads/graduated_whole_op.txt`), before → after:
  - Focus bf16: T640 C7168 218.3 → 213.7 (−2%); T640 C1792 67.0 → 67.1 (flat); **T1280 C4096 268.9 → 246.0 (−8.5%)**.
  - fp32 perf cells (help carved out): T640 C7168 ~444 → 437; T640 C1792 ~134 → 136; T1280 C4096 ~532 → 533. All flat.
  - Domain (bf16), for example: T1024 C5120 275 → 247 µs, T1280 C6144 422 → 396 µs, T4096 C2560 613 → 585 µs, T1024 C7168 406 → 381 µs. All help-off shapes are within ±2% of before.
- Guard set (fp32 / bf16 / mixed × T640 / T1000 × C1792 / C7168; `orig` measured in the same session): every cell is flat within noise, except T1000 C7168 bf16 at 403 → 385 µs (−4%).
  - The T640 C1792 bf16 guard cell ranges 67–73 µs on the same compiled program, which also measures 67.1 µs in the perf test. That matches Refinement 3's recorded 66–71 µs run-to-run range.
- Correctness:
  - Golden `eval/golden_tests/mhc_post/`: 208/208, run after every kernel/descriptor change.
  - Unit dir: 110/110 + 33 perf.
  - The knob test now covers help default / forced on (incl. fp32) / off × block policies × {T640 C1792 bf16 row-straddling, T100 C224 fp32 non-aligned, n = 5}.

### Findings (not follow-ups)
- **Escaped op bug:** found by the split_noc_reads optimizer; it predates this round.
  - n = 2 with B ≥ 2 does not compile: the SFPU register spills (`cannot write SFPU object to memory`, `mhc_post_compute.cpp` `WeightedSumSfpu::row`). The optimizer reproduced it on the real op at T640 C1792 n = 2 bf16.
  - Under the device profiler, n = 1 also fails to build (T640 C1792 n = 1).
  - The existing n = 2 tests use only B = 1 shapes. This is a generality gap to report upstream, not a perf lever.
- Remaining headroom: C1792 (the pipeline tail at ~3 blocks per core) and the 3–5-block band, where the help does not amortize. The fp32 cells are compute-bound.

### Helper bypasses — none
The new DM kernel uses dataflow_api only: noc_async_*, CB sync, L1 semaphores. There is no raw LLK. The compute kernel is unchanged; its R2 `WeightedSum` justification still stands.

## Perf 2 — perf tournament round 2 (2 experiments, both graduated)
- Date: 2026-09-30. Box: Blackhole p150, 110 cores. All numbers are DEVICE KERNEL DURATION.
- Focus: feature_spec `_PERF_FOCUS`, all bf16 streams, fp32_dest_acc_en = True, TILE, DRAM interleaved. All three are in SUPPORTED: T640 C7168, T640 C1792, T1280 C4096.
- Instrumentation: the Perf 1 zones were reused unchanged. No new stage was added: the eager load reuses `writer_coef_start` / `writer_coef_expand`, and the split is host-only.
- Artifacts: `perf_experiments/r2_breakdown/` holds the ablation reports, the per-core tools `coretl.py` / `coregrid.py`, and the whole-op before/after CSVs with `final_table.txt`.

### Measured breakdown (op at Perf 1; focus T640 C7168 / T640 C1792 / T1280 C4096, µs)
| cut | µs |
|---|---|
| full | 213.7 / 67.8 / 246.3 |
| compute stubbed (CB handshakes and coefficient expansion kept) | 223.0 / 66.0 / 247.7 |
| NoC stubbed (compute kept) | 183.5 / 60.0 / 207.9 |
| SFPU stubbed, rest full | 246.9 / 72.8 / 280.5 |
| compute + NoC stubbed at once (sync + expansion floor) | 43.7 / 28.1 / 43.7 |
| DRAM target (0.8 × 512 GB/s) | 201.7 / 50.4 / 230.5 |

Per-core timelines give the model: wall = head + compute + second-set stall.
- **Head:** the core waits for its first block's reads. This is 6 µs on the fast cores and up to 45 µs on the slow ones (T640 C7168 core (1,3): the block-0 barrier alone is 43 µs).
- **Compute:** back to back, 32.5 µs per 8-column block, never starved after the head. At that pace the 110 cores demand ~95% of DRAM peak.
- **Second-coefficient-set stall:** hits cores that straddle two token rows. It is 7.8 µs at T640 C1792, and 15.3 µs on the slowest T1280 core. The cause: the writer started set s+1 only after writing segment s's first output block.
- The head depends on grid position: rows y = 2–3 run ~205–214 µs and row y = 11 ~173–182 µs. So the fast cores sit idle for up to 40 µs at the end.

Ranked bottleneck:
1. **Makespan imbalance across cores** (head skew). Compute alone is not the limiter: faster compute measured slower, because DRAM service gets less fair.
2. **Second-set stall on straddling cores.** It dominates at C1792 and on the slowest T1280 core.

The whole-op DM and compute stages are balanced: the wall equals the compute-stubbed floor, and both sit above the compute-only run.

### Portfolio (cap: 2 experiments)
- **Selected:**
  1. **coef_prefetch**: eager look-ahead of the next coefficient set, a cheaper expansion loop, and an optional NCRISC share of the expansion.
  2. **core_balance**: equalize per-core finish times, either by NoC-atomic dynamic tail claiming or work stealing, or by a static position-weighted split.
- **Floated, not selected** (with the measured reason):
  - Coefficient-stationary DEST (8 → 5 copies per output tile) and an SFPLOADMACRO `WeightedSum`: compute pace already equals steady-state DRAM pace, and faster compute measured slower (SFPU stub 246.9 vs 213.7 µs).
  - A small first block: steady-state DRAM is saturated, so the starvation just moves to block 1. It was folded into core_balance as a candidate and measured a regression.

### Verdicts
- **coef_prefetch — WIN over the domain; flat on the focus shapes in isolation.**
  - Bit-exact on 22 shapes (n = 1/2/3/5, fp32, both mixed pairs, non-aligned, T32, batch, rank 4).
  - Medians, µs:

    | variant | T640 C7168 | T640 C1792 | T1280 C4096 |
    |---|---|---|---|
    | base | 213.7 | 67.7 | 246.9 |
    | eager | 212.8 | 67.0 | 244.8 |
    | eager_fast | 214.9 | 67.1 | 244.3 |
    | eager_share_fast | 214.5 | 66.8 | 245.7 |

  - Domain, base → eager_fast: T512 C2560 83.3 → 71.6 (−14%), T1024 C1792 113.8 → 108.5, T256 C1792 37.6 → 36.3, T2560 C6144 901 → 879, T640 C4096 148 → 145. All other cells are flat within ±1.1%, including fp32 and mixed.
  - The share variant beats eager_fast only at T256 C1792 (33.8 µs). It was not graduated: it needs two more semaphores and ~n(n+1)/2 NCRISC zones per set, which puts the marker budget at risk on many-segment cores.
  - Measured regression: HELP_NB (polling the help flush) +2–8%.
  - Domain: everywhere, with no exceptions.
- **core_balance — WIN for the static row-weighted split (ow16).** Every run-time balancing candidate was a REGRESSION:
  - NoC-atomic tail pool: +4…+28%.
  - Per-core queues with stealing: +2…+56%.
  - Ramped first blocks: +1…+28%.
  - Why the run-time options lose: each stolen chunk needs a new ~13 µs coefficient set, claims and scans cost 4–30 µs, and the final writes stall.
  - ow16 weights each core 1 + 0.16·(y − y_mid)/(rows − 1) by logical grid row, host only. Bit-exact on 40 shapes; helped on cards 0, 2 and 3. 0.08 / 0.12 / 0.20 / 0.24 measured worse overall.
  - Focus: 213.2 → 211.2 / 67.3 → 63.2 / 245.4 → 232.1 µs. Domain bf16: −3…−10%.
  - fp32-X in isolation: T1000 C7168 X fp32 / F bf16 +9.4%.

### What graduated (one unified path each; the replaced code was deleted)
1. **Eager coefficient look-ahead** (`mhc_post_dm.cpp`). The writer starts set s+1 as soon as set s's job ends and `cb_pages_reservable_at_back(cb_coef_bcast, n·P)` is true. The check is non-blocking; COEF_DEPTH = 2.
   - Deleted: `loads_pending`, the bump on the first output block, and `Blk::first_of_segment`.
2. **Software-pipelined `expand_half`** (`mhc_post_coef_expand.hpp`). The load of row r+1 is issued ahead of row r's 16 stores. A set now takes 11.6 µs instead of 13.3 µs.
   - The old loop was deleted.
3. **Row-weighted work split** (`_work_assignment(grid, units, row_weight)`, `ROW_WEIGHT = 0.16`). It keeps the same cores and the same contiguous r-major order, and every core still gets at least 1 unit. The block policy and read-help carve-out derive from the returned assignment.
   - **One carve-out**, earned by a measured regression: X float32 with F bfloat16 keeps the uniform split (`ROW_WEIGHT_MIXED_FP32_STREAMS = 0.0`).
     - Re-measured on the graduated kernels, uniform vs weighted, 2 runs each: T1000 C7168 679.5 vs 705.1 µs (+3.8%), T1000 C1792 178.1 vs 185.9 µs (+4.4%).
     - The same dtype pair gains at T640 (−6.8% / −1.8%), but no predicate separates those cells from the regressions.
   - The subagent had proposed carving out all fp32 X. With the graduated kernels, fp32 / fp32 measured −6.5% (T640 C1792), −1.3…−1.6% (C7168) and +0.2% (T1000 C1792, flat). So fp32 / fp32 takes the weighted split, and the carve-out was narrowed to the mixed pair.

### Whole op, before (Perf 1 op, 3 runs) → after (graduated), same session
Medians in µs. bf16 cells use 8 runs of the identical bf16 program; fp32 cells use the 2 final runs. Full table: `r2_breakdown/whole_op/final_table.txt`.
- **Focus bf16:**
  - T640 C7168: 213.1 → 205.5 (−3.6%).
  - T640 C1792: 67.1–72.5 → 62.3 (−7…−14%).
  - T1280 C4096: 245.6 → 230.4 (−6.2%; one 253 µs outlier in 8 runs).
  - Remaining gap to the DRAM target: +1.9% / +24% / −0.04% (T1280 now meets its 230.5 µs target).
- **fp32 perf cells:** T640 C7168 449.7 → 445.4, T640 C1792 134.3 → 115.4 (−14%), T1280 C4096 532.4 → 509.7 (−4.3%).
- **Guard set** (fp32 / bf16 / mixed × T640 / T1000 × C1792 / C7168): every cell is faster or flat. No regression.
  - C1792 T640: 132.9 / 67.1 / 133.4 → 117.5 / 62.7 / 121.8.
  - C1792 T1000: 193.3 / 115.0 / 196.8 → 191.5 / 98.3 / 178.8.
  - C7168 T640: 448.8 / 214.2 / 429.7 → 438.1 / 206.0 / 427.1.
  - C7168 T1000: 712.7 / 380.3 / 678.1 → 708.8 / 354.7 / 677.2.
- **Domain bf16** (15 cells): −1.6% to −13%.
  - T1024 C2560 149.9 → 130.4, T256 C7168 111.4 → 97.4, T2048 C1792 184.2 → 169.0, T640 C4096 150.3 → 136.7, T1024 C7168 382.7 → 357.6.
  - T2048 C4096 431.1 → 430.5 is flat.
- **Correctness:**
  - Golden `eval/golden_tests/mhc_post/`: 208/208, run after each graduation step.
  - `test_regression.py`: 12/12.
  - Unit tests: acceptance 42, streams 16, dataflow knobs 36, precision baseline 16.
  - Precision is unchanged: the arithmetic is the same kernel, and outputs are bit-exact against the Perf 1 op.

### Findings (not follow-ups)
- The low grid rows' first-block service is a Blackhole p150 property, seen on 3 cards. The row weighting is untested on Wormhole and on other harvesting patterns. It applies there too, because untested is not excluded.
- Remaining headroom:
  - C1792 is bound by head latency plus the per-set expansion (~11.6 µs per set on BRISC). The NCRISC share measured another −7% only at T256 C1792.
  - Big-C bf16 cells are within ~2% of the DRAM target. The residual is per-core DRAM service skew.

### Helper bypasses — none
The graduated changes use dataflow_api only (`cb_pages_reservable_at_back`, L1 word stores) plus a host-side split. There is no raw LLK. The compute kernel is unchanged, and its R2 `WeightedSum` justification still stands.
