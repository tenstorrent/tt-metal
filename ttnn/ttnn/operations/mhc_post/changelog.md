# Changelog: mhc_post

## Phase 0 — Core Implementation
- **Date**: 2026-11-06
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - Regime: plain-2d.
  - Mix: SFPU-only fp32 `eltwise_chain` in DEST.
  - Work split: 2-D `(R, Q)` over token-tile rows × C tiles across the full grid.
  - Coefficients: column offsets are read in-kernel.
- **SUPPORTED at Phase 0**: dtype=[float32], sublayer_dtype=[float32], layout=[TILE], fp32_dest_acc_en=[True],
  mode=[plain], coef_source=[separate], alignment=[tile_aligned, h_non_aligned]
- **Accuracy achieved**: PCC ≈ 1.0 (≥ 0.99999), max_abs_err ≤ 1.07e-6, rel_rms_err = 5.3e-8, ULP p99 ≤ 14,
  signed bias ≤ 1e-9. Measured on 4 shapes via `test_mhc_post_precision_baseline.py`.
- **Golden suite at Phase 0**: 88 / 88 supported cells passing; 379 xfail_expected; 348 invalid_skipped; 0
  supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`).
- **Perf at Phase 0** (device-kernel ns, Blackhole, plain fp32):

  | Shape | Time | Notes |
  |---|---|---|
  | T160 C5120 | 125.9 µs | 100 cores |
  | T640 C7168 | 550 µs | 110 cores; the 81-op composite took 3.78 ms |
  | T4096 C1792 | 820 µs | |
  | T1 C7168 | 49 µs | |

- **Issues encountered**: one verifier fix. `validate()` now refuses sharded inputs with `ValueError`; before
  that, they would have been read through the interleaved accessor. The verifier also measured the
  `MIN_PIPELINE_BLOCKS` departure from the coarsest-block default and found it justified (see
  `verification_report.md`).
- **Tests added**: `test_mhc_post.py` (acceptance), `test_mhc_post_debug.py`, `test_mhc_post_perf.py`,
  `test_mhc_post_precision_baseline.py`.

## Refinement 1 — bf16 streams + packed mixes + stats mode (stats-2d-tree)
- **Date**: 2026-11-06
- **What was done**:
  - **bf16 streams and sublayer.** Descriptor only: CB formats come from the tensors; bf16 CBs stay Default
    unpack (exact through srcA). Probe of the packer's fp32 → bf16 rounding: X' bits equal torch's
    round-to-nearest-even of the fp32 mix on every element, and the signed-bias gate passes. No SFPU
    pre-rounding was needed.
  - **Packed mixes.** `coef_source` comes from `post` and `comb` sharing a buffer. When the post and comb
    windows share a tile column, the reader reads ONE page per (core, row) (`n_src` = 1) and expands both
    from it. The page is never sliced.
  - **stats mode (stats-2d-tree).** Each part follows the design:
    - Resident W slice, block-major, read once per core.
    - The mix chain gained a second X' pack (`cb_xprime_mm`, same bits as the written X') and an exact
      `ss += D0²` from the unrounded D0.
    - `project_block` = `matmul_block` against the W slice, plus an exact SFPU add into the row partial.
    - `finish_row_partial`: a new `SfpuRowSum` chain element (a thin UnaryOp over `sfpu_reduce`). The
      writer patches the row sums into col 24.
    - Deterministic two-level combine:
      - Fixed leaders, one per `COMBINE_FANIN` = 16 cores. The root rotates over the leaders by row.
      - Fixed slot per sender.
      - Consumer grants (credit semaphores, issued after the consumer reserves the slot set) and per-set
        cumulative arrival semaphores.
      - Leader: row − LAG. Root: row − 2·LAG.
      - Sum in slot-index order on the SFPU.
    - Caller `stats_output` is supported.
    - Shared geometry lives in `kernels/mhc_post_combine.hpp`.
  - **Exact projection at HiFi3/HiFi4.** A device finding, not in the design. Probes showed the FPU matmul
    rounds every 16-term face partial sum to ~tf32 at ANY fidelity: two exact 4-bit products in one face
    already lose ~2^-11, while one product per face, and DEST accumulation across K tiles, are exact. A
    plain `matmul_block` therefore gives rel 3e-4 at HiFi4, against the 1e-5 gate.
    - At HiFi3/HiFi4 the projection now runs `PROJ_EXACT_PASSES` = 16 passes. Pass j multiplies the X'
      block by a 0/1 column mask {j, j+16} (exact bf16 FPU multiply), matmuls it with W (one product per
      face), and accumulates with the exact SFPU add.
    - HiFi2/LoFi (the DeepSeek-V4.1 setting) keep the single matmul, at rel 4.2e-3 vs the 1e-2 gate.
    - The mask CBs are charged in sizing at every fidelity, so `(R, Q, block_c)` stays fidelity-invariant.
  - **Mode-invariant sizing.** `choose_split` prices fused calls with the stats cost (P excluded).
    `choose_block_c` sizes against the collapse inventory (`fused_fixed_bytes` / `fused_per_c_bytes`).
- **Accuracy achieved** (bf16 X'):
  - X' bit-exact RNE; PCC ≥ 0.99995.
  - Projection rel L2: 4e-8 at HiFi4 (exact path), 4.2e-3 at HiFi2.
  - Sum of squares rel 5.6e-8.
  - Stats rows bit-identical across calls (`test_output_buffers[stats]`).
- **Golden test progress** (targeted slices):
  - `test_op` stats: 58 / 58.
  - `test_op` plain (all dtype combos, separate and packed): 232 / 232.
  - `test_op_loose` stats / packed: 8 passed, 4 xfail (collapse).
  - `test_regression.py`: 15 passed, 12 skipped (collapse).
  - Unit dir: 49 passed, 2 skipped (collapse).
- **Perf** (device-kernel ns, Blackhole, bf16 / bf16, packed):

  | Call | T160 C5120 | T640 C5120 |
  |---|---|---|
  | plain | 84.6 µs (100 cores) | 343 µs |
  | stats HiFi2 | 151 µs (108 cores, R=2 Q=54) | 564 µs |
  | stats HiFi4 (exact 16-pass projection) | 275 µs | 877 µs |

  Reference bringup stats call: 103 µs.
  - Ablation: removing the ss elements from the mix chain gives 138 µs / 522 µs, so exact ss is ~9 %.
  - Rewriting the combine from up to 16 chain calls to 3 measured flat (148 → 151 µs, noise); it is kept
    as the simpler form.
  - The op is SFPU-mix-bound: plain bf16 runs ~2.6 µs per output tile. The stats delta is the extended
    per-tile chain, the per-block matmul + accumulate chains, and a 36 vs 32 tile busiest core (R=2 splits
    5 rows 3 / 2).
  - Levers left for Refinements 3 and 5:
    - the `choose_split` objective (R=5/Q=22 balances rows);
    - `COEF_DEPTH` 2, which hides the per-row coefficient expansion; it matters at 10 rows / core (T640);
    - block count;
    - a fused SFPU mix.
- **Issues encountered**:
  - The FPU accumulation precision above (fixed with the exact multi-pass projection).
  - A RuntimeArgs `VectorUInt32 +=` TypeError (fixed).
- **Tests added**: `test_mhc_post_perf.py::test_perf_bf16_packed` (plain / stats × HiFi2 / HiFi4 × T160 / T640
  perf probe). The acceptance file already covers `test_plain_mixed_dtypes`, `test_plain_packed` and
  `test_fused_modes[stats-*]`.

## Refinement 2 — collapse mode (hc SFPU program + y)
- **Date**: 2026-11-06
- **What was done**: `"collapse"` added to `SUPPORTED["mode"]`. Built collapse-2d-tree on top of stats-2d-tree;
  the split, block and stats summation order are unchanged.
  - **P in-kernel.** The reader deduplicates coefficient source pages by (buffer, tile column) on the host
    (`_coef_sources`). Packed mixes `[pre | post | comb]` are still ONE page per (core, row); a separate P is
    one more page. P_0..3 expand into `cb_coef_cols` tiles 20..23.
  - **`collapse_block`**: y = Σ_i P_i · X'_i, an `eltwise_chain` (UnaryBcast<Col> + CopyTile + MulBinary /
    Addcmul in fp32 DEST, one bf16 pack) over the bf16 X' block already in `cb_xprime_mm`. It runs after the
    mix and before the projection pops the block, so y never reads DRAM. `cb_y_tiles` goes to the writer
    (`y` or the caller's `y_output`).
  - **hc program** (`kernels/mhc_post_hc_sfpu.hpp`): a custom lane-wise fp32 SFPI program, wrapped as an
    `eltwise_chain` element (`HcProgram`) so the chain still owns DEST sync, CB lifecycle and reconfig.
    - Steps: rsqrt (bit seed + 3 Newton steps); sigmoid = exact exp (`_sfpu_exp_fp32_accurate_`) + reciprocal
      (approx + 2 Newton); row-max softmax + eps; Sinkhorn column step, then `iters − 1` row/column pairs.
      Every division is a reciprocal × multiply. No FPU operand, and no dependence on `math_approx_mode`.
    - The hc scalars and the 24 `b_k` are compute RT args (fp32 bits; 1/width computed in double on the host),
      loaded as SFPU immediates. The design's constants lane tile is therefore deleted.
  - **Lane layout probe** (design Key Risk). Running the robust layout's program on only vector 2k left every
    odd token unprocessed. So on Blackhole one SFPU load = 4 dest rows × the 8 even (or 8 odd) columns.
    - Built a **dense layout** knob (`mhc_post_hc::DENSE`, on for `ARCH_BLACKHOLE`): one load per quantity,
      all 25 quantities in one lane tile, one pass.
    - The robust 2-tile layout stays for other arches. It was verified on Blackhole by forcing the knob off.
  - **Where hc runs**: on the core that produces the row's stats (two-level: the rotating root; single-level:
    the leader).
    - The writer permutes the stats tile into lanes (`stats_to_lanes`), compute runs `HcProgram`, and the
      writer inverse-permutes (`lanes_to_hc`) and writes the hc page.
    - The writer drains hc results by polling between blocks (blocking only when the lane ring is full, or
      at the end).
    - Compute runs the hc `HC_LAG` iterations after the combine (host knob, default 1; 0 measured equal).
  - **Output plumbing**: hc (…, T, 24) fp32 and y (…, T, C) bf16 are allocated, or the caller's buffers are
    validated (shape / dtype / layout / DRAM / no aliasing). `collapse_pre` is validated as fp32 TILE
    (…, T, ≥ n). The call returns `(X', stats, hc, y)` from one `generic_op`.
  - **Small-grid fix** (`choose_split`): when no split of a small `max_grid` (3×2) meets `W_RESIDENT_BUDGET`,
    the bound relaxes to "the whole fused inventory fits `L1_CB_BUDGET` at `block_c = 1`". This is a no-op
    wherever the preferred bound is feasible, so it is mode-invariant and leaves the stats bits unchanged.
- **Accuracy achieved** (bf16 / bf16):
  - hc: rel L2 vs DeepSeek's float64 hc on the op's own stats row ≤ 1e-5 per part (`HC_OWN_REL` gate, all
    cells and HiFi2 / HiFi4); comb column sums within 2e-7 of the float64 reference.
  - y: bf16 PCC ≥ 0.99995, rel RMS ≤ 4e-3, and signed bias within the bf16 floor vs y on the op's own X'.
  - X' and stats are bit-identical to stats mode (`test_fused_modes_agree`).
  - Shapes: the 62 golden collapse cells (T 1 … 2048, C 32 … 7168, aligned and ragged, rank 2–4), plus
    (1,1,160,20480) and (1,1,150,2560).
- **Golden test progress** (targeted slices):
  - `test_golden.py` collapse: 62 / 62.
  - collapse + stats + plain-packed: 244 / 244 (348 INVALID skips).
  - `test_regression.py`: 26 / 27.
  - Unit dir: 55 / 55.
- **Perf** (device-kernel ns, Blackhole, bf16 / bf16, packed, HiFi2):

  | Shape | plain | stats | collapse |
  |---|---|---|---|
  | T160 C5120 | 84.5 µs | 152.3 µs | 191.8–192.6 µs |
  | T640 C5120 | 343 µs | 566 µs | 636.5 µs |

  - The first collapse build measured 211.5 µs at T160. Ablations at T160, one variant at a time:

    | Variant | Time |
    |---|---|
    | hc SFPU stubbed | 200.1 µs |
    | y chain stubbed | 202.4 µs |
    | hc + y compute stubbed | 191.2 µs |
    | + no y DRAM writes | 191.4 µs |
    | no hc round trip, no y | 160.6 µs |

    The lane permutes on BRISC were the hidden cost: ~1,600 volatile single-word L1 accesses per hc row, on
    the tail.
  - Non-volatile, unrolled permutes: 209.8 → 192.6 µs. Dense lanes: 211.5 → 208.9 µs. The HC_LAG / polled
    drain lever measured flat (kept as a knob).
  - Remaining collapse − stats ≈ 40 µs at T160:
    - the permutes still cost ~11 µs (181.2 µs with them stubbed), L1-latency-bound word moves on the
      writer RISC;
    - y compute ~9 µs;
    - hc SFPU ~5 µs;
    - ~8 µs of extra CB / coefficient traffic.
  - Next levers, for the perf refinements: move the in/out permutes off the writer's critical path
    (split across NCRISC / BRISC, or an SFPU-side transpose: `transpose_dest` + lane shuffles), and fold y
    into the mix chain while D0 is live.
- **Issues encountered**:
  - y was wrong on ragged-tail blocks. `cb_xprime_mm` is packed compact `[j][c]` (stride = extent, as the
    matmul reads it), so the y operand offset is `i·extent`, not `i·block_c`. Fixed.
  - **Golden-suite defect (finding, not fixable in the op):** `test_regression.py::test_hc_scalars_are_runtime`
    sets `hc_eps = 1.1e-5 / 1.2e-5`.
    - The last Sinkhorn step is a column normalization, s / (s + eps), so the comb column sums are 1 − eps.
      The float64 DeepSeek reference itself (`helpers.deepseek_hc`) has max |colsum − 1| = 1.10e-5 / 1.20e-5,
      above the fixed `HC_COLSUM = 1e-5` gate. The device gives 1.116e-5 / 1.211e-5.
    - A probe with only that gate widened to the reference's own deviation shows every other gate of the
      test passing for both seeds.
    - The gate should scale with `hc_eps` (e.g. `HC_COLSUM + hc_eps`). Proposed for a human to change in
      `eval/golden_tests/mhc_post/helpers.py`. I did not distort the hc output to meet it.
- **Tests added**: `test_mhc_post_perf.py::test_perf_bf16_packed` gains `mode=collapse`. Probe
  `probes/probe_010.py` (y tile-error map) and the lane-mapping probe are recorded above.

## Refinement 3 — Speed up the perf-flagged profile: block / split knob co-tune
- Date: 2026-11-06
- What was done:
  - Measured 40+ (R, Q, block_c, fan-in, depth) variants on device with a monkeypatching sweep
    (`test_mhc_post_knob_sweep.py`, one dispatch per variant).
  - `choose_split`, fused modes: cost = busiest core's streamed bytes + `FUSED_ROW_OVERHEAD_BYTES` (33 KiB) per
    owned row. The fit on device is ~20 µs per row vs ~11 µs per 18 KiB (row, c-tile) unit. W is dropped
    from the cost and stays a residency constraint, because it now streams under the first row. Mode-invariant:
    stats and collapse see the same inputs.
  - Tie-break, all modes: more distinct DRAM start banks (`_bank_spread`, banks from
    `device.dram_grid_size()`), then fewest cores, then smaller Q.
    - Plain T160: Q=20 puts every core's start on bank 0 (84.9 µs); Q=21/22 measure 74.0/73.4 µs.
    - Plain fp32 T1/C7168 with separate coefficients: Q=75 is 48.7 µs, Q=110 is 53.2 µs, because every core
      re-reads the shared coefficient pages.
  - Reader: the resident W slice is filled per block during the first row (cumulative `cb_push_back`). The
    compute waits the cumulative count before each first-row projection. W was previously read and barriered
    before the first F/X block (165 → 136 µs at R=5/Q=22).
  - `COEF_DEPTH` 1 → 2: the reader no longer stalls at row boundaries (T640 collapse 428 → 397 µs).
  - `choose_block_c`: the old `MIN_PIPELINE_BLOCKS` bound, capped at `BLOCK_C_TARGET` = 2 (bf16 streams) or 3
    (fp32 streams).
    - bf16: block_c 4 costs +35 %, 3 costs +12–17 %.
    - fp32 at multi-row shapes: 3 beats 2 by 25 % (T640/C7168: 617 / 457 µs).
  - Combine fan-in derived per call: `combine_fanin(Q)` = smallest of (8, 16, 32) fitting `MAX_SEMAPHORES`
    (fan-in 8 vs 16: 136 vs 140 µs at the focus, 169 vs 179 µs at R=2).
  - Reused: every kernel path, CB and helper call. Added: a host cost term, the tie-break, per-block W pushes,
    and the knob constants.
- Perf (device kernel ns, Blackhole, 110-core grid; `test_mhc_post_perf.py`, before → after):

  | Shape | plain | stats | collapse |
  |---|---|---|---|
  | bf16 packed HiFi2 T160 | 84.8 → 74.2 µs | 152.4 → 108.0 µs | **193.5 → 138.1 µs** |
  | bf16 packed HiFi4 T160 | 84.7 → 74.2 µs | 275.7 → 204.5 µs | 312.4 → 236.7 µs |
  | bf16 packed HiFi2 T640 | 342.7 → 231.3 µs | 561.1 → 332.8 µs | 636.2 → 362.0 µs |
  | bf16 packed HiFi4 T640 | 343.1 → 230.9 µs | 877.1 → 657.3 µs | 949.8 → 726.9 µs |

  Plain fp32, separate coefficients: T160/C5120 126.3 → 111.1 µs; T640/C1792 146.4 → 146.6 µs; T640/C7168
  550.0 → 460.3 µs; T1/C7168 49.3 → 49.5 µs; T4096/C1792 822.2 → 736.2 µs.
- Flat or worse (kept at the current value):
  - `IO_DEPTH`/`OUT_DEPTH` 3: 136 → 150 µs at the focus.
  - Coarser blocks: block_c 3 / 4 / 8.
  - R=1 splits: 240–260 µs at T160; 880 µs at T640.
- Accuracy achieved: unchanged gates (stats / X' bits mode-invariant; collapse hc ≤ 1e-5 rel per part, y PCC ≥ 0.99995)
  on every golden cell run below.
- Golden test progress (targeted slices):
  - `test_golden.py -k "collapse or stats"`: 124 passed / 348 skipped (INVALID).
  - `-k plain`: 332 / 332.
  - `test_regression.py`: 26 / 27. The one failure is the pre-existing `test_hc_scalars_are_runtime` `hc_eps`
    gate defect recorded in Refinement 2.
  - Unit `test_mhc_post.py`: 30 / 30.
- Issues encountered / findings:
  - The op is per-core latency- / compute-bound, not DRAM-bound: 29.5 MB in 138 µs ≈ 215 GB/s.
  - Remaining levers:
    - the SFPU mix (~2.7 µs per output tile; Refinement 5);
    - w-mcast: W is now read 5× (Refinement 4);
    - a rotated per-core block walk, which would fix bank-aligned starts at any Q (it changes the stats
      accumulation order, so it is not taken here).
- Tests added: `tests/ttnn/unit_tests/operations/mhc_post/test_mhc_post_knob_sweep.py` (`MHC_SWEEP="R,Q,block_c,fanin[,io,out,coef];..."`,
  `MHC_T/MHC_C/MHC_MODE/MHC_DT/MHC_SEP`; without env it dispatches the default config once).

## Refinement 4 — Speed up the perf-flagged profile: reuse multicast (W and coefficients)
- Date: 2026-11-06
- What was done:
  - **w-mcast**: one W injector per column group multicasts each first-row block's W slice into the R
    group members' resident `cb_w_slice` (kernel_lib `McastArgs` sender/receiver, host `ttnn.Mcast` +
    `attach`). No handshake, Counter signal: the slice is written once, so the injector never waits.
    Injectors are staggered across the row groups (`McastSenderPlacement.Staggered`).
  - **coef-mcast**: the row segment's sender reads the coefficient source page(s) and multicasts them to
    the segment's `cb_coef_src`. Handshake, because the scratch is reused every row. Receivers skip their
    reads; cores without a role read for themselves.
  - **Placement** (`core_layout`): the helper partitions receivers into equal groups that are consecutive
    in row- or column-major order.
    - Fused with R > 1 and R·⌈Q/gx⌉ ≤ gy: band layout. W uses ColumnMajor groups of R. Coefficients use
      RowMajor full-width segments plus a tail-segment channel.
    - Otherwise (plain, R = 1, or no fit): row-major layout. The coefficient channel is one RowMajor group
      of Q per row group, and W is read per core.
    - The combine transport takes physical coordinates from the same `pos(k)`.
  - Semaphores: the channels take ids after the combine's; a channel past `MAX_SEMAPHORES` is dropped
    (`attach_absent_mcast`).
  - Reused: every CB, the compute and writer kernels, the split / block / fan-in choice (stats and X' bits
    unchanged), and the reader's block loop. Added: `core_layout`, `_attach_mcasts`, the `W_MCAST` /
    `COEF_MCAST` knobs, and ~40 reader lines (roles, `share_coefficients`, a per-block W send/receive).
- Perf (device kernel ns, Blackhole 11×10; Refinement 3 → Refinement 4):

  | Config | before | after |
  |---|---|---|
  | **bf16 packed collapse HiFi2 T160 (flagged)** | **137.9 µs** | **129.5 µs** |
  | stats HiFi2 T160 | 107.3 | 96.0 |
  | plain bf16 T160 | 74.3 | 73.6 |
  | stats / collapse HiFi4 T160 | 204.5 / 236.7 | 195.1 / 228.5 |
  | stats / collapse HiFi2 T640 | 314.1 / 361.8 | 284.7 / 354.0 |
  | plain fp32 sep T160 / T1 C7168 | 109.2 / 49.3 | 94.7 / 40.7 |
  | plain fp32 sep T640 C7168 / T4096 C1792 | 460.3 / 740.7 | 454.2 / 734.9 |
  | plain fp32 sep T640 C1792 | 146.1 | 147.4 (+0.9 %, noise band) |

  - Channel split at the focus: W only 130.2, coef only 137.5, both 129.5.
  - Ceiling check (non-injector W reads stubbed, wrong output): 129.9 µs. The multicast reaches it.
- Accuracy achieved: unchanged bits. The data is moved, not computed, and (R, Q, block_c, fan-in) are
  unchanged. Every gate as in Refinement 3.
- Golden test progress (targeted slices):
  - `test_golden.py -k "collapse or stats"`: 124 passed / 348 skipped (INVALID).
  - `-k plain`: 332 / 332.
  - `test_regression.py`: 26 / 27. The one failure is the pre-existing `test_hc_scalars_are_runtime`
    `hc_eps` gate defect (Refinement 2).
  - Unit `test_mhc_post.py`: 30 / 30.
- Issues encountered:
  - With the helper's equal-consecutive-group partition, both channels cannot be single groups when
    Q > gx. A row group would have to be consecutive in row-major order while a column group is
    consecutive in column-major order, which forces a product layout (Q ≤ gx).
  - The coefficients are therefore read M = ⌈Q/gx⌉ times per row at the focus (2, was 21). W is read
    exactly once.
  - Shapes where the band layout does not fit (e.g. T640 C1792 fused, R = 20 > gy) keep per-core W reads.
- Tests added: none (the existing unit / golden / perf files cover the changed path; `test_fused_modes`
  T160 exercises the band layout with both W and coefficient channels).

## Refinement 5 — Speed up the perf-flagged profile: SFPU-bound mix
- Date: 2026-11-06
- What was done:
  - **Confirmed compute-bound first** (ablations, flagged config T160 C5120 bf16 collapse packed HiFi2, 129.3 µs):
    - mix chain stubbed to copy + packs: 101.2 µs; mix + y stubbed: 93.7 µs;
    - mix with no SFPU ops (unpacks only): 109.9 µs; mix with no coefficient / X unpacks: 104.9 µs.
    - Unpacks and SFPU ops serialize: together they cost ~28 µs, more than the sum of each alone.
  - **Fused SFPU mix** (new `kernels/mhc_post_mix_sfpu.hpp`, `MIX_FUSED` knob + `MIX_FUSED_STREAM_DTYPES`
    gate):
    - Once per row, `pack_coefficients` compacts the column-broadcast coefficient tiles into packed DEST-layout
      tiles (`cb_coef_packed`): A_j = {post_j, comb_0j, comb_1j, comb_2j}, and P = {P_0..P_3} in collapse.
    - Each coefficient is per-token (per-row), so the 4 SFPU vectors k0 + {0, 1, 8, 9} (2 parities × 2 face
      columns, the same 4 rows) need one coefficient vector. Coefficient s lives in quadrant {0,1,8,9}[s].
      Compaction is 24 in-place SFPU moves from 4 broadcast tiles, then one pack.
    - Per output tile (half-sync fp32 DEST, 4 slots): A_j, F, X_0, X_1 → `MixPass1`
      (post·F + c0·X_0 + c1·X_1); X_2 → `MixPass2`; bcast comb_3j (+ ss_run) and X_3 → `MixPass3` (+ ss).
      That is 7 unpacks (+ss) and 3 SFPU passes, instead of 10 unpacks + 5 SFPU ops + the ss op.
    - y: P, X'_0, X'_1 → `YPass1`; X'_2, X'_3 → `YPass2`. That is 5 unpacks + 2 passes, instead of 8 + 4.
    - Every pass walks the 8 row groups with a moving dst base: operands sit at constant immediate offsets,
      inner loop unrolled. Pairs of quadrants are interleaved so the MUL → MAD chains overlap.
    - Arithmetic order and rounding are identical to the per-term chain (fp32 mul, then SFPMAD, with
      Addcmul's 1.0 scale exact), so X', stats and y bits are unchanged.
  - **Path gate**: the fused mix is used on Blackhole (it relies on the probed SFPU lane mapping) with bf16 X
    streams only.
    - With fp32 streams, the per-term chain already paces a full-grid call at the aggregate DRAM limit
      (~360 GB/s). There, the faster mix exposes a NoC hotspot instead: the bottom-right cores
      (x 12–15, y 8–10) straggle.
    - Zone-measured at T4096 C1792 fp32: the mix block runs 22.4k vs 39.8k cycles and the median core
      finishes at 708k vs 966k cycles, but the slowest core takes 1141k.
    - fp32 results (per-term → fused): T4096 C1792 736 → 823 µs, T640 C7168 454 → 475, T320 C7168 236 → 260.
      Small fp32 calls win: T160 C5120 105 → 98.5, T640 C1792 148 → 131, T640 C3584 259 → 233,
      T1 C7168 40.7 → 36.3.
    - fp32 stays on the per-term chain until the hotspot is fixed.
  - Tried and kept as a non-regressing form:
    - Fully unrolled passes with a free coefficient layout (8s + g) measured the same as the loop form, so code
      size was not the T4096 cause.
    - Interleaving was flat at the focus. Inner-loop unroll took the focus from 109.1 to 106.5 µs (2 runs each).
  - Reused: the eltwise_chain, every CB, the coefficient expansion, the y / mix call sites, and the chain-element
    pattern of `HcProgram`. Added: one header (6 SFPU programs as chain elements), one compute-internal CB, 2 CT
    args, and the host knob + gate.
- Perf (device kernel ns, Blackhole 11×10; `test_mhc_post_perf.py`; before = Refinement 4 / knob-off on this
  build → after):

  | Config | before | after |
  |---|---|---|
  | **bf16 packed collapse HiFi2 T160 (flagged)** | **129.3** | **106.5–107.1** |
  | bf16 plain / stats HiFi2 T160 | 73.6 / 96.0 | 56.9 / 78.1 |
  | bf16 stats / collapse HiFi4 T160 | 195.1 / 228.5 | 177.2 / 206.8 |
  | bf16 plain / stats / collapse HiFi2 T640 | 231.3 (R3) / 284.7 / 354.0 | 172.1 / 213.7 / 266.8 |
  | bf16 stats / collapse HiFi4 T640 | 657.3 / 726.9 (R3) | 575.1 / 630.1 |
  | bf16 plain / collapse HiFi2 T4096 C1792 (new probe) | 493.7 / 835.0 | 356.7 / 643.4 |
  | bf16 plain / collapse HiFi2 T640 C7168 (new probe) | 309.9 / 476.1 | 225.1 / 348.8 |
  | plain fp32 (gated off; same path) | T160 110.3 (median of 3, R4 code) | 107.3 (median of 3); others within ±0.3 % |

- Accuracy achieved: bits unchanged by construction (same operation order). Golden gates as before: X'
  rounding / bias gates, stats SUMSQ ≤ 1e-6, hc ≤ 1e-5, y PCC ≥ 0.99995, on every cell run below.
- Golden test progress (targeted slices):
  - `test_golden.py -k "BFLOAT16-fp32_dest_acc_en"` (every bf16-X cell, i.e. every cell on the new path:
    plain / stats / collapse, separate / packed, mixed sublayer): 292 passed / 116 skipped (INVALID).
  - `test_regression.py`: 26 / 27. The one failure is the pre-existing `test_hc_scalars_are_runtime` `hc_eps`
    gate defect (Refinement 2).
  - Unit dir: `test_mhc_post.py` 30 / 30; debug + precision baseline + knob sweep 9 / 9.
- Issues encountered:
  - The SFPU `dst_reg +=` immediate is a signed 4-bit row count, so the 8-vector face skip is 4 × `+= 2`.
  - A `dst_reg` → `dst_reg` copy must go through a `vFloat`.
  - The fused-mode `fp32_copy_cbs` list was assigned, not extended; it now appends, so `cb_coef_packed` keeps
    its UnpackToDestFp32 tag.
- Findings / what is left:
  - At the focus, the SFPU passes stubbed give 104.2 µs, and the pre-change mix + y stubbed floor is 93.7 µs. The
    mix now costs ~10 µs of unpack / element overhead on the wall. The rest of the op (~94 µs) is the non-mix
    floor: DM, the projection, and the per-row combine / hc tail (the R3 fit gives ~20 µs fixed per row).
  - A per-c window (load F / X once for all 4 j) needs 5 inputs + 5 packed coefficient tiles + ss = 11 fp32
    DEST tiles, more than even full-sync's 8, so input re-unpacks per j are inherent in this form.
  - Next levers:
    - fix the fp32 NoC hotspot (`noc_placement` / `bank_stagger`), then widen `MIX_FUSED_STREAM_DTYPES`;
    - pack comb_3j into per-j packed tiles (CopyTile instead of a broadcast unpack);
    - the per-row fixed cost of the fused modes.
- Tests added: `test_mhc_post_perf.py`: plain fp32 T640 C3584 and T320 C7168, and
  `test_perf_bf16_packed_large` (T4096 C1792 / T640 C7168, plain / collapse). They are perf probes for the
  DRAM-saturated regime.
