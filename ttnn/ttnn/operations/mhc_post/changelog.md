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
