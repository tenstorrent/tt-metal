# Changelog: genify_matmul

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Regime R1 `interleaved_out_split`:
  - an independent M×N output split over the full device grid;
  - per-core blocks solved coarsest-first against the L1 budget;
  - the K-walk accumulated in L1 by the `matmul_block` helper (packer L1 acc, ragged last K-block);
  - one `ttnn.generic_op` dispatch per call.
- **SUPPORTED at Phase 0**:

  | Axis | Values |
  |---|---|
  | dtype | bfloat16 |
  | weight_dtype | bfloat16 |
  | layout | TILE |
  | fp32_dest_acc_en | True |
  | placement | interleaved (DRAM or L1) |
  | epilogue | none |
  | alignment | tile_aligned, m_non_aligned |
  | weight_batch | single |

- **Accuracy achieved**: PCC ≥ 0.9999985, max_abs_err ≤ 0.018, relative RMS 0.0017, ULP p99 3–4.
  Measured on 5 shapes (single tile through 2048×4096×2048 and K = 16384) with
  `test_genify_matmul_precision_baseline.py`. Bit-identical across repeated calls.
- **Golden suite at Phase 0**:
  - 94 supported_pass, 2088 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode
    (`verifier_report.json`);
  - 1 supported_fail: a harness L1-capture false-negative on a correctly raised `ValueError`.
- **Issues encountered (fixed by the verifier)**:
  - **Grid factorization.** Ranking W strictly first produced DRAM-traffic-pathological splits:
    9472×5120×3456 took 28 ms with every core reading all of A. It is now `choose_split()`: within
    `GRID_W_SLACK` (0.25) of the best W, take the min of aggregate read traffic *including* the
    block re-reads. Measured 0.21–0.46× device time on the mis-split shapes and no regression beyond
    noise on the guard set; golden geomean 0.95×.
  - **L1 CB budget.** It ignored fragmentation (a freed L1 tensor above a live one), which gave a
    CB/L1-buffer clash TT_THROW. It now ends at the lowest allocated L1 block.
  - **DRY.** The duplicate `TILE` constant was removed.
- **Tests added**:
  - `test_genify_matmul_precision_baseline.py`;
  - `test_genify_matmul_extended.py`: output-dtype override, fragmented L1, L1 weight;
  - `perf/test_genify_matmul_perf_guard.py`: opt-in device-ns guard set (`GENIFY_MATMUL_PERF=1`).
  - `test_genify_matmul.py` (acceptance, 19 tests) already existed and is unchanged.
