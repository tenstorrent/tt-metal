# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-12
- **What was done**: Initial Flash Attention implementation via the incremental
  pipeline (planner → implementer → verifier). Online-softmax recurrence
  (running max / sum / output), O(S) memory — the full S_q × S_kv score matrix
  is never materialized. Multi-core via `split_work_to_cores` over
  `(b, h, q_block)` work units. `generic_op` + `ProgramDescriptor`; reader
  (NCRISC) / compute (TRISC) / writer (BRISC); helper-based compute
  (`matmul_block`, `reduce`, `eltwise_chain`, `binary_sfpu` BinaryMax).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned],
  attention_kind=[self, cross], kv_heads_mode=[mha], mask_mode=[none, custom],
  scale_mode=[auto, explicit]. EXCLUSIONS=[]. INVALID lives in feature_spec.py
  (`[]` — SDPA is TILE-only, so the bf8b+ROW_MAJOR rule is vacuous).
- **Accuracy achieved**: PCC ≈ 0.9999, max_abs_err ≈ 0.009–0.024,
  mean_abs_err ≈ 0.0006–0.0035, relative RMS ≈ 0.012–0.016 (measured on 4 shapes
  via `test_scaled_dot_product_attention_precision_baseline.py`; bf16,
  tile-aligned, random-normal inputs).
- **Golden suite at Phase 0**: 140 / 1156 cells passing; 976 xfail_expected;
  loud categories all 0 (supported_fail / xpass_drift / xfail_wrong_mode), per
  `verifier_report.json`.
- **Issues encountered (fixed during verification)**:
  - `__init__.py` did not re-export `INPUT_TAGGERS` / `SUPPORTED` / `EXCLUSIONS`,
    causing a golden-suite collection error. Added the re-exports.
  - Running accumulators (`m_i`, `l_i`, `O_i`) and their scratch were stored in
    bf16 CBs, violating the design's fp32-accumulation requirement and
    compounding rounding across KV blocks (error grew with S; S=8192 cells
    failed numerical-precision). Switched the accumulator/scratch CBs to
    `float32` and fixed the `cb()` page-size helper to size by data format.
    Result: `supported_pass` 138 → 140 (S=8192 now passes); regression-canary
    PCC improved (e.g. `test_negative_input[B1_H12_S512]` 0.866 → 0.936).
  - Reader rebuilt the mask `TensorAccessor` every KV iteration; hoisted it to
    construct-once alongside the Q/K/V accessors.
- **Tests added**: `test_scaled_dot_product_attention_precision_baseline.py`
  (PCC + abs/RMS across 4 shapes). The acceptance suite
  (`test_scaled_dot_product_attention.py`, 24 cases) and the golden suite were
  already present; both pass.
