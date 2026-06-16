# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-16
- **What was done**: Initial FlashAttention (online-softmax) fused op via the
  incremental pipeline (planner → implementer → verifier). Per-work-unit
  `(b, h, q-chunk)` recurrence with `Bq_t = Bkv_t = 1` blocking; the
  S_q×S_kv score matrix is never materialized. Work units split
  embarrassingly-parallel across the compute grid via
  `split_work_to_cores`. Reader/compute/writer kernels; compute uses the
  kernel_lib helpers (matmul_block, reduce, eltwise_chain, binary_sfpu, …).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False],
  layout=[TILE], alignment=[tile_aligned], attention_kind=[self, cross],
  kv_heads_mode=[mha], mask_mode=[none, custom], scale_mode=[auto, explicit].
  EXCLUSIONS=[].
- **Accuracy achieved**: PCC ≥ 0.99992, max_abs ≤ 0.0146, rel_rms ≤ 0.018
  (4 shapes, bf16, via `test_scaled_dot_product_attention_precision_baseline.py`).
- **Golden suite at Phase 0**: 346/2767 passed, 0 hangs; verifier
  supported_pass=328, xfail_expected=2274, supported_fail=20 (all
  numerical-precision, long-context bf16 accumulator — queued as R1),
  xpass_drift=0, xfail_wrong_mode=0 (per `verifier_report.json`).
- **Issues encountered (verifier fixes)**:
  - **Fixed a per-unit `cb_m` deadlock** (hang on any shape with > 64
    work-units, i.e. `num_units > 1`/core): the running-max CB was never
    drained after the KV loop, blocking the next unit's init. Added
    `cb_pop_front(cb_m, 1)` at unit end. This was the root cause of the
    golden-suite hang cascade.
  - **Added batch-broadcast attention-mask support**: `validate()` now
    accepts `mask.shape[0] ∈ {1, B}` (and `shape[1] ∈ {1, H_q}`); the program
    descriptor threads `mask_B`; the reader collapses the batch index. Cleared
    10 supported_fail + 16 xfail_wrong_mode.
  - **Migrated** `noc_async_read_tile`/`noc_async_write_tile` →
    `noc_async_read_page`/`noc_async_write_page` (deprecation cleanup).
- **Tests added**: `test_scaled_dot_product_attention_precision_baseline.py`
  (PCC + abs/RMS baseline over 4 shapes). Acceptance suite
  `test_scaled_dot_product_attention.py` (24 cases) passes.
