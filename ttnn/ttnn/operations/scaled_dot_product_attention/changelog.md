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

## Refinement 1 — Numerical configurability + fp32 accumulator precision (partial)
- **Date**: 2026-06-16
- **What was done**:
  - `SUPPORTED["dtype"]` extended to `[bfloat16, float32, bfloat8_b]`.
  - `EXCLUSIONS` armed with `{"dtype": float32, "fp32_dest_acc_en": False}`
    (maxed input + non-maxed DEST acc refused; `test_exclusion_fp32_no_acc`
    verifies the `NotImplementedError`).
  - Program descriptor: input/output/mask CBs follow their own tensor dtype
    (already did); intermediate + online-softmax accumulator CBs
    (`cb_o/cb_l/cb_m/cb_pv/cb_o_resc/cb_scores/cb_p/cb_q/...`) now use
    `acc_format = float32 if fp32_dest_acc_en else bfloat16` — the running
    statistics stop re-rounding to bf16 every KV-chunk. `cb()` derives
    `page_size` from `data_format`.
  - **Default config change (the load-bearing fix)**: `compute_kernel_config=None`
    now maps to **HiFi4** (was HiFi2) + `fp32_dest_acc_en=True`. SDPA chains two
    matmuls whose operands unpack to TF32; under HiFi2 the matmul truncation
    dominates and long-context mask=none rows (near-uniform softmax → tiny
    output) miss the relative-RMS gate. Probe (S=2048 bf16): HiFi2 rel_rms=0.146
    → HiFi4 rel_rms=0.0145 (~10×). fp32 accumulator compounds it (HiFi4
    acc=True 0.0145 < acc=False 0.0203). Zero compute-kernel changes — helpers
    carry data-format reconfig; matmul_block default reconfig is INPUT_AND_OUTPUT.
- **Accuracy achieved** (randn, default HiFi4+fp32acc unless noted):
  - bf16: PCC 0.99997 @ S=2048/4096, rel_rms ~0.014.
  - fp32: PCC 0.9999 @ S≤2048; PCC 0.99990 / rel_rms 0.0289 @ S=4096.
  - bf8b: PCC 0.9999 @ (1,2,128,64).
  - Short/medium (128/256): 60/60 golden pass across all dtypes.
- **Golden test progress**: **674 passed** (Phase 0: 346), 1532 xfailed,
  26 failed. supported_pass nearly doubled. Remaining 26 failures (all at the
  precision/memory frontier, NONE excluded):
  - 4× **fp32 @ D=1024** (`Q1x1x128x1024`): L1 OOM (fp32 CBs scale with d_t).
    → **Refinement 4** (/memory-budget-metal).
  - 6× **fp32 acc=True @ S≥4096**: rel_rms 0.028–0.053 vs tight 0.02 (TF32
    matmul floor; HiFi4 is max). → **Refinement 5**.
  - 8× **bf16 acc=False** + 8× **bf8b acc=False @ S≥4096**: rel_rms 0.15–0.77
    vs 0.12. **Proven 16-bit DEST floor** (probe: fp32-CB-always → identical
    rms; golden pins acc=False+HiFi2). Only lever is algorithmic Bkv blocking.
    → **Refinement 5**.
- **Issues encountered**: the verifier hypothesis (fp32 accumulator alone
  clears all 20 long-context supported_fail) was partially incorrect — at HiFi2
  the accumulator change is swamped by (and slightly worsens, 0.146 vs 0.096)
  the TF32 matmul error. The actual lever for the acc=True cells is HiFi4. The
  acc=False long-context cells remain a 16-bit-DEST hardware floor the golden's
  pinned config makes unreachable within R1's descriptor-level scope.
- **Why partial ([~])**: dtype axis (float32+bf8b) + EXCLUSION fully landed and
  the acc=True bf16 long-context cells (bulk of the original 20) now pass; the
  acc=False long-context cells and the fp32 OOM/precision-frontier cells are
  characterized at depth (probe data above) and filed as R4 + R5.
- **Tests added**: `test_scaled_dot_product_attention_precision_matrix.py`
  (dtype × fp32_acc × math_fidelity × shapes, incl. `test_exclusion_fp32_no_acc`
  and `test_bfloat8_b_supported`); `precision_matrix_results.md`.
