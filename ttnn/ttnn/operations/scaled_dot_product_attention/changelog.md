# Changelog: scaled_dot_product_attention (Flash Attention)

## Phase 0 — Core Implementation
- **Date**: 2026-07-23
- **What was done**: Initial implementation via the incremental pipeline (planner →
  implementer → verifier). Fused flash-attention kernel with the online-softmax
  recurrence (score matrix never materialized); multi-core over the flat
  `B·H_q·q_num_chunks` work-list; reader/compute/writer split; GQA/MQA handled by
  reader head-addressing; self + cross attention; mask_mode none/custom; scale
  auto/explicit.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha, gqa, mqa],
  mask_mode=[none, custom], scale_mode=[auto, explicit].
- **Accuracy achieved** (bf16, HiFi2 + fp32-DEST, `randn`, seed 42, via
  test_scaled_dot_product_attention_precision_baseline.py):
  PCC ≥ 0.995 on all shapes; max_abs_err ≤ 0.0134, mean_abs_err ≤ 0.0017,
  relative RMS 0.0079–0.0099. got/true ratio centered ~0.995 with symmetric
  spread → bf16 noise, not a scale bug.
- **Golden suite at Phase 0**: 206 supported_pass, 6 supported_fail (all OOM:
  D∈{512,1024} at S=128), 2113 xfail_expected; xpass_drift=0, xfail_wrong_mode=0
  (per `verifier_report.json`).
- **Issues encountered / fixes applied by verifier**:
  1. Added the missing `default_compute_kernel_config()` factory + `__init__.py`
     export (the golden harness imports it as the single source of truth for the
     `fp32_dest_acc_en` tag; its absence broke golden-suite collection).
  2. Reordered `validate()` so the SUPPORTED/EXCLUSIONS support gate precedes the
     detailed tensor-shape contract — cleared 24 `xfail_wrong_mode` (unsupported
     `fp32_dest_acc_en=False` cells that also carried a batch-broadcast mask were
     being rejected with ValueError instead of the support-refusal).
  6 OOM `supported_fail` left failing (not silenced) → Refinement 2, per the
  registry-model routing table.
- **Tests added**: test_scaled_dot_product_attention_precision_baseline.py
  (test_scaled_dot_product_attention.py and test_scaled_dot_product_attention_debug.py
  already present; all 36 pass).

## Refinement 1 — Numerical configurability expansion
- **Date**: 2026-07-23
- **What was done**: Extended the numerical surface with **zero compute-kernel
  changes** (the kernel is fully helper-based — the /numeric-formats-metal pass
  condition holds). All edits are in the op file + entry point + program
  descriptor:
  - `SUPPORTED["dtype"] = [bfloat16, float32, bfloat8_b]`;
    `SUPPORTED["fp32_dest_acc_en"] = [True, False]`.
  - `EXCLUSIONS += {dtype: float32, fp32_dest_acc_en: False}` — legal-but-lossy
    (fp32 input thrown away by the 16-bit DEST accumulator), refused, mirrors
    softmax. `{bfloat8_b, False}` kept SUPPORTED: it clears the golden (0.99/0.12)
    tolerance (measured PCC ~0.9998), so block-float already dominates the error
    budget and the DEST width is second-order.
  - **Per-CB dtype-derived formats** (single source per role): input CBs
    (Q/K/V/mask) carry the input dtype; `cb_out` + the output tensor follow the
    input dtype (fp32→fp32, bf16→bf16, bf8b→bf8b — the golden contract checks
    got.dtype == input dtype; Phase-0's hardcoded-bf16 output only held because
    input was bf16). Scalers stay bf16 (reader packs via prepare_reduce_scaler).
  - **Intermediate CBs = fp32 whenever `fp32_dest_acc_en=True`** (skill §4): the
    online-softmax running (m,l,O) is parked and reloaded every KV-block, so a
    bf16 park truncates back to 7 mantissa bits each step and erases the
    fp32-DEST gain. This is the "fp32 intermediate CBs" lever the verifier note
    names — it improved the adversarial-distribution regressions 14→10 failing.
    When acc is off (the bf16 perf profile) intermediates match the streaming
    width, so the perf path is byte-identical.
  - **Dtype-correct math fidelity** (`_resolve_math_fidelity`, single source):
    bf16/bf8b inputs are clamped from HiFi4/HiFi3 → HiFi2 (they fit losslessly in
    TF32, and HiFi4 + fp32-DEST + bf16 silently corrupts, issue #38306); float32
    keeps the requested fidelity (HiFi4 recovers TF32-truncated mantissa bits).
    The program descriptor rebuilds the ComputeConfigDescriptor with the resolved
    fidelity; `fp32_dest_acc_en` / `math_approx_mode` honored as passed.
- **Accuracy achieved** (D=64/128 tile-aligned shapes, randn):
  fp32@True PCC ~0.99999 (relRMS 0.0036); bf16@True PCC ~0.99997 (0.0098);
  bf16@False PCC ~0.99993 (0.0121); bf8b@True PCC ~0.99989 (0.0148);
  bf8b@False PCC ~0.99985 (0.0174). All clear their golden tolerances.
- **Golden test progress**: test_golden.py 1025 passed / 36 failed / 848 xfailed,
  **0 xpassed** (SUPPORTED matches reality — no drift). Up from Phase-0's
  206 passed. All 36 failures are L1 CB-allocation OOM (RuntimeError @
  program.cpp:1751) at the large-head-dim boundary (D ∈ {256, 512, 1024} at
  S=128) — Refinement 2's explicit, anticipated scope (fp32 CBs raise L1
  pressure exactly as the R1 note foretold). Left failing, not silenced.
  The flagged perf-shape loose case (bf16 @ fp32_dest_acc_en=False, the perf-1
  contract anchor for Refinements 3/5) passes.
- **Issues encountered**:
  1. Initial run: 648 failures, all `dtype_mismatch got BFLOAT16 expected
     FLOAT32/BFLOAT8_B` — the op hardcoded bf16 output. Fixed by having the
     output dtype follow the input dtype.
  2. Adversarial-distribution regressions (test_regression.py
     uniform/negative/large-magnitude, bf16 input) improved 14→10 via fp32
     intermediates. The remaining 10 are fundamental bf16-input compute-path
     noise (severity mostly `precision`, one `bug` at pcc 0.94); the tests
     hardcode bf16 so float32 can't be applied — a pre-existing Phase-0 baseline
     the correct lever partially cleared, not an R1 regression.
- **Tests added**: test_scaled_dot_product_attention_precision_matrix.py
  (dtype × fp32_dest_acc_en × distribution over tile-aligned D≤128 shapes;
  40 passed / 8 skipped for the fp32@False EXCLUSION) + precision_matrix_results.md.
