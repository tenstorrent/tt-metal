# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-18
- **What was done**: Initial Flash-Attention (online-softmax) implementation
  via the incremental pipeline (planner → implementer → verifier). Fused
  reader/compute/writer kernels built on the helper library
  (`matmul_block`, `reduce<MAX/SUM,REDUCE_ROW>`, `eltwise_chain`,
  `add`/`mul`/`copy`/`unary`/`binary_sfpu`, `transform_in_place`). Score CBs
  sized to one `B_q × B_kv` block (the load-bearing Flash constraint — the
  full `S_q × S_kv` matrix is never materialized). Multi-core from the start
  via `split_work_to_cores` (one `(b, h_q, qb)` work item per stamp,
  interleaved DRAM, no inter-core communication). Host applies an L1-aware
  block cap (`B·DHt ≤ 16`) so large `D` and long sequences fit L1.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE],
  alignment=[tile_aligned], attention_kind=[self, cross],
  kv_heads_mode=[mha, gqa, mqa], mask_mode=[none, causal],
  scale_mode=[auto, explicit]. EXCLUSIONS=[]. (INVALID lives in
  feature_spec.py and is []; not declared in the op file.)
- **Accuracy achieved** (bf16, well-conditioned randn, fp32 reference,
  `test_scaled_dot_product_attention_precision_baseline.py`):
  PCC=0.99992–0.99996, max_abs_err≈0.012, mean_abs_err≈0.001,
  relative_rms_err≈0.009–0.012 across 4 shapes
  [(1,1,32,64), (1,4,128,64), (1,8,256,64), (2,4,512,64)].
- **Golden suite at Phase 0**: 207 / 208 supported cells passing (per
  `verifier_report.json`); 536 xfail_expected; xpass_drift=0;
  xfail_wrong_mode=0. The 1 red cell is `Q1x1x128x1024` bf16 explicit-scale,
  category `numerical-precision` (rel-RMS 0.0505 vs 0.05, PCC 0.9987) —
  queued as Refinement 1, not silenced.
- **Issues encountered**:
  - **Code-review fix**: the reader rebuilt the mask `TensorAccessor` (and
    re-called `get_tile_size`) inside the per-KV-block loop — hoisted to a
    single loop-invariant construction before the work loop. Acceptance
    34/34 still passes (mask + no-mask paths).
  - **Regression suite** (`test_regression.py`, numerics-tagged, not
    registry-gated): 15 failures on `negative`/`uniform`/`large` input
    distributions. Diagnosed as **metric conditioning, not algorithmic
    bugs** — all-same-sign V yields a near-constant softmax-averaged output
    (std≈0), so relative-RMS/PCC are ill-conditioned even though absolute
    error is at the bf16 floor (≤0.03). The genuine-precision subset
    (large ×10) is addressed by Refinement 1's higher-precision config.
    Golden-suite files were not modified (upstream-authored).
  - Three Phase-0 bring-up bugs were already found+fixed by the
    implementer/expert-debugger: UNPACK hang from in-place pre-scale of the
    reader-fed `cb_q_in` (scale moved to the locally-produced scores);
    `j>0` α-correction wrong for `B_q>1` (held operands switched to
    `OperandKind::Block`+`HeldBulk`); large-head-dim (D≥512) L1 overflow
    (L1-aware block cap).
- **Tests added**: `test_scaled_dot_product_attention_precision_baseline.py`
  (PCC + abs/RMS across 4 shapes). Existing:
  `test_scaled_dot_product_attention.py` (acceptance, 32 cases),
  `test_scaled_dot_product_attention_debug.py` (single-tile no-hang
  regression, 2 cases). Full op test suite: 34 passed.
