# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-19
- **What was done**: Initial Flash-Attention implementation via the incremental
  pipeline (planner → implementer → verifier). Tiled online softmax with O(S)
  memory — `cb_scores`/`cb_p` hold one `(1 × kv_chunk_t)` block; running m/l/O are
  the only KV-surviving state. Multi-core via `split_work_to_cores` over `B·H·Sq_t`
  units (embarrassingly parallel, no inter-core communication).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE_LAYOUT],
  alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha],
  mask_mode=[none, causal], scale_mode=[auto, explicit]. EXCLUSIONS=[].
- **Accuracy achieved** (bf16, randn, seed 0, via
  test_scaled_dot_product_attention_precision_baseline.py over 4 shapes):
  PCC=0.99997 / 0.99997 / 0.99997 / 0.99996; max_abs_err ≤ 0.0137;
  mean_abs_err ≤ 0.0016; relative RMS err 0.0101 → 0.0114 as S/H scale.
- **Golden suite at Phase 0**: **140 / 140** supported cells passing
  (per `verifier_report.json`: supported_pass=140, supported_fail=0,
  xpass_drift=0, xfail_wrong_mode=0; 604 cells correctly xfail toward TARGET;
  40 non-registry rows = test_regression.py numerics + empty loose placeholder).
- **Issues encountered**:
  - No drift, no code-review fixes required — the implementation is clean and
    idiomatic (helpers used throughout, CB push/pop balanced, TensorAccessor,
    correct includes, correct broadcast/reduce dims).
  - Three documented design deviations, all forced by real LLK constraints and
    mirrored by the production SDPA op (not bugs): (1) bf16 intermediate CBs with
    fp32 DEST accumulation instead of fp32 CBs (Issue #13364 `pack_reconfig` hang);
    (2) scale folded after the QK matmul instead of into Q (datacopy→matmul-AB
    hang); (3) `mm_init` boot instead of `compute_kernel_hw_startup`+`mm_block_init`.
  - Precision limitation (no in-scope lever, documented in verification_report.md,
    NOT a refinement): bf16-CB storage of the running softmax statistics makes error
    grow across the KV recurrence under adversarial distributions (uniform/negative
    inputs in test_regression.py reach rel RMS up to 0.30–0.67). The fp32-CB lever
    hangs the device; HiFi2 + fp32_dest_acc are already set. Revisit when Issue
    #13364 is resolved.
- **Tests added**: test_scaled_dot_product_attention_precision_baseline.py
  (PCC + abs/RMS baseline over 4 shapes). Acceptance test
  (test_scaled_dot_product_attention.py, 24 cases) and golden + regression suites
  pre-existed; all 24 acceptance cases pass.
- **Refinement queue** (op_requirements.md): R1 numerical configurability
  (float32 + bfloat8_b + compute_kernel_config, /numeric-formats-metal) →
  R2 GQA/MQA (kv_heads_mode, reader already wired) → R3 non-tile-aligned
  (alignment w/h, /memory-layouts). Covers every TARGET−SUPPORTED gap.
