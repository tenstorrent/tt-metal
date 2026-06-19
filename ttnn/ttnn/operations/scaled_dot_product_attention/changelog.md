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

## Refinement 1 — Numerical configurability (float32 + bfloat8_b + compute_kernel_config)
- **Date**: 2026-06-19
- **What was done** (descriptor-level only — zero compute-kernel changes; the
  helper-based kernel carried all data-format reconfig automatically):
  - `SUPPORTED["dtype"]` += `ttnn.float32`, `ttnn.bfloat8_b`.
  - Public entry point gained `compute_kernel_config: ttnn.ComputeKernelConfig = None`.
    Resolved via `init_device_compute_kernel_config(arch, cfg, …)` — `None` reproduces
    the Phase-0 defaults byte-for-byte (HiFi2, fp32_dest_acc_en=True, approx=False,
    full_sync=False); a user config overrides per-field.
  - Output dtype now follows `Q.dtype` (was hard-coded bf16) — satisfies the golden
    `output.dtype == input.dtype` contract for fp32/bf8b.
  - Program descriptor: input-side CBs (cb_q/k/v/mask) and the output CB (cb_out)
    follow the tensor dtype (`ttnn.tile_size(dtype)`); **all** matmul/reduce/eltwise
    intermediate CBs stay bf16 with fp32-DEST accumulation (Issue #13364 — fp32 CB
    storage hangs this LLK). fp32 inputs unpack through srcA/srcB (→ TF32) for the
    matmuls — production SDPA behavior. ComputeConfigDescriptor now driven by the
    resolved config. No `UnpackToDestFp32` tagging (intermediates are bf16 and feed
    FPU matmul/reduce — tag would be incompatible).
- **Accuracy achieved** (test_..._numerical.py precision matrix, randn, 5 shapes
  32×32…512×64/128×128 × {bf16,fp32,bf8b} × {HiFi4,HiFi2,LoFi} × fp32_acc{T,F},
  90/90 pass):
  - bf16:  PCC ≥ 0.99992 (HiFi4/2), ≥ 0.99896 (LoFi); rel-RMS 0.5–1.0% (HiFi), 6–8% (LoFi).
  - fp32:  PCC ≥ 0.99989 (HiFi4/2), ≥ 0.99887 (LoFi); rel-RMS ~same as bf16 (inputs
    unpack to TF32, so fp32 storage adds range/contract, not mantissa precision).
  - bf8b:  PCC ≥ 0.99980 (HiFi4/2), ≥ 0.99905 (LoFi); rel-RMS 1.7–2.2%.
  - bf8b + causal additive −inf mask: PCC 0.9999 (the flagged sharp edge — passes,
    no EXCLUSIONS needed). Full table in tests/.../precision_matrix_results.md.
- **Golden test progress**: **414 / 420** supported cells passing (Phase-0 was
  140/140 bf16; 324 cells correctly xfail toward R2/R3; 1 skipped loose placeholder).
  - `bfloat8_b`: **fully landed** — every tile-aligned/MHA bf8b cell passes.
  - `float32`: all pass **except 6 corners**, both left failing (NOT silenced in
    EXCLUSIONS, per the OOM / precision-near-miss protocol):
    1. `float32 + Q1x1x128x1024` (D=1024, 4 cells) — **hard L1 OOM**: static CBs
       grow to 2,767,616 B > 1,572,864 B because fp32 input CBs double the footprint
       (cb_k/cb_v ≈ 1 MB each at Dt=32). **Lever proven** (temp `kv_chunk_t=1` →
       all 4 run+pass). Deferred to **R4** (footprint-aware kv_chunk_t,
       /memory-budget-metal) rather than hardcoding an L1 heuristic into a
       numeric-formats refinement.
    2. `float32 + Q1x1x8192x64 + no-mask` (2 cells) — **precision near-miss**:
       rms=0.0206 > 0.02 target (PCC=0.9999). bf16-CB softmax-accumulator limit over
       2048 KV chunks (Issue #13364; fp32 CBs hang). Lever-less for the default-config
       golden — tracked in verification_report.md, not a refinement. (Causal S=8192
       fp32 passes — masking halves the accumulation depth.)
- **Issues encountered**: the 6 float32 corners above. The bf8b+causal and
  bf8b+non_aligned EXCLUSIONS candidates flagged in the R1 plan did **not**
  materialize (bf8b+causal passes; bf8b+non_aligned isn't reachable until R3 adds
  the alignment axis). EXCLUSIONS stays `[]`.
- **Tests added**:
  - tests/.../test_scaled_dot_product_attention_numerical.py — test_dtype_mask_scale
    (48: dtype×mask×scale×attn-kind, incl. bf8b+causal), test_compute_kernel_config_honored
    (18: config-knob plumbing), test_scaled_dot_product_attention_precision_matrix (90).
  - tests/.../test_sdpa_dtype_probe.py — minimal isolated dtype smoke test.
  - tests/.../precision_matrix_results.md — characterization table + observations.
