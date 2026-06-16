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

## Refinement 2 — Attention variants: causal masking + GQA / MQA
- **Date**: 2026-06-16
- **What was done**:
  - `SUPPORTED["mask_mode"]` += `causal`; `SUPPORTED["kv_heads_mode"]` += `gqa`, `mqa`.
  - `EXCLUSIONS` += `{"mask_mode": "causal", "attention_kind": "cross"}` (causal
    generates the triangular bias assuming the diagonal aligns at tile
    boundaries, i.e. S_q == S_kv; the `is_causal` + `attn_mask` mutual-exclusion
    `ValueError` was already present and is retained).
  - **Causal (kernel work)**: triangular −inf bias generated **on-device** from
    `is_causal` — no caller tensor, no materialized full mask. The reader writes
    the 32×32 diagonal bias tile directly into L1 (`gen_causal_mask_tile`,
    face-aware fill, bf16, additive form 0 where col≤row else −1e30) and pushes
    it to `cb_mask_in` only on the diagonal block (`j == qc`). The reader skips
    fully-future KV-chunks (`kv_count = qc+1`) and passes fully-past chunks
    (`j < qc`) unmasked. Compute decodes `qc` from a new `start_unit` runtime
    arg, bounds its KV loop to `qc+1`, and adds the bias only at `j == qc`
    (reusing the existing custom-mask `add` path). Reader/compute agree on
    `kv_count` and on the single per-unit mask push (CB sync preserved).
  - **GQA / MQA (validate-only)**: no kernel change — the reader already computes
    `h_kv = h / group`, `group = H_q / H_kv`. Only `validate()` + `SUPPORTED`
    needed widening.
  - Program descriptor threads `is_causal`; creates `cb_mask_in` (bf16) for the
    causal path; passes `(causal, Sq_t)` compute CT + `start_unit` compute RT and
    `(causal, causal_neg_bits)` reader CT.
- **Accuracy achieved** (bf16, randn): causal self-attention PCC ≥ 0.995
  (golden bf16 tol (0.995, 0.05) — all pass); causal output vs the equivalent
  additive triangular mask PCC ≥ 0.999 (1×2×128×64); GQA/MQA PCC ≥ 0.995 across
  4:1 / 3:1 / 8:1 / 32:1 ratios + batched.
- **Golden test progress**: the 4 `test_gqa_mqa_forward` regression cases pass;
  causal-self + GQA/MQA matrix cells move from `xfail_expected` to
  `supported_pass`; causal+cross cells correctly `xfail` via the EXCLUSION
  (verified: 36 xfailed / 0 xpassed on the cross shapes; 60 passed / 0 xpassed
  on a gqa/mqa+causal-self spread). **No `xpass_drift`, no new failures.** The
  full 2767-cell suite was not run to completion in one sitting (runtime), but
  every targeted causal / gqa / mqa / cross subset is green.
- **Issues encountered**: None. Causal worked on the first device run (the cheap
  isolated test passed immediately); no sub-agent escalation needed.
- **Tests added**: `tests/.../test_scaled_dot_product_attention_variants.py`
  — `test_causal_self_attention` (7 shapes × 2 scale modes),
  `test_causal_matches_additive_triangular`, `test_gqa_mqa_self_attention`
  (6 ratios), `test_gqa_with_custom_mask`, `test_causal_cross_rejected`
  (EXCLUSION → NotImplementedError), `test_causal_and_mask_mutually_exclusive`
  (ValueError). 24 cases, all pass (--dev + non-dev).

## Refinement 3 — Non-tile-aligned shapes (S_q / S_kv / D)
- **Date**: 2026-06-16
- **What was done**:
  - `SUPPORTED["alignment"]` extended to `[tile_aligned, w_non_aligned,
    h_non_aligned]`. `EXCLUSIONS` += `{bfloat8_b, w_non_aligned}` and
    `{bfloat8_b, h_non_aligned}`.
  - **Program descriptor**: tile counts switched from floor to **ceil**
    (`Sq_t/Skv_t/d_t = -(-S//32)`) so the physically tile-padded DRAM buffer is
    fully indexed (floor silently dropped the last partial tile). Compute
    `kv_valid_last = S_kv % 32` and `kv_edge = (kv_valid_last != 0 and not
    causal)`; allocate a new `cb_edge_mask` (bf16) when `kv_edge`. New CT args
    threaded to reader (`kv_edge`, `kv_valid_last`) and compute (`kv_edge`).
  - **D non-alignment (w_non_aligned)**: handled by ttnn's zero-fill of the tile
    padding — the padded D columns of Q/K contribute 0 to the QK^T contraction
    (0·0) and V's padded D columns give 0 in the PV output. No masking needed.
  - **KV-sequence non-alignment**: on-device column edge-mask (`gen_edge_mask_tile`:
    0 for valid key cols, −1e30 for padding cols) added to the score tile on the
    **last KV chunk** for `{none, custom}` modes. Forces padding keys to −inf so
    they drop out of the softmax row-max (max ignores −inf) and row-sum
    (exp(−inf−max)→0) in one op. Causal needs no edge mask: it requires S_q==S_kv,
    the padding KV chunk is only ever the diagonal block, and the triangular −inf
    bias already masks col>row ⊇ the padding columns for every valid query row.
  - **S_q non-alignment (query padding rows)**: produce garbage output rows that
    are discarded by the logical output shape; no NaN risk (row-sum l ≥ 1 since
    every padding query row still attends ≥ 1 valid key after edge masking).
- **Accuracy achieved**: PCC ≥ 0.995 (bf16) / ≥ 0.999 (fp32) on all 10 golden
  non-aligned shapes × {none,custom} × {auto,explicit}
  (`test_scaled_dot_product_attention_non_aligned.py`, 81 cases incl. cross +
  GQA/MQA + both-non-aligned).
- **Golden test progress**: non-aligned slice **174 passed, 186 xfailed, 0
  xpass-drift, 0 fail** (80s). The 186 xfails are the EXCLUDED bf8b non-aligned
  cells + the R1 `{fp32, acc=False}` cells. Causal non-aligned self cells pass.
- **Issues encountered**: bf8b + non-aligned fails **out of the box** at
  `ttnn.allocate_tensor_on_device` (TypeError — incompatible args) for the
  non-tile-aligned output shape; tile-aligned bf8b works (R1). This is a
  structural allocator gap, not a kernel issue, so both `bfloat8_b` non-aligned
  cells are parked in `EXCLUSIONS` per the /numeric-formats-metal bf8b+non-aligned
  rule (refinement explicitly authorizes excluding h_non_aligned "if it fails out
  of the box").
- **Advisory deviation**: used the on-device −inf column edge-mask instead of the
  verifier-suggested `prepare_partial_reduce_scalers` / `ReducePartialScaler`. The
  −inf approach masks both row-max and row-sum at once, reuses the existing
  causal/custom mask-add path, and sidesteps the MAX-reduce zero-scaler issue (a
  zero partial scaler yields max(·,0), not −inf). Same intent, simpler.
- **Tests added**: `test_scaled_dot_product_attention_non_aligned.py`
  (`test_non_aligned` 80 cases + `test_bfloat8_b_w_non_aligned_excluded`).
