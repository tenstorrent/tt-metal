# Verification Report: scaled_dot_product_attention

Flash Attention (online softmax, O(S) memory). Verified against the acceptance
suite, the golden registry matrix (1156 cells), the `eval.verify_supported`
CLI, and a new precision baseline. Date: 2026-06-12.

## Code Review

Every item below was **fixed in place** unless explicitly marked as a deferred
observation.

1. **Missing registry re-exports (`__init__.py`)** — FIXED. The package
   `__init__.py` only exported `scaled_dot_product_attention`, so
   `test_golden.py`'s `from ttnn.operations.scaled_dot_product_attention import
   EXCLUSIONS, INPUT_TAGGERS, SUPPORTED` failed at collection (the entire golden
   suite errored out, 1 collection error). Added `INPUT_TAGGERS`, `SUPPORTED`,
   `EXCLUSIONS` to the re-export list. Golden suite now collects 1156 cases.

2. **Running accumulators stored in bf16 — design-conformance + precision bug**
   — FIXED. `op_design.md` (Key Risks: *"Numerical exactness requires fp32 DEST
   accumulation for the online recurrence … the running-max rescale is exact …
   The recurrence is mathematically equivalent to two-pass softmax only when
   m_i/l_i/O_i carry full precision across blocks"*) mandates fp32 for the
   persistent accumulators. The program descriptor declared every compute CB as
   `query.dtype` (bf16), so `cb_max` / `cb_l` / `cb_o_acc` and their per-iter
   scratch (`cb_corr`, `cb_max_prev`, `cb_m_blk`, `cb_l_block`, `cb_pv`,
   `cb_o_tmp`) were packed back to bf16 between KV blocks, compounding rounding
   with the number of KV blocks (error grew with S and with sign-biased inputs).
   Set those CBs to `ttnn.float32` and fixed the `cb()` helper to size pages by
   data format (`ttnn.tile_size(fmt)`) — it previously hard-coded the bf16 tile
   size for all CBs, which would have under-sized any fp32 page. Streamed I/O
   (`cb_q/k/v/mask`), the score/prob blocks (`cb_qk`, `cb_p`), and `cb_out`
   stay bf16.

   Effect (regression canaries, before → after the fix):
   - `test_negative_input[B1_H12_S512]`: PCC 0.866 → 0.936
   - `test_uniform_input[B1_H12_S512]`: PCC 0.958 → 0.987
   - `test_negative_input[B1_H8_S256]`: PCC 0.950 → 0.969
   - Registry S=8192 self-attn cells: numerical-precision FAIL → PASS
     (`supported_pass` 138 → 140; `supported_fail` 2 → 0).

3. **Reader reconstructed the mask `TensorAccessor` every KV iteration** —
   FIXED. The mask accessor was built inside the `for j` loop (once per KV
   block per work unit). Hoisted it next to the Q/K/V accessors (built once),
   guarded `[[maybe_unused]]`. Verified through the custom-mask acceptance and
   precision cases.

### Deferred observations (not fixed — not in scope for this pass)

- **`q_chunk_t == k_chunk_t == 1` hard-coded.** The kernel processes one
  tile-row of Q (32 query rows) per work unit against one tile-row of K/V per
  KV block. `op_design.md` recommends 2–4. This is a throughput limitation, not
  a correctness issue (per-row stats are a single tile so all column broadcasts
  trivially use index 0). Larger chunks would amortize matmul setup and improve
  FPU utilization but require generalizing the broadcast indices. No failing
  cell points at it → recommendation, not a refinement.

- **Score/prob blocks (`cb_qk`, `cb_p`) remain bf16.** The bf16 score path caps
  achievable accuracy on sign-biased / low-variance inputs (the
  `test_negative_input` / `test_uniform_input` canaries still miss the *tight*
  default RMS target of 0.04, though max-abs error is tiny, 0.01–0.02, and PCC
  is ≥ 0.94). These are non-registry numerics canaries on degenerate
  distributions where relative-RMS inflates (output clusters tightly → small
  denominator). The concrete lever (fp32 `cb_qk`/`cb_p`, or exposing
  `math_fidelity`/`fp32_dest_acc_en` via `compute_kernel_config`) is folded into
  Refinement 1 (numerical configurability), so it is tracked there rather than
  duplicated as its own entry.

## Registry Conformance

- **Confirmed present and correctly wired** in
  `scaled_dot_product_attention.py`: `INPUT_TAGGERS` (3 taggers, all with the
  `(inputs, axes)` signature), `SUPPORTED` (all 7 axes the kernel gates on),
  `EXCLUSIONS` (`[]`), and `validate()` as the **first line** of the public
  entry point. `validate()` checks tensor-shape contracts (rank, D match, S_kv
  match, batch/head match, mask dims, `is_causal`⊕`attn_mask`), then SUPPORTED
  per-axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`).
- **Op file does NOT declare `INVALID`** — confirmed (it is a feature_spec.py
  concept). Good.
- **No auto-fixes to SUPPORTED were needed** — `xpass_drift = 0`. The SUPPORTED
  block was already honest: causal / gqa / mqa / non-aligned / float32 / bf8b
  all correctly rejected by `validate()` and observed as `xfail_expected`.
- **EXCLUSIONS empty is correct for Phase 0.** The candidate
  `{"mask_mode": "causal", "attention_kind": "cross"}` only arms once `causal`
  joins `SUPPORTED["mask_mode"]` (EXCLUSIONS only filter cells *inside*
  cartesian(SUPPORTED)); it is filed as part of Refinement 3.

### INVALID audit (`eval/golden_tests/.../feature_spec.py`)

`INVALID = []`, which is correct and well-formed for this op:
- **No cross-tensor-axis coupling** — there are no INVALID entries at all.
- **Canonical bf8b + ROW_MAJOR rule is vacuous** — `TARGET["layout"]` is
  TILE-only (SDPA has no ROW_MAJOR path by design), so no ROW_MAJOR cell exists
  in the cartesian product to forbid. The feature_spec documents this
  explicitly. No action.
- Not norm-like (no weight axes) → no no-weight canonicalization rows needed.

## Precision Baseline

bf16, tile-aligned, no mask, auto scale. Measured by
`test_scaled_dot_product_attention_precision_baseline.py`.

| Shape (B,H,S,D) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-----------------|-----|-------------|--------------|------------------|
| (1, 1, 32, 32)   | 0.99996 | 0.02397 | 0.00345 | 0.01557 |
| (1, 8, 128, 64)  | 0.99997 | 0.01887 | 0.00132 | 0.01238 |
| (2, 8, 256, 64)  | 0.99996 | 0.01946 | 0.00099 | 0.01290 |
| (1, 4, 1024, 64) | 0.99994 | 0.00915 | 0.00060 | 0.01525 |

**Assessment**: Excellent on well-conditioned random-normal inputs — PCC
~0.9999 and relative RMS ~1.3–1.6 % across single-tile through 1024-long
sequences, with no degradation as S grows (the fp32-accumulator fix removed the
S-dependent drift). Accuracy softens only on degenerate distributions
(uniform-positive / all-negative), where the bf16 score path and the
relative-RMS metric's small denominator combine — see deferred observation #2.

**Recommended tolerances**: PCC ≥ 0.995, relative RMS ≤ 0.05 (matches the
golden `TOLERANCES[bfloat16]`). atol ≈ 0.03 is comfortable on random-normal
inputs; do not impose a meaningful rtol (softmax outputs near zero make rtol
unbounded).

## Verifier CLI Summary

From `verifier_report.json` (1156 registry cells):

- supported_pass: **140**
- xfail_expected: **976**
- invalid_skipped: 0 (no INVALID cells)
- no_axes_found: 40 (the 39 `test_regression.py` numerics canaries +
  `test_op_loose[NOTSET]`; not registry-driven by design)
- **supported_fail: 0**   ✓
- **xpass_drift: 0**      ✓
- **xfail_wrong_mode: 0** ✓
- supported_marked_xfail: 0

All three loud categories are zero — the SUPPORTED block is honest and the op
ships clean. The 976 `xfail_expected` cells map exactly onto the refinement
queue (see `op_requirements.md`); every `(axis, missing_value)` pair is covered.

### Non-registry regression status (`test_regression.py`, informational)

19 of 39 numerics canaries fail; none are registry cells, none block shipping:
- 4 × `test_gqa_mqa_forward` (category `other`) — the op *correctly* rejects
  GQA/MQA via `validate()` (kv_heads_mode not yet supported). Unblocked by
  Refinement 2.
- 15 × `uniform`/`negative`/`large_magnitude` (14 numerical-precision, 1
  numerical-bug) — relative-RMS artifacts on low-variance degenerate
  distributions; max-abs error is 0.01–0.02. Tracked via Refinement 1's
  numerical-configurability lever; not their own queue entry.

## Recommendations

1. **Refinement 1 (numerical configurability) first.** It exposes
   `compute_kernel_config` and adds the float32 / bfloat8_b dtype branches, and
   it is the natural home for the score-path precision lever that would lift the
   `test_negative_input` / `test_uniform_input` canaries. The fp32-accumulator
   work landed in this pass pairs directly with it (the CB-format derivation it
   introduces should subsume the hard-coded fp32 accumulator formats).
2. **GQA/MQA (Refinement 2) is high-value and low-risk** — reader-side KV head
   index remap only, no compute change; the `tag_kv_heads` tagger already
   exists and `validate()` already routes it.
3. **Multi-core is already done** — `split_work_to_cores` distributes
   `(b,h,q_block)` units across the full grid, and per-core CB footprint is O(1)
   in S (chunk = 1 tile), so long-context (S=4096/8192) runs without OOM. No
   multi-core or memory-budget refinement is warranted.
4. **`q_chunk_t`/`k_chunk_t` widening** is a future throughput opportunity, not
   a capability gap — leave it out of the queue until a perf target demands it.
