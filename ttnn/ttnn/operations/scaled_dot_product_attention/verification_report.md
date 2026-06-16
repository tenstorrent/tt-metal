# Verification Report: scaled_dot_product_attention

FlashAttention (online-softmax) fused op. Phase 0 surface: bf16 / TILE /
tile-aligned / MHA / mask ∈ {none, custom} / scale ∈ {auto, explicit},
single-core-per-work-unit with embarrassingly-parallel work split across the
grid. Reviewed against `op_design.md` and the prompt; tested via the
acceptance suite, the golden suite + `eval.verify_supported`, and a precision
baseline.

## Code Review

### Fixed

1. **Per-unit `cb_m` deadlock (correctness, hang).** The running-max CB
   (`cb_m`) was pushed at init (phase 0b) and re-pushed every KV iteration
   (phase K, `copy cb_m_new → cb_m`), and popped only in phase G — leaving one
   stale tile resident after the KV loop. `cb_m` is sized to a single page and
   is **not** consumed after the loop (finalize uses `cb_l`/`cb_o` only), so
   the next work-unit's init `cb_reserve_back(cb_m, 1)` blocked forever.
   - Symptom: every shape with **> 64 work-units** (i.e. `num_units > 1` per
     core) hung. The acceptance suite never exercised this (all its shapes
     have ≤ 64 work-units = 1/core), so it passed while the golden suite hung
     at `(1,12,512,64)` (192 units) and cascaded device timeouts.
   - Fix: drain `cb_m` at unit end alongside the retained-Q pop
     (`cb_pop_front(cb_m, 1)` in `..._compute.cpp`). Verified: `(1,12,512,64)`
     and `(1,8,1024,128)` now pass at PCC 0.9999 in < 1 s; long-context
     `(1,1,4096,64)` / `(1,1,8192,64)` complete in < 2 s.

2. **Batch-broadcast attention masks rejected (capability gap).** `validate()`
   only accepted masks with `shape[0] == B`, rejecting the standard
   batch-broadcast form `(1, …, S_q, S_kv)`. The reference broadcasts the mask
   over batch; rejecting it is an artificial limitation that surfaced as 10
   `supported_fail` (bf16) + 16 `xfail_wrong_mode` (bf8b, rejected with
   `ValueError` *before* the dtype `SUPPORTED` check).
   - Fix: `validate()` now accepts `mask.shape[0] ∈ {1, B}` and
     `shape[1] ∈ {1, H_q}` with trailing `(S_q, S_kv)`; the program descriptor
     threads `mask_B`; the reader collapses the batch index
     (`mask_b = (mask_B == 1) ? 0 : b`). Verified all four broadcast variants
     (batch×head, batch-only, head-only, none) at PCC 0.99996.

3. **Deprecated NoC tile API.** `noc_async_read_tile` / `noc_async_write_tile`
   are deprecated (compiler `-Wdeprecated-declarations`). Migrated reader and
   writer to `noc_async_read_page` / `noc_async_write_page` (the non-deprecated
   `TensorAccessor` overload). No behavior change.

### Advisory (not blocking, no fix applied)

- **Single-tile blocking (`Bq_t = Bkv_t = 1`).** The design recommended
  `Bq_t = Bkv_t = 2` for amortizing per-tile helper overhead; the
  implementation uses 1×1 blocks. Correct, but the per-tile overhead makes
  long-context shapes comparatively slow (still < 2 s for S=8192). A
  performance tuning item, not a correctness one — no failing cell points at
  it, so it is **not** a refinement.
- **Hard-coded intermediate-CB dtype.** Scaler / running-stat / score CBs are
  created as `ttnn.bfloat16` regardless of input dtype. This is correct for
  Phase 0 (bf16-only) but is the precise seam the dtype refinement (R1) must
  open — see `op_requirements.md`.

## Registry Conformance

- **INPUT_TAGGERS** present and correctly signed `(inputs, axes)`:
  `tag_alignment`, `tag_attention_kind`, `tag_kv_heads`. ✓
- **SUPPORTED** declares every gated axis (dtype, fp32_dest_acc_en, layout,
  alignment, attention_kind, kv_heads_mode, mask_mode, scale_mode). ✓
- **EXCLUSIONS** = `[]` at Phase 0 (none armed yet). ✓
- **validate()** checks SUPPORTED per-axis then EXCLUSIONS (correct order),
  raising `UnsupportedAxisValue` / `ExcludedCell` from
  `ttnn.operations._op_contract`; shape-contract violations raise `ValueError`.
  The public entry point calls `validate()` as its first line. ✓
- **No INVALID symbol in the op file.** Confirmed — `INVALID` lives only in
  `feature_spec.py`. ✓
- **No auto-fix drift** needed: `xpass_drift = 0` (SUPPORTED does not
  under-claim). The bf16-accumulator long-context failures are genuine
  precision misses inside the claimed surface, queued as a refinement (not
  silenced).

### INVALID audit (`feature_spec.py`)

`INVALID = []`. Well-formed against the three sanity rules:
- SDPA's TARGET `layout` is TILE-only (no ROW_MAJOR), so the canonical
  `{bf8b, ROW_MAJOR}` activation cell is **vacuous** — correctly absent.
- No cross-tensor-axis coupling entries.
- No "kernel doesn't support yet" entries masquerading as structural
  impossibilities (those belong in EXCLUSIONS).
- Not a norm-like op — no no-weight canonicalization cells required.
No changes requested.

## Precision Baseline

bf16 / TILE / MHA / no-mask / auto-scale, seed 0, fp32 reference. Measured by
`test_scaled_dot_product_attention_precision_baseline.py`.

| Shape (B,H,S,D) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-----------------|-----|-------------|--------------|------------------|
| (1,1,32,32)     | 0.999978 | 0.01190 | 0.00135 | 0.00832 |
| (1,4,128,64)    | 0.999965 | 0.01120 | 0.00091 | 0.00910 |
| (2,8,256,64)    | 0.999954 | 0.01456 | 0.00074 | 0.00988 |
| (1,8,512,128)   | 0.999922 | 0.01170 | 0.00093 | 0.01772 |

**Assessment**: Excellent for short/medium sequences (PCC ≥ 0.9999, rel-RMS
≤ 0.018). Relative RMS grows with sequence length because the online-softmax
running statistics (`cb_o`, `cb_l`, `cb_m`) are stored in **bfloat16**, so each
KV-chunk's contribution re-rounds; over hundreds of chunks (S ≥ 2048) the
error compounds past the golden RMS bound. This is the single root cause of
all 20 remaining `supported_fail` cells and of the data-distribution
regression misses (uniform/negative inputs: PCC 0.999 but RMS 0.072 > tight
0.04 target). The lever is fp32 intermediate/accumulator CBs — queued as R1.

**Recommended tolerances** (short/medium): PCC ≥ 0.999, rtol = 0.05,
atol = 0.05. Long-context (S ≥ 2048) should use the looser golden bounds until
R1 lands.

## Verifier CLI Summary

Golden suite: 346/2767 passed, 0 hangs (run `/tmp/sdpa_results`, artifact
`verifier_report.json`).

- supported_pass: 328
- xfail_expected: 2274
- invalid_skipped: 0 (INVALID is empty)
- **supported_fail: 20** — all `numerical-precision`: long-context bf16
  accumulator (S ∈ {2048, 4096, 8192}). Queued as **R1**; per the registry
  model these legitimately remain failing until the precision refinement lands
  (the PCC/RMS *is* the signal — not silenced via EXCLUSIONS).
- **xpass_drift: 0** ✓
- **xfail_wrong_mode: 0** ✓ (was 16 before the batch-broadcast-mask fix)
- no_axes_found: 145 — the non-registry `test_regression.py` /
  `test_translated.py` cases (106 skipped nightly, 18 passed, 21 failed). The
  21 failures are: `test_gqa_mqa_forward` (×4 — GQA/MQA not yet in SUPPORTED,
  unblocks with R2) and the distribution tests (`uniform`/`negative`/
  `large_magnitude`/`long_context`, severity=**precision**, same
  bf16-accumulator ceiling as R1). None are numerical-bugs (no inf/NaN, all
  PCC > 0.99).

## Recommendations

- **R1 (numerical) is the highest-value refinement**: fp32
  intermediate/accumulator CBs both expand the dtype surface (float32,
  bfloat8_b) *and* clear all 20 long-context `supported_fail` cells plus the
  distribution-test precision misses. Land it first.
- **GQA/MQA already works at the kernel level** — the reader's `h_kv = h/group`
  path was verified correct (PCC 0.99997 / 0.99996 by temporarily widening
  SUPPORTED). R2's GQA/MQA portion is essentially a `validate()` + SUPPORTED
  change with no kernel work; bundled with causal masking to avoid an
  artificially tiny refinement.
- **Performance**: consider `Bq_t/Bkv_t = 2` blocking for long context as a
  follow-up tuning pass (no failing cell depends on it; tracked here, not in
  the queue).
