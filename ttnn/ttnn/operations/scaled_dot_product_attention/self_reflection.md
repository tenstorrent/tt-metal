# Self-Reflection: scaled_dot_product_attention

_Advisory only. Nothing here is auto-applied — every item is a proposal a human ratifies._

## Summary

Final blind pass: **golden matrix clean** (test_golden 1511 passed / 398 xfail / **0 failed**),
but **68 failures in the reference-derived suites** — translated 54 fail / 217 pass, regression
14 fail / 25 pass. The failures cluster on **three facets the golden axis model cannot see**:
(1) batch-broadcast attn_mask (24), (2) `generic_op` not registering in the device program cache
(22), (3) non-centered input distributions + long-seq precision drift (14 regression + 8
translated). `verify_supported` reports 54 `supported_fail`, but only ~24 are genuine
correctness over-claims — the rest are a caching-capability gap and marginal precision. The
problems are **framework/test-universe-level** (the tagger set is blind to these facets), not
kernel-logic bugs — the kernel is numerically correct on everything golden exercises.

## 1. Golden coverage

**Finding 1 — attn_mask batch-broadcast is an axis-blind gap (24 translated fails).**
- **What**: All `test_sdpa_noncausal_mask*` failures pass a mask with batch dim `1` while `B>1`
  (`bcast_mask_batch_dim=True`). The op rejects it in `validate()`; golden never exercises it
  because the harness hardcodes the mask shape.
- **Evidence**: `helpers.py:242` `torch_mask = make_causal_mask(B, S_q, S_kv, ...)` → always
  `(B,1,S,S)` (`helpers.py:143`); op rejects `m_b != b_q` at
  `scaled_dot_product_attention.py:207`. Failures e.g.
  `test_sdpa_noncausal_mask__nightly[...-2-8-1-160-64-...]` → `ValueError: attn_mask (1,8,160,160)
  incompatible with (B=2,...)`. `mask_mode=custom` is in TARGET (`feature_spec.py:79`) and
  SUPPORTED, so this region *looks* covered but the distinguishing facet (mask batch dim vs `B`)
  is captured by no tagger.
- **Recommendation**: Add a mask-shape knob to `_build_inputs` (build `(1,H,S,S)` when a facet is
  set) + a `mask_batch_bcast: ["full","bcast"]` `INPUT_TAGGERS` axis; pin one `LOOSE_CASES` entry
  with a batch-broadcast mask on a `B=2` shape (minimal repro: `((2,8,160,64)×3)`, `mask_mode=custom`).
  Coordinate with SUPPORTED finding 1 (implement-or-exclude).
- **Confidence**: high.

**Finding 2 — input distribution is axis-blind (14 regression fails).**
- **What**: `test_uniform_input` / `test_negative_input` / `test_large_magnitude_input` fail on
  PCC/RMS; the same shapes pass in the golden matrix. The discriminator is the *distribution*
  (positive-only, negative-only, ×10), not the shape.
- **Evidence**: golden `_build_inputs` uses only centered `torch.randn`/`fa_rand`
  (`helpers.py:224-232`); regression worst case `test_negative_input[B1_H12_S512_D64]`
  pcc=0.9168 (numerical-bug), degrading with S/H (uniform: 0.999→0.979). Shapes
  `(1,12,512,64)` etc. all present and passing in golden `INPUTS` (`feature_spec.py:134`).
- **Recommendation**: Add one `LOOSE_CASES` entry per non-centered distribution via
  `extras.input_gen` (e.g. `"neg"`, `"uniform"`) on a mid shape like `(1,12,512,64)`, so the
  matrix — not just the ungated regression file — gates non-centered precision. Consider
  promoting `input_gen` to a first-class facet.
- **Confidence**: high (that it's untested); med (on whether it should gate, given it's a known
  bf16-softmax precision edge, not a logic bug).

**Finding 3 — no long-seq MQA cell at the low-precision config (8 translated fails).**
- **What**: `test_sdpa_tt__nightly` s=8192 and `test_sdpa_tt_large_seq` s=131072 (MQA, causal)
  miss RMS by a hair (e.g. 0.00987 vs 0.0092) under `math_approx_mode=True` / bf8b.
- **Evidence**: golden long-context `INPUTS` tops out at `(1,1,8192,64)` MHA and `(1,4,4096,64)`
  MQA (`feature_spec.py:183-186`) — no s≥8192 MQA cell, and `math_approx_mode` is not a golden
  axis. `test_translated.py:426` threshold 0.0092.
- **Recommendation**: Add a `LOOSE_CASES` MQA long-seq entry (`((1,8,8192,128),(1,1,8192,128)×2)`,
  `fp32_dest_acc_en=False`) with a `rmse_threshold` extra; low urgency (marginal).
- **Confidence**: med.

## 2. SUPPORTED honesty

`verify_supported` (blind dir): `supported_fail=54`, `xpass_drift=0`, `supported_pass=1728`. The
54 decompose into three axis clusters — the aggregate overstates real correctness bugs.

**Cluster A — over-claim, 24 cells (`custom` × {mha,mqa}, bf16).** Genuine: the op declares
`mask_mode=custom` SUPPORTED but rejects the common batch-broadcast custom mask
(`scaled_dot_product_attention.py:207`). *Recommendation*: **fix** (support `m_b==1` broadcast in
reader/validate) or **honest demote** — add the `mask_batch_bcast` tagger (Finding 1) and
`EXCLUSIONS += {mask_batch_bcast: "bcast"}` so the refusal is a support-refusal, not a raw
ValueError. Confidence: high.

**Cluster B — not an over-claim, 22 cells (`causal` × mqa, program-cache tests).** The op is
numerically correct; it fails `assert device.num_program_cache_entries() - n0 == 1`
(`test_translated.py:552`) because the `ttnn.generic_op` path adds **0** cache entries
(`assert (25 - 25) == 1`). This is a capability/limitation, not a wrong SUPPORTED claim — no
axis expresses it. *Recommendation*: do not demote; record as a known limitation (see §3) and,
if program-cache participation is in-scope, file it as a capability refinement. Confidence: high.

**Cluster C — marginal precision, 8 cells (long-seq, see Finding 3).** *Recommendation*: precision
refinement or threshold review; not a demote candidate. Confidence: med.

No `xpass_drift` → no under-claims to promote. Net: the honest count of correctness over-claims
is ~24, not 54.

## 3. Helper / reference docs

**Finding — `references/ttnn-generic-op-workflow.md` is silent on program-cache participation.**
- **What**: The reference workflow doc for building `generic_op` ops never states whether/how a
  `generic_op`-based op registers in the device program cache. The reference SDPA tests assert
  exactly one new cache entry across repeated dispatch; the generated op adds zero → 22 fails.
- **Evidence**: `grep -i cache references/ttnn-generic-op-workflow.md` → no hits; failure
  `test_translated.py:552` `assert device.num_program_cache_entries() - n0 == 1`. No breadcrumb
  in `agent_logs/*.jsonl` mentions program cache — the implementer never considered it (absence).
- **Recommendation**: Add a "Program caching" note to the generic-op workflow reference stating
  whether `ttnn.generic_op` programs are cached and, if a hashed/cached descriptor is required for
  `num_program_cache_entries` to grow, how to build one. One line closes a 22-cell blind spot.
- **Confidence**: med (high that the doc is silent; med on the exact remedy).

No misleading helper docstring surfaced — the kernel-helper usage in the breadcrumbs
(`[plan]`, `[fix_applied]`) shows correct helper semantics; refinement struggle was perf-tuning
(3b–5a), not helper confusion.

## 4. Agent prompts

**Finding — verifier SUPPORTED-honesty check is blind to sub-axis facets.**
- **What**: The Phase-0 verifier ran `verify_supported` over golden only, got a clean report, and
  reordered `validate()` so batch-broadcast-mask cells raise a support-refusal — but never flagged
  that `mask_mode=custom` is a SUPPORTED **over-claim** (the op rejects a common custom-mask
  shape). A "clean verify_supported over golden" gave false confidence because golden's taggers
  can't express mask batch-broadcast or input distribution.
- **Evidence**: `changelog.md:26-29` "Reordered validate() … batch-broadcast mask were being
  rejected with ValueError instead of the support-refusal" — the verifier saw the exact facet and
  treated it purely as exception-ordering. `blocking-verifier.md:142` routes `supported_fail` but
  assumes the golden cartesian is the ground truth for honesty.
- **Recommendation**: Add to `blocking-verifier.md`: when a SUPPORTED axis value (e.g. a mask mode)
  admits shape sub-facets the taggers don't capture, spot-check the tensor-shape contract against
  the reference op's accepted shapes — a clean `verify_supported` over golden is necessary, not
  sufficient. Confidence: med.

**Finding (low) — planner never scoped program-cache as a capability.**
- **Evidence**: `op_design.md` / `op_requirements.md` contain no program-cache requirement; the
  reference suite asserts on it. *Recommendation*: `blocking-planner.md` could add program-cache
  participation to the capability checklist for ops translated from cached reference ops.
  Confidence: low.
