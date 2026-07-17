# Self-Reflection: scaled_dot_product_attention

_Advisory only. Nothing here is auto-applied; every item is a proposal for a human to ratify._

## Summary

Final blind pass: **golden 1685 passed / 0 failed** (all matrix cells pass or xfail), but
**translated 369 pass / 47 fail** and **regression 32 pass / 7 fail** (2086/54/79-skip/584-xfail overall;
`verifier_report.json`: `supported_fail=47`, `xpass_drift=0`). The op's *own axis matrix is clean* — every
failure lives in a facet the registry model doesn't capture. The single most important finding: a
**batch-broadcast attention mask** (`mask.B=1` while `B>1`) is silently mis-computed (PCC ~0.56), and the
golden harness can't even express that input. The problems read **framework/test-universe-level**, not
op-kernel-level: the failures cluster into (a) an axis-blind mask facet, (b) a `generic_op` program-cache
introspection assertion that inflates `supported_fail` by 22 without any correctness defect, and (c) a
regression tolerance mismatch.

## Golden coverage → feature_spec.py

**Finding 1 — batch-broadcast mask is untested AND axis-blind (top priority).**
- **What:** All 24 custom-mask failures share one conjunction: `bcast_mask_batch_dim=True ∧ B>1`
  (mask batch dim = 1 while Q/K batch > 1). Differential is clean — every `no-bcast-mask-batch-dim` cell
  and every `B=1` cell passes; only real batch-broadcast fails. The golden matrix's `custom` mask is
  *always* full-batch (`make_causal_mask(B,…) → (B,1,S,S)`), so this region is never exercised, and no
  `INPUT_TAGGERS` axis models mask broadcast — the axes-tuple looks "covered" yet fails.
- **Evidence:** blind cells `test_sdpa_noncausal_mask__nightly[bcast-mask-...-2-8-1-160-64-...]` PCC 0.627,
  `[...-128-4-4-128-32-...]` PCC 0.564; helper builds full-batch masks at `helpers.py:242`
  (`make_causal_mask(B, S_q, S_kv)` → `helpers.py:143` shape `(B,1,S_q,S_kv)`); taggers at
  `scaled_dot_product_attention.py:76-80` are alignment/attention_kind/kv_heads only.
- **Recommendation:** add a `mask_batch_bcast` (and `mask_head_bcast`) knob to
  `helpers.run_scaled_dot_product_attention` (build the mask with batch/head dim = 1), plus a minimal
  `LOOSE_CASES` entry pinning it — e.g. `inputs=((2,4,128,64),(2,4,128,64),(2,4,128,64))`,
  `mask_mode="custom"`, `extras={"mask_batch_bcast": True}`. The kernel has a distinct mask-stride path for
  broadcast vs full mask, so also consider promoting `mask_bcast` to an `INPUT_TAGGERS` facet (satisfies
  the blocking-verifier.md:142 "structurally different code path" bar). *Gate note:* whether batch-broadcast
  is in-TARGET is a policy call (PyTorch SDPA supports it; the op's contract at op_requirements.md:31 does
  not) — if the team rules it out-of-scope, the fix belongs in `validate()` (Finding 4), not golden.
- **Confidence:** high (that it's untested + axis-blind); med (that it should be added to TARGET).

**Finding 2 — golden's input distribution is fixed zero-mean; the same-sign regime is a blind spot.**
- **What:** The 7 regression precision failures are all same-sign inputs (`test_uniform_input` positive-only,
  `test_negative_input` negative-only), with RMS growing along sequence length (S128→0.04, S256→0.05,
  S512→0.08–0.16). The golden matrix only ever draws zero-mean `torch.randn`/`fa_rand`, so it never stresses
  the large-systematic-score regime that same-sign inputs create.
- **Evidence:** `test_regression.py:95` (`torch.rand`), `:107` (`-(rand+0.5)`) fail; `helpers.py:230`
  (`torch.randn`) and `fa_rand` at `helpers.py:125` are both zero-mean; regression `test_small/large_magnitude`
  (also zero-mean) pass.
- **Recommendation:** regression already covers this (working as intended); if the same-sign precision
  characteristic matters for TARGET, add one `LOOSE_CASES` entry with a same-sign `input_gen` and a documented
  looser threshold rather than widening `INPUTS`. Otherwise leave golden as-is.
- **Confidence:** low-med.

## SUPPORTED honesty → op file SUPPORTED / EXCLUSIONS

`verify_supported`: `supported_fail=47`, `xpass_drift=0`. The aggregate 47 is **misleading** — it splits into
three very different buckets (24 real / 22 artifact / 1 marginal):

**Finding 1 — 24 `mask_mode=custom` cells: real over-claim (batch-broadcast).**
- **What:** The op declares `mask_mode:"custom"` SUPPORTED, but the batch-broadcast sub-case mis-computes.
  Because broadcast is not an axis, the verifier can only attribute this to the whole `custom` axis — yet
  full-batch custom masks pass fine.
- **Evidence:** 24 `supported_fail` entries with `axes.mask_mode="custom"` (verifier_report.json), all
  `bcast_mask_batch_dim=True`; `validate()` checks mask *head* dim only (`scaled_dot_product_attention.py:222`
  `if m_shape[1] not in (1, h_q)`), never the batch dim.
- **Recommendation:** **fix** the kernel's mask reader to broadcast over batch (matches the torch reference),
  **or demote** by having `validate()` reject `mask.B ∉ {1==B, B}` (turn a silent wrong answer into a clean
  `ValueError`). Do not leave it silently accepted.
- **Confidence:** high.

**Finding 2 — 22 `mask_mode=causal` cells: false `supported_fail` (program-cache artifact).**
- **What:** These are `test_sdpa_tt_with_program_cache__nightly` — the op computes **correctly** (PCC ~0.999);
  the only failing line is `num_program_cache_entries() delta == 1` (observed delta = 0). The `generic_op`
  path does not register a program-cache entry, so this asserts a framework capability, not op correctness.
- **Evidence:** all 22 have PCC 0.9994–0.9998; message `assert (42 - 42) == 1`; assertion at
  `test_translated.py:552`.
- **Recommendation:** do **not** treat these as SUPPORTED dishonesty (nothing to fix/demote in the axis
  model). Real over-claims are 24, not 47 — the aggregate over-counts by ~47%. Root cause is
  framework/test-scope (see Docs + Prompts sections).
- **Confidence:** high.

**Finding 3 — 1 `mask_mode=causal` cell: marginal precision at extreme S (singleton).**
- **What:** `test_sdpa_tt_large_seq__nightly[1-8-1-131072-128-...]` misses a tight RMSE gate (0.0119 vs 0.0094;
  PCC 0.9989). Non-catastrophic singleton at S=131072 — per method, no standalone action, but it reinforces
  the "precision degrades with sequence length" theme (Finding 2, golden coverage).
- **Evidence:** blind nodeid above, `assert 0.0119… < 0.0094`.
- **Confidence:** med.

**`xpass_drift=0`** — no under-claim; nothing to promote.

## Helper / reference docs

**Finding 1 — `generic_op` references are silent on program caching.**
- **What:** 22 translated cells fail purely on a program-cache-count assertion because a `generic_op`-based op
  doesn't populate `num_program_cache_entries`. Neither the workflow reference nor the template mentions
  program-cache behavior, so the implementer had no rule to satisfy (or to explicitly opt out of).
- **Evidence:** `grep program.cache/num_program` returns nothing in
  `.claude/references/ttnn-generic-op-workflow.md` or `.claude/references/generic_op_template/`; failures at
  `test_translated.py:552`.
- **Recommendation:** add one line to `ttnn-generic-op-workflow.md` stating whether/how `generic_op` ops
  participate in the program cache (and, if they don't, that `num_program_cache_entries`-delta assertions are
  expected to be xfail/collapsed for such ops).
- **Confidence:** med.

**Finding 2 — regression tests bypass the op's own tolerance profile.**
- **What:** `test_regression._run_self_attn` calls `check_output(...)` with **no** `tolerance=`, so it falls
  back to `metrics.DEFAULT_TOLERANCES[bf16]=(0.995, 0.04)` — *tighter* than SDPA's own
  `helpers.TOLERANCES[(bf16,True)]=(0.995, 0.05)`. The borderline `uniform S128` case (RMS 0.041) fails at
  0.04 but would pass at the op's declared 0.05 → a tolerance-mismatch artifact, not a real regression.
  (S256/S512 and negative cases genuinely miss even 0.05, so those stay real.)
- **Evidence:** `test_regression.py:59-62` (no tolerance arg); `metrics.py:61` default `(0.995,0.04)`;
  `helpers.py:162` op profile `(0.995,0.05)`.
- **Recommendation:** have `_run_self_attn` pass `tolerance=TOLERANCES[(dtype, True)]`, or document on
  `helpers.TOLERANCES` that regression tests must use it so the op is judged by one consistent bar.
- **Confidence:** high.

**Finding 3 — compute-config default doc drift (minor).**
- **What:** `default_compute_kernel_config()` sets **HiFi4** + fp32 DEST, but `feature_spec.py`'s comment
  claims the None-default is "HiFi2 + fp32 DEST acc". Harmless today but misleads anyone reasoning about the
  default fidelity.
- **Evidence:** `scaled_dot_product_attention.py:36-37` (`math_fidelity = HiFi4`) vs `feature_spec.py:41`
  ("HiFi2 + fp32 DEST acc").
- **Recommendation:** correct the `feature_spec.py` comment to HiFi4 (or align the factory to the intended
  default).
- **Confidence:** high (severity low).

## Agent prompts

**Finding 1 — verifier `supported_fail` triage has no bucket for framework/introspection artifacts.**
- **What:** blocking-verifier.md:142 routes every `supported_fail` to fix-kernel / precision-refinement /
  shrink-SUPPORTED — none of which fit the 22 program-cache cells (correct output, failing only a
  `num_program_cache_entries` assertion the `generic_op` path can't satisfy). Such cells would be mis-triaged
  as kernel defects and inflate the honesty metric.
- **Evidence:** blocking-verifier.md:142 category table; the translated suite already *collapsed* the
  incompatible `mask_dtype` axis with an inline note (`test_translated.py:650-652`) — the same collapse/xfail
  treatment should apply to program-cache-count assertions.
- **Recommendation:** add a triage branch to blocking-verifier.md: a `supported_fail` whose failing assertion
  is a framework-introspection check (e.g. program-cache counting) with passing output is **not** an op
  defect — xfail/collapse it during translation, don't count it as SUPPORTED dishonesty.
- **Confidence:** med.

**Finding 2 — planner/validate under-scoped the declared mask-shape contract.**
- **What:** The op documents a mask contract `(B, {1,H}, S_q, S_kv)` (op_requirements.md:31) but `validate()`
  enforces only the head dim `∈ {1, h_q}`, S_q and S_kv — never the batch dim. A hazardous choice
  (accept-but-mis-compute batch-broadcast) shipped unspecified, producing the 24 silent wrong answers.
- **Evidence:** contract at op_requirements.md:31; enforcement gap at `scaled_dot_product_attention.py:214-223`
  (mask checks cover S_q/S_kv/head, omit `m_shape[0]`).
- **Recommendation:** blocking-planner.md / blocking-verifier.md should require every dimension in a declared
  input-shape contract be either enforced or explicitly rejected in `validate()` — no contract dim left
  silently unchecked.
- **Confidence:** med.
