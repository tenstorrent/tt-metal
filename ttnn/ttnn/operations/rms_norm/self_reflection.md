# Self-Reflection: rms_norm (advisory — nothing here is auto-applied)

## Summary
Final blind pass is **clean**: golden 5155 passed / 0 failed / 0 hang / 33918 skipped / 1365 xfail;
translated 84 and regression 15 all green (`golden_blind_final/golden_results.txt`,
`test_results.json`). SUPPORTED honesty is perfect — `supported_fail=0`, `xpass_drift=0`
(`verifier_report.json`), with SUPPORTED grown to the **entire** TARGET rectangle and a single
legitimate EXCLUSION. So the *measurement* found nothing; the signal is entirely in the
**trajectory** (phase0→6g, ~15 refinements, ~$210). The single most important finding: a
**NaN (rsqrt of garbage)** shipped into an already-SUPPORTED BLOCK-sharded config during a perf
refinement and **golden stayed green** — it was caught only by the agent's own ablation test.
This is a **framework/test-universe** gap (golden is blind to shard *geometry*), not an op defect
(the op is fixed and clean). The other findings are helper/reference doc absences surfaced by the
perf work.

## 1. Golden coverage → `feature_spec.py`

**Finding 1 — Axis-blind gap: BLOCK-sharded non-uniform core-ownership NaN (high severity).**
- **What**: Refinement 6e's two-phase fold produced a NaN on any BLOCK-sharded group whose
  per-group tile-width `K` does not divide its tile-row count `C` (non-uniform folder ownership
  breaks the `get_write_ptr` fixed-base remote-address proxy). Golden never saw it: every golden
  BLOCK case is a uniform grid (the `_SHARDED` loose case 256×512→8×8, and perf case
  8192×1024→8×8, `feature_spec.py:245,331`), so `memory_layout=BLOCK_SHARDED` reads "covered"
  while the discriminating facet — shard *geometry* uniformity — is untested.
- **Evidence**: commit `d34f2681cb` "fix two-phase non-uniform-ownership nan … K=7, C=8"; changelog
  "R6e passed earlier only because every tested topology (8×8 K=8/C=8; 4×4 K=4/C=8) happens to have
  UNIFORM ownership"; the bug reproduced only under the agent's private
  `test_rms_norm_r6_ablation.py::test_ablate[K28_HT16_512x224]`, while `golden test_golden.py
  5056 passed / 0 failed` with the NaN present. In-TARGET (BLOCK_SHARDED ∈ SUPPORTED, tile-aligned
  bf16), so it is a real gap, not correct-to-fail.
- **Recommendation**: add a `LOOSE_CASES` entry pinning the exact non-uniform geometry (helpers.py:192
  honors `extras` shard specs and takes precedence over the auto path):
  `{"inputs": ((1,1,512,224),), **_SHARDED, "memory_layout": _ML.BLOCK_SHARDED, "extras": {"shard_shape":[128,32], "core_grid":(7,4)}}`.
  The distinguishing facet (K∤C) derives from the `memory_config` shard spec, which the shape-only
  `INPUT_TAGGERS` cannot see — so it is **structurally axis-blind** and a pinned loose case (not a new
  axis) is the right lever. Consider a small suite of "awkward-grid" sharded loose cases (odd
  `core_grid`, K∤C) alongside the clean-fit ones.
- **Confidence**: high.

_No other golden gap._ Every in-TARGET defect the run hit *elsewhere* was caught by golden itself
(e.g. the R5a fp32/W=8192 HEIGHT L1 overflow — breadcrumb "592 passed, 7 failed"; the R1/R2
fp32_dest_acc_en=False/W=4096 precision defect — "12 failed"), so coverage there is adequate.

## 2. SUPPORTED honesty → op file `SUPPORTED` / `EXCLUSIONS`

**No finding — clean.** `verifier_report.json` for the blind dir: `supported_pass=5140`,
`supported_fail=0` (no over-claim in any axis cluster), `xpass_drift=0` (no under-claim). SUPPORTED
(`rms_norm.py:102`) equals the full TARGET rectangle across all 9 axes; the lone EXCLUSION
`{float32, fp32_dest_acc_en=False}` (`rms_norm.py:138`) is design-legitimate (lossy corner,
`references/precision_convention.md`) and is held xfail-strict (`xfail_expected=1365`). Nothing to
fix, demote, or promote.

## 3. Helper / reference docs

**Finding A — `cross_core_reduction_design.md` states the "divide cleanly" rule for the host shard
grid but is silent on the generalized invariant the NaN violated (high).**
- **What**: The doc says the sharded norms "require the shard grid to divide cleanly and reject
  configs that don't" (`.claude/references/cross_core_reduction_design.md:436`) — but that rule is
  scoped to the *host shard grid*. The R6e two-phase path introduced an *internal* decomposition
  (`NUM_FOLDERS`) that reuses the same `get_write_ptr`-as-fixed-base-proxy pattern
  (`kernels/rms_norm_xcore_writer.cpp:18-20,194` "uniform CB base across cores") yet is not a shard
  grid, so the doc's rule didn't obviously apply — and it broke.
- **Evidence**: breadcrumb (6e-debug) "the gather-push uses get_write_ptr(cb_gather) as a
  remote-address proxy, which assumes uniform base across all cores … Divergent pointers → nan."
- **Recommendation**: generalize the invariant in the doc — "**any** cross-core CB fan-in that uses
  `get_write_ptr` as a fixed-base remote proxy requires uniform per-core advance (`owned*K == depth`
  every round), including internal fold/tile-index decompositions, not just the shard grid."
- **Confidence**: high.

**Finding B — `element_size()` raises for bf8b, undocumented (med-high).**
- **What**: `tensor.element_size()` throws for block-float (bf8b) dtypes; no reference warns of it,
  so the implementer had to write a defensive `_elem_size()` wrapper.
- **Evidence**: breadcrumb "element_size() raises ValueError for bf8b (block-float). Fixed via
  _elem_size() defensive helper"; `rms_norm_program_descriptor.py:176-187`.
- **Recommendation**: add one line to `.claude/references/ttnn-python-utility-bindings.md` (or
  `torch_ttnn_api_divergence.md`): "`element_size()` raises for block-float dtypes (bfloat8_b) — 16
  values share one exponent; guard before calling."
- **Confidence**: med-high.

**Finding C — `compute_fusion` / `binary_dest_reuse_tiles` docs silent on no-broadcast + a measured
dest-reuse perf dead-end (med).**
- **What**: The pass-2 `x·rstd·gamma` fusion lever was pursued across R6f and R6g before being
  abandoned: the helper has no broadcast form (gamma needs ROW-bcast) and FPU-consumes-DEST reuse is
  *slower* than a pack-to-L1 roundtrip on Blackhole (and WH). Docs didn't state either limit, so the
  cost was paid by measurement.
- **Evidence**: breadcrumb "binary_dest_reuse_tiles has no bcast; BinaryFpu cant consume DEST …
  compute_fusion 0.82x measured-inferior"; R6g "dstreuse … 0.94-1.00x … never beats pack-to-L1
  roundtrip; matches WH 0.82x."
- **Recommendation**: add a caveat to the `compute_fusion` example / `binary_dest_reuse_tiles`
  docstring: "no broadcast variant; FPU dest-reuse measured ≤1.0× vs pack-to-L1 on BH/WH — prefer the
  L1 roundtrip for bcast-multiply chains."
- **Confidence**: med.

**Finding D — two reduce-helper wrappers were abandoned as stale at phase-0 (med/low).**
- **What**: The implementer dropped `accumulate_reduce_block` (CBs are runtime args, expected
  template) and `prepare_partial_reduce_scalers` (4 template args, expected 3), deviating to
  `reduce<>` / two `prepare_reduce_scaler` calls.
- **Evidence**: breadcrumb "two forced kernel_lib deviations … accumulate_reduce_block … stale
  wrapper: CBs runtime vs template; prepare_partial_reduce_scalers … 4 template args vs 3"; live
  signatures at `streaming_reduce_helpers.hpp:53`, `reduce_helpers_dataflow.hpp:140`.
- **Recommendation**: reconcile the doc/reference that set the implementer's expectation with the
  shipped signatures (direction of the drift is unclear from the breadcrumb — worth a human check).
- **Confidence**: med/low.

## 4. Agent prompts → `.claude/agents/*.md`

**Finding — verifier no-regression guard set is geometry-blind; a perf refinement shipped a NaN past
the golden gate (high).**
- **What**: `blocking-verifier.md:315` defines the no-regression guard as "one representative per
  distinct kernel path × layout × **placement**." A perf refinement that adds an internal path gated
  on a *geometry predicate* (two-phase fold, engaged when `K` divides `C`) is then only exercised on
  the single (uniform) placement representative — so the NaN class was invisible to the gate. Golden
  (the independent measurement) is equally blind, so the only net that caught it was the agent's own
  discretionary `test_rms_norm_r6_ablation`.
- **Evidence**: `op_requirements.md` R6f-debug — "Bullet 2 FAIL: acceptance/refinement tests failing:
  … test_rms_norm_r6_ablation.py::test_ablate[K28_HT16_512x224] - AssertionError: nan"; changelog
  "R6e passed earlier only because every tested topology … happens to have UNIFORM ownership";
  golden was `5056 passed / 0 failed` with the bug live.
- **Recommendation**: in `blocking-verifier.md`, extend the guard-set rule — when a perf refinement
  introduces an internal path **predicated on a shape/geometry condition**, `**Done when**` must
  require a guard shape at the predicate *boundary* (e.g. a shard grid with `K∤C` → non-uniform
  ownership), not just one representative per placement. Equivalently, require such refinements to
  land a matching golden `LOOSE_CASES` entry (§1) so the independent gate, not a private ablation,
  covers the regression.
- **Confidence**: high.

_Positive note (no change requested)_: the verifier's "ablate-in-op before committing" discipline
(breadcrumb R6e "verifier directs ablate-in-op-yield-before-committing") worked well — the two perf
dead-ends (allgather 0.52×, gamma-fusion 0.94–1.00×; commits `aca1ae3274`, R6g) were characterized
cheaply and *not* shipped. The ~8 phase-6 sub-iterations were productive (BLOCK 8×8 5.76×→2.11×
above achievable), not churn; flagging only in case an explicit perf-iteration budget is desired
(confidence: low).
