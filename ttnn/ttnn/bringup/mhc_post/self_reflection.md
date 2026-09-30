# Self-reflection: mhc_post (run_1)

## Summary
- **Final blind pass: 208/208 green.** That is 196 golden (`test_op` + `test_op_loose`) plus 12 regression. The pass had **no translated cells**. verify_supported: 208 `supported_pass`, 0 fail, 0 drift. Every TARGET value is in SUPPORTED, with no EXCLUSIONS.
- **Most important finding: a bug inside the op's contract that the golden suite cannot see.** With n = 2 streams and ≥ 2 column tiles per block, the compute kernel fails to compile. The perf phase found this (changelog.md:217-220). `validate()` still accepts n ∈ [1, 5], but every golden input uses n = 4 (`feature_spec.py:52 _N = 4`), and no tagger captures n.
- The problems are mostly **framework-level**:
  - a golden axis model that is blind to n;
  - the perf-phase route for escaped bugs covers only sharding;
  - resume overwrites the Phase-0 golden baseline.
- The op-level problem is the n = 2 kernel spill itself.

## 1. Golden coverage → `eval/golden_tests/mhc_post/feature_spec.py`

**1a. Axis-blind gap: the stream count n (n ≤ 2 with block ≥ 2 column tiles)**
- **What:** The compute kernel picks its DEST-window regime from n at compile time. The design says "K > 1 at n <= 2" (op_design.md:115). `WINDOW_COL_TILES = min(WINDOW_FIT, block_col_tiles)` (mhc_post_compute.cpp:60), so the K > 1 path only exists when n ≤ 2 **and** B ≥ 2. That path does not compile.
  - Golden cannot reach it: `_case()` hardcodes n = 4, and `INPUT_TAGGERS` only has `alignment`.
  - The unit test does not reach it either. `test_mhc_post_streams.py` (T64 C224, T45 C96) gives B = 1.
  - The failing facet is the conjunction `n ≤ 2 ∧ B ≥ 2`. That is not an input the axis model can express.
- **Evidence:**
  - perf-part-optimizer breadcrumb: "REAL OP BUG: n=2 with B>=2 (fits regime K=2) fails to compile (SFPU register spill … compute.cpp:102-103). Reproduced with the real op at T640 C1792 n=2 bf16".
  - changelog.md:219: "Under the device profiler, n = 1 also fails to build".
  - No later commit touches it (`git log -S` shows only R2, Perf 1 and Perf 2).
- **Recommendation:** Propose these `LOOSE_CASES` entries (non-`_case` inputs; `helpers.make_inputs` already derives n from `post_shape[-1]`):
  - `{"inputs": ((1,1,640,1792),(1,1,640,3584),(1,1,640,2),(1,1,640,4)), "dtype": bf16, "sublayer_dtype": bf16, "layout": TILE, "fp32_dest_acc_en": True}` — n = 2, B = 4.
  - The same entry with n = 1: `(1,1,640,1792),(1,1,640,1792),(1,1,640,1),(1,1,640,1)`.
  - Also consider promoting n to an `INPUT_TAGGERS` axis (e.g. `streams: {"n4", "n_le2", "n3", "n5"}`) so that SUPPORTED has to state what it claims about n.
- **Confidence:** high (n = 2). Low for n = 1 outside the profiler build.

**1b. Other shape-keyed code paths: covered, no gap.**
The perf carve-outs branch on facets that no tagger captures, but the existing golden cartesian already exercises both sides of each:
- `HELP_MIN_BLOCKS = 6` (<6 vs ≥6 blocks per core);
- `HELP_FP32_STREAMS` (fp32 X);
- `ROW_WEIGHT_MIXED_FP32_STREAMS` (fp32 X / bf16 F).

Small `INPUTS` shapes fall below 6 blocks, the loose sweep up to T2048 C7168 goes above, and the dtype pairs come from TARGET. No proposal. Confidence: med (block counts inferred from changelog tables, not re-derived per cell).

## 2. SUPPORTED honesty → `mhc_post.py` `SUPPORTED` / `EXCLUSIONS`

- **Declared axes:** `supported_fail` = 0 and `xpass_drift` = 0 in every phase (golden_blind_final/verifier_report.json). Nothing to fix on the declared axes.

**2a. Over-claim on an undeclared facet: n ∈ {1, 2}**
- **What:** `validate()` accepts `1 <= n <= MAX_STREAMS` (mhc_post.py:111), and op_design.md:19 advertises `1 <= n <= 5`. But n = 2 with B ≥ 2 is a build failure, not a clean refusal. verify_supported cannot catch this because n is not an axis. It is an over-claim in the op's contract rather than in `SUPPORTED`.
- **Evidence:** changelog.md:218 "n = 2 with B ≥ 2 does not compile: the SFPU register spills (`cannot write SFPU object to memory`…)".
- **Recommendation: fix, don't demote.** The cluster is one compile-time regime, `WeightedSumSfpu::row` with K = 2. The likely site is register pressure in `WeightedSumSfpu::row` (mhc_post_compute.cpp:93-104) when the chain unrolls K WeightedSums. The cheapest honest fix is to clamp `WINDOW_COL_TILES` to 1 until the K > 1 regime compiles. As an interim, validate could refuse n ≤ 2 with `UnsupportedAxisValue`, after n becomes an axis (§1a).
- **Confidence:** high.

## 3. Helper / reference docs

**3a. The SFPU face-advance rule is undocumented (absence)**
- **What:** The implementer lost a probe cycle on this. `_llk_math_eltwise_sfpu_inc_dst_face_addr_()` rebases on the carry (face-start) register, which discards any in-loop `dst_reg++`. So a custom SFPU loop that walks rows needs two calls to get from face 0 to face 2. The LLK itself has no comment saying so.
- **Evidence:**
  - tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h:28-32 (two bare `inc_dst_addr<8>()`, no comment).
  - Implementer breadcrumb: "sfpu face inc rebases on carry reg; need 2 incs face0->face2".
  - changelog.md:67 "Found with a structured probe (face 0 was correct, faces 1–3 were wrong)".
- **Recommendation:** Propose a doc line at the LLK and in the kernel_lib custom-SFPU guidance: "inc_dst_face_addr advances from the face *start*; in-loop `dst_reg++` is not preserved; face 0→2 = two calls." Also note there that the `dst_reg +=` immediate range is [-8, 7] (changelog.md:68).
- **Confidence:** high.

**3b. No documented recipe for a custom DEST-only SFPU chain element**
- **What:** `WeightedSum` works around the missing recipe in two ways:
  - It borrows `llk_math_eltwise_ternary_sfpu_init<SfpuType::addcmul>()` as its init (mhc_post_compute.cpp:128).
  - It hand-wraps `_llk_math_eltwise_sfpu_start_/done_` (mhc_post_compute.cpp:131-133).
- The `UnaryOp` CRTP contract (chain.inl:563-577) documents `init`/`exec_impl` only for wrapping an existing `*_tile` API. It says nothing on which init or start/done a raw-SFPI body needs.
- **Evidence:** as cited above. The borrowed init is correct only by coincidence of SFPU config.
- **Recommendation:** Propose adding to the chain.inl CRTP comment a sketch of a raw-SFPI `UnaryOp` element: the generic init, start/done, and `lane_width` override. Alternatively, provide a `SfpuBody<F>` base that owns these.
- **Confidence:** med.

#### Helper gaps (perf)
Both perf rounds recorded "Helper bypasses — none" (changelog.md:223, 326). A grep of the kernels confirms that the DM kernels use only dataflow_api. It also shows raw LLK in compute, which predates perf (Refinement 2). It is justified in a kernel comment but appears in **no** bypass table.

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| eltwise ternary `Addcmul` / chain (n-term weighted sum) | not recorded (R2 comment: "No kernel_lib element expresses an n-term weighted sum") | **missing** — and an unrecorded bypass | ternary.inl:36-52: `Addcmul` is a single out = in0 + v·in1·in2 with no accumulate-across-terms or face-subset coefficient. The raw body is at mhc_post_compute.cpp:73-134 (`_llk_math_eltwise_sfpu_*`, lines 116-117 and 131-133). R2 measured 388 → 305 µs at T640 C7168 bf16 together with the other R2 levers (not isolated). | Propose an n-term `WeightedSum` / `Accumulate` chain element in kernel_lib. Propose that the perf coordinator carries forward pre-existing raw-LLK sites as rows (not "none"). |

```cpp
// confidence: low (single call site); derived from WeightedSumSfpu's raw sequence
template <uint32_t NumTerms, Dst Coef0, CoefLayout L /*Full | HalfPacked*/, Dst Data0, Dst Out, bool Accumulate = false>
struct WeightedSum;   // Out = [Out +] Σ_k coef(k) * DEST[Data0 + k], fp32, accumulator in LREGs
// call site:
eltwise_chain(..., WeightedSum<n + 1, Dst::D0, CoefLayout::HalfPacked, slot(P), slot(P)>{}, PackTile<...>{});
```

## 4. Agent prompts / pipeline

**4a. The perf coordinator routes only sharding bugs upstream, so a correctness bug stayed a changelog note**
- **What:** The escaped n = 2 build failure was filed under "Findings (not follow-ups)". It did not go into op_requirements.md or the refinement queue, so nothing will pick it up.
- **Evidence:**
  - changelog.md:216-220.
  - perf-coordinator.md:286-292: the escaped-completeness rule names only sharding ("escaped completeness bug from the sharding refinement").
- **Recommendation:** Propose generalizing perf-coordinator.md:286. Any build or correctness failure inside `validate()`'s accepted contract is an escaped bug. It must be appended to op_requirements.md as an unchecked follow-up refinement, or surfaced as a run-level flag, never only as a changelog finding.
- **Confidence:** high.

**4b. An implementer or verifier accepted a compile-time regime that no test instantiates**
- **What:** Refinement 2 introduced three n-dependent regimes ("n ≤ 2 fits K > 1 … n = 5 runs a grouped path", changelog.md:52). It added `test_mhc_post_streams.py` "the n-dependent window regimes" (changelog.md:73), but at shapes where B = 1, so K > 1 is never built.
- **Evidence:** test_mhc_post_streams.py:23-24 (T64 C224, T45 C96).
- **Recommendation:** Propose a line in ttnn-implementer.md and incremental-verifier.md: "for every compile-time regime a refinement adds, cite one test whose shapes select it (show the derived knob values); a regime test at a shape that collapses the knob does not count."
- **Confidence:** high.

**4c. The runner overwrites the Phase-0 golden baseline on resume (pipeline, not a prompt)**
- **What:** golden_phase0 shows 208/208, and its registry snapshot already includes bf16. The real Phase 0 was 84 pass / 123 xfail (commit 36e56473ffd, changelog.md:19).
  - The dir is timestamped 06:07, after Refinement 3 (04:24) and during the perf_1 resume.
  - `_skip_current_golden` (eval/run_refinements.py:408-425) reuses results only if they postdate HEAD. On resume it therefore re-measures "Phase 0 baseline" on the current tree and replaces the historical baseline.
- **Evidence:** as above. The file mtimes in `refinements/` order phase0 (06:07) after refinement_3 (04:30).
- **Recommendation:** Propose that once a phase-labelled golden dir is scorable, it is immutable. On resume, write a separate `golden_resume_baseline/` instead of reusing the `golden_phase0` name. This keeps the dashboard's trajectory honest.
- **Confidence:** high.

**4d. The verifier precision protocol does not define the ULP base (absence)**
- **What:** The verifier logged that output-relative ULP blows up on near-zero outputs of signed multi-term sums. It reported "p99 = 12–14 on an exact fp32 kernel" and switched to term-magnitude ULP on its own.
- **Evidence:** incremental-verifier breadcrumb (surface `prompt`, "verifier prompt § Testing Protocol 3"). A grep finds no "ULP" in incremental-verifier.md or precision_convention.md.
- **Recommendation:** Propose adding to incremental-verifier.md §3 / precision_convention.md: "for sum/mix ops, report ULP in units of Σ|term_k| (alongside output-relative ULP)."
- **Confidence:** med.
