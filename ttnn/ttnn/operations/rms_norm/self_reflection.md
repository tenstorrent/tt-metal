# Self-Reflection: rms_norm (run_1)

## Summary
- **Final blind pass:** 5550 passed, 0 failed, 0 hangs (golden 5429 pass + 1365 expected xfail; translated 106/106; regression 15/15). `supported_fail = 0` and `xpass_drift = 0`. The queue drained: Phase 0 plus R1–R6 are all `[x]`. No perf phase ran.
- **Most important finding:** the Phase-0 golden run reported 435/0, but it hid a real kernel bug. A shared `reduce` helper returned a wrong rstd from the 2nd tile-row of every multi-row block on. Golden could not see it, because no supported cartesian cell gives a core more than one tile-row. The relevant fact, tile-rows per core, is not an axis ("axis-blind").
- The remaining problems are mostly at framework level: two helper-library pitfalls (one fixed in-run inside shared code), a hardcoded L1 budget in a skill, and two `INVALID` entries in the golden spec that the op has outgrown.

## 1. Golden coverage (`eval/golden_tests/rms_norm/feature_spec.py`)

**F1.1 — Axis-blind gap: no supported golden cell puts more than one tile-row on a core.**
- **What:** The bug where the reduce post-op is not re-armed (every block with `block_rows >= 2` got a wrong rstd from row 2 on) was present from Phase 0. The golden run was clean anyway. The largest Phase-0 golden shape is `1x1x2048x256`, which is 64 tile-rows on 64 cores, so 1 row per core. Every loose case with more rows than the grid (resilience `99991x64`, perf 7168/8192-row cases) is pinned to `fp32_dest_acc_en=False`. That value was an EXCLUSION / unsupported at Phase 0, so those cells were xfail.
- **Evidence:** Commit `1786aebd9c8`: "Pre-existing Phase 0 bug surfaced: any block with >=2 tile-rows (interleaved 7680x512 too) gets a wrong rstd". `refinement_1_output.json`: "interleaved fp32 at 7680×512 had a max error of 0.48 … Phase 0 tests never caught it". `golden_phase0/registry_snapshot.json`: `fp32_dest_acc_en: [true]`. The maximum tile-row count among Phase-0 passes is 64 (computed from `golden_phase0/test_results.json`).
- **Recommendation:** Add `LOOSE_CASES` entries pinned to the Phase-0 corner so they are supported from day one:
  - `{"inputs": ((1,1,7680,512),), dtype=float32, fp32_dest_acc_en=True, layout=TILE, gamma_mode="gamma", gamma_dtype=float32, gamma_layout=ROW_MAJOR, memory_layout=INTERLEAVED}`
  - a bf16 twin `((1,1,8192,256),)`

  Both have more than 2 tile-rows per core on any grid of 120 cores or fewer. Also consider an op-agnostic `rows_per_core` facet: a tagger bucketing `ceil(Rt / grid)` into `1` / `>1`. INPUTS spans many row counts, but the generic golden template has nothing that guarantees one of them exceeds the grid.
- **Confidence:** high

**F1.2 — Two `INVALID` entries are now stale: the op supports bf8b on non-aligned shapes, and golden skips them.**
- **What:** The verifier report has `invalid_unexpected = 2`. Two translated tests with `dtype=BFLOAT8_B, alignment=w_non_aligned, WIDTH_SHARDED` pass. The op's `SUPPORTED` covers bf8b × all alignments with no EXCLUSION, so every `{bf8b, w_non_aligned | h_non_aligned}` cell is claimed. Golden skips every one of them as INVALID, and no LOOSE_CASE can reach them, since loose cases project through the same INVALID filter.
- **Evidence:** `golden_blind_final/verifier_report.json` `invalid_unexpected`: `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w72_c2_nonaligned-bfloat8_b]` passed, and the same for `w200_c3`. `feature_spec.py`: `{"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"}` / `"h_non_aligned"`, labelled "Author-scoped exclusions ('for now')".
- **Recommendation:** Propose that a human remove those two author-scoped INVALID entries. The bf8b × non-aligned cells then enter the golden cartesian, where they are graded instead of only claimed. If the cost is a concern, keep them and add a feature_spec comment that the op already claims them, so the coverage hole is visible.
- **Confidence:** high (that they are stale); med (on whether to widen versus keep parked)

**F1.3 — The regression tests carry no axes (`no_axes_found = 15`).**
- **What:** All 15 `test_regression.py` cells (small/large magnitude, uniform, sign-only inputs) pass, but none has an axes dict. They are invisible to SUPPORTED-honesty accounting.
- **Evidence:** `golden_blind_final/verifier_report.json` `no_axes_found`, e.g. `test_regression.py::test_small_magnitude_input[32x64]`.
- **Recommendation:** Consider having `test_regression.py` emit `observed_axes`, as `test_golden.py` does, so a future regression failure clusters with the rest.
- **Confidence:** med

*Infeasible skips are not a gap.* The 21 `infeasible_skipped` cells (fp32 W=8192 HEIGHT_SHARDED with a single 32-row shard; `5x3x928x544` / `99991x64` WIDTH_SHARDED) need input and output shards larger than one core's L1. They are correct to skip.

## 2. SUPPORTED honesty (`rms_norm.py` `SUPPORTED` / `EXCLUSIONS`)

No over-claim or under-claim in the final blind pass: `supported_fail = 0`, `xpass_drift = 0`. All 1365 xfails are the single declared EXCLUSION `{float32, fp32_dest_acc_en=False}` (`rms_norm.py:87-90`). It matches the precision convention noted in `feature_spec.py` TARGET.
- One related fact, already covered in F1.2: SUPPORTED claims bf8b × non-aligned, and only 2 translated tests confirm it. The claim is honest as far as those tests go, but golden cannot grade it.
- History note: at Phase 0 the claim `fp32_dest_acc_en=True × any rows` was dishonest without anyone knowing (F1.1). It was fixed in R1, commit `4c914010b23`.

## 3. Helper / reference docs

**F3.1 — The `reduce<…AccumulateViaAdd, Collapse>` helper does not re-arm the SFPU reduce after a caller's SFPU `post_reduce_op`.** This was a real bug in shared code, fixed in-run.
- **What:** `sfpu_reduce_init` was armed once before the output loop. A caller post-op (here `rsqrt_tile_init` and friends) replaces that macro, so outputs 2..n collapsed with the wrong SFPU program. The implementer patched `kernel_lib/reduce_helpers_compute.inl` inside an op refinement commit.
- **Evidence:** Commit `4c914010b23` adds the re-arm at `reduce_helpers_compute.inl:592-599`: "re-arm it so the NEXT output's sfpu_reduce collapses with the reduce macro". Breadcrumb: "reduce AccumulateViaAdd re-arms sfpu_reduce_init after SFPU post_reduce_op".
- **Recommendation:** Propose upstreaming this hunk as its own helper-library change, together with a helper unit test (`n_out >= 2`, with an SFPU post-op). Also add a `.hpp` note that `post_reduce_op` may call any `*_tile_init`, and that the helper restores its own state.
- **Confidence:** high

**F3.2 — `tilize` example 6 (back-to-back calls) corrupts output with the default reconfig mode.**
- **What:** The `.inl` runs `pack_reconfig_data_format` on every call whose `reconfig_mode` includes pack, regardless of `InitUninitMode`. On Neither/UninitOnly calls that come after `fast_tilize_init`, this breaks the fast-tilize pack setup. Example 6 in the `.hpp` uses exactly that default.
- **Evidence:** `tilize_helpers.hpp:175-180` (example 6 sets no reconfig mode); the default is `UnpackAndPackReconfigure` (`:200-201`). `tilize_helpers.inl:162-165` reconfigures unconditionally. Commit `172be59777b`: "RM wide rows (c=128) wrong". Breadcrumb: "pcc ~0.5 on RM W=4096, K=2". The workaround is at `rms_norm_compute.cpp:171-181`.
- **Recommendation:** Two options, in order of preference:
  - In the `.inl`, apply reconfig only when the mode includes Init, or `static_assert` that Neither/UninitOnly come with `NoReconfigure`.
  - At minimum, fix example 6 to pass `ReconfigureRegisterDatatypeMode::NoReconfigure` on the middle and last calls.
- **Confidence:** high

**F3.3 — The `AccumulateViaAdd` partial-mask fold silently requires the mask CB format to match the input CB format.**
- **What:** `fold_partial_last` unpacks the mask as srcB without reconfiguring srcB away from the input format. A bf16 mask, which is what the docs imply, is misread when the input CB is Float32. The `.hpp` partial-scaler docs say nothing about format.
- **Evidence:** `reduce_helpers_compute.inl:350-357` (no `reconfig_data_format_srcb`). `reduce_helpers_compute.hpp:312-333` names no format requirement. Op workaround: `rms_norm_program_descriptor.py:157-158`: "the masked fold unpacks the mask as srcB under cb_sumsq's format".
- **Recommendation:** Either reconfigure srcB to `scaler_dfb` inside `fold_partial_last` and restore it afterwards, or add one `.hpp` line: "`only_partial()` mask CB must use the input CB's data format".
- **Confidence:** high

**F3.4 — `memory-budget-metal` skill hardcodes an L1 budget that overstates the real CB space.**
- **What:** `L1_BUDGET = 1499136` (and `get_max_worker_l1_unreserved_size()` = 1530976) are both above the CB region actually available (~1460288 on this Blackhole). Sizing against them overflowed L1 for fp32 W=8192. Later, L1-resident shards shrank the region further, and the op had to bound CBs by the lowest allocated L1 address.
- **Evidence:** `.claude/skills/memory-budget-metal/SKILL.md:128`. Commit `172be59777b`: "fp32 W=8192 throws". Commit `b70f85c23cc`: "lowest-allocated-address CB budget"; implemented at `rms_norm_program_descriptor.py:325-329`.
- **Recommendation:** Replace the constant with a documented runtime query: `get_memory_view(device, L1)`, bounded by the lowest allocated buffer address. State that L1-resident tensors, sharded inputs included, reduce the CB budget.
- **Confidence:** high

## 4. Agent prompts

**F4.1 — Planner/verifier never required a "more than one tile-row per core" acceptance case.**
- **What:** The Phase-0 acceptance test (106/106) and the verifier's precision baseline both passed with the multi-row bug latent (F1.1). Block rows ≥ 2 is a structurally distinct path: the per-row loop inside a block. Neither prompt asks for it to be exercised.
- **Evidence:** Commit `782664df484`: "PASS: rms_norm acceptance test 106/106". The bug was found only in R1 (`1786aebd9c8`).
- **Recommendation:** Add one line to `.claude/agents/incremental-planner.md` (acceptance-test section) and to `incremental-verifier.md` (gate): "include at least one shape whose per-core work exceeds one block-quantum (tile-rows > grid size) for every loop level the design declares".
- **Confidence:** med

**F4.2 — The planner applied the CB ring-wrap rule to push quanta, but not to an indexed read window on an aliased CB.**
- **What:** The design aliased `cb_sumsq` with `cb_normed` at capacity `round_up(r·max(c,n_sq), lcm(c,n_sq))`. The reduce reads an `r·n_sq` window by index, and that window does not divide the capacity, so it drifted until it straddled `fifo_limit`. `ttnn-cb-memory-fundamentals.md:85-163` covers push quanta closing the cycle, but not windows read by index (BulkWait/NoWait).
- **Evidence:** `op_design.md:298` (cap_alias). Breadcrumb `op_design.md:298`: "reduce reads its r*n_sq window by index, so the window drifts … straddles fifo_limit".
- **Recommendation:** Add one line to `ttnn-cb-memory-fundamentals.md` §Ring-Wrap Invariant: "a window read by index (BulkWait / NoWait / Offset) is a quantum too; capacity must be a multiple of every consumer's per-block window, not just of every push size."
- **Confidence:** med

**F4.3 (minor) — The verifier prompt does not say whether `verifier_report.json` is committed.**
- **Evidence:** Verifier breadcrumb: "on a full golden run it is ~25 MB and the check-large-files pre-commit hook (500 KB) rejects the commit". `incremental-verifier.md:205`.
- **Recommendation:** State at `incremental-verifier.md:205` that the full report stays in `<results_dir>` and is not committed.
- **Confidence:** high
