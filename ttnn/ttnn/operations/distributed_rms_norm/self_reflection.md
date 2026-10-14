# Self-Reflection: distributed_rms_norm

## Summary
- **Blind pass:** 6762 passed, 3 failed, 0 hangs (of 15912 cells). Verifier categories: 6734 supported_pass, **2 supported_fail**, 0 xpass_drift, 1 xfail_wrong_mode. Every golden cartesian cell in SUPPORTED passes. All 3 failures are translated cells.
- **All 5 refinements are `[x]`.** No queue stop. No trailing perf phase ran: changelog.md has no `## Perf` entry and there is no `perf_experiments/` directory.
- **Most important finding:** both supported_fails are a single cluster: INTERLEAVED inputs with a large per-device row count (M ≥ 8192). The interleaved path budgets CBs as if it had the whole L1. It ignores the per-call exchange-scratch tensor, and that scratch grows linearly with M. Golden misses this because no tagger tracks M and no INPUTS shape has more than 4096 rows.
- **Op-level or framework-level?** The bug is in the op. Its root causes are framework-level:
  - a planner rule that only covers CBs, not per-call L1 tensors;
  - a refinement that widened SUPPORTED onto cells it knew failed;
  - a reference that never states that L1 allocation is lockstep across cores.
- **The third failure is a harness artifact**, not an op failure.

## 1. Golden coverage → `eval/golden_tests/distributed_rms_norm/feature_spec.py`

### 1a. Axis-blind gap: per-device row count M (large-prefill interleaved)
- **What:** The failing cells are `test_ds_distributed_norm[random_weights-prefill-{8192,131072}]`, per-device `(1,1,M,1792)` bf16, INTERLEAVED, tile_aligned, rank 4. Every one of their axes is in TARGET and SUPPORTED. The discriminating facet is M, which no tagger captures:
  - At decode, 128 and 1024 rows the same test passes.
  - The largest golden shapes are 4096×1024 and 2048×2048, and they pass.
  - Golden's projection therefore looks "covered" while the region fails.
  - 32768 passes, so the failure is not monotone in M. Two separate L1 limits are involved (§2a).
- **Evidence:**
  - `golden_blind_final/test_results.json`, prefill-8192: "Statically allocated circular buffers … clash with L1 buffers on core range [0-1 - 10-9]".
  - prefill-131072: "Not enough space to allocate 14751744 B L1 buffer across 4 banks". 4 banks means the scratch tensor sharded over the port cores.
  - `changelog.md:170` says the failure also hits "fp32_dest_acc_en=True for (1,1,8192,3584)", so it is independent of the precision axes.
- **Recommendation:** Add two `LOOSE_CASES` entries, both pinned to `_PHASE0`:
  - `{"inputs": ((1, 1, 8192, 1792),), **_PHASE0, "fp32_dest_acc_en": False}` reproduces the CB-vs-scratch clash at the translated repro point.
  - `{"inputs": ((1, 1, 65536, 1792),), **_PHASE0}` is the scratch-overflow boundary. The scratch is linear in M: about 3.69 MB per port at 131072 rows, so about 1.84 MB at 65536, which exceeds the ~1.46 MB bank. The boundary is extrapolated, not measured.

  Do not expand INPUTS: a large-M entry would multiply the 7776-cell cartesian.

  Promoting M to an `INPUT_TAGGERS` axis is optional and low priority. The op does switch regime on M (R1 one-round vs R2 multi-round, `_fit_rounds`), so a tagger would have a structural basis. But `incremental-verifier.md:215` forbids taggers that "bucket shapes by failure mode". Loose cases catch this without that conflict.
- **Confidence:** high for 8192×1792. Medium for the 65536 boundary.

## 2. SUPPORTED honesty → `distributed_rms_norm.py` `SUPPORTED` / `EXCLUSIONS`

### 2a. supported_fail = 2, one cluster (INTERLEAVED ∧ large M), so **fix, do not demote**
- **What:** Two L1 accounting bugs on the interleaved path, at the same site:
  1. **CB budget ignores the scratch.** `distributed_rms_norm_program_descriptor.py:705-710`: only the sharded path budgets against `_l1_free_below_buffers()`. The interleaved path uses the fixed `WORKER_L1_BUDGET` (1 MiB), even though the scratch at `:691` was already allocated in lockstep and lowered every core's CB ceiling.
  2. **Scratch is unbounded in M and checked too late.** `G·slot_bytes` (`slot_bytes = 128·num_blocks·max(round_recs)`, `:499`) scales with all rows of the call. The `P_scratch` check (`:715`) runs *after* `_l1_sharded_tensor` allocates it (`:691`), so the allocator throws before the op's own guard can.
- **Evidence:** both blind failure messages above. `changelog.md:170` gives the diagnosis ("never subtracts the per-call exchange scratch"). `op_design.md:160` names the failure mode as "L1 allocation failure".
- **Recommendation:** Propose a fix refinement:
  - (1) Use `min(WORKER_L1_BUDGET, _l1_free_below_buffers())` on the interleaved path too. This is a one-line change; the knob solve already exists.
  - (2) Bound the scratch per round. The slot is already round-major since R3, so it can be sized to `(1+ROUND_LOOKAHEAD)` rounds with slot reuse after the ready fence. Alternatively, land records in DRAM.
  - (3) Until then, move the `P_scratch` check ahead of the allocation so the error is clean.

  Demotion is not expressible: no axis separates these cells, and the verifier policy says OOM cells stay failing and become refinements.
- **Confidence:** high.

### 2b. xfail_wrong_mode = 1: harness artifact, not op dishonesty
- **What:** `test_translated.py::test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc` behaves correctly. It expects a raise, and the op raises `ExcludedCell`, which subclasses `NotImplementedError` (`_op_contract.py:23-32`). The run then fails in L1-profiling teardown because the refused call allocated nothing.
- **Evidence:**
  - `eval/l1_profiling.py:85`: "L1 profiling returned peak_total=0; expected a positive allocation peak".
  - It was flagged at Phase 0 in `incremental-verifier_breadcrumbs.jsonl:3` and in `changelog.md` Phase 0 ("harness L1-profiling artifact on a correct refusal"), and it is still unfixed in blind.
- **Recommendation:** In `capture_for_item`, skip the positive-peak assertion when the test passed through a `SupportRefusal`, or for tests with no graded allocation. Alternatively, make `verify_supported` classify this signature as `xfail_expected`. No op change.
- **Confidence:** high.

### 2c. xpass_drift = 0
No under-claim. The 24 `infeasible_skipped` cells (FLOAT32 HEIGHT_SHARDED at W=8192) are harness buffer OOMs, which is correct.

## 3. Helper / reference docs

### 3a. Missing invariant: L1 tensor allocation is lockstep across all cores (absence)
- **What:** No reference states this rule: an L1 tensor sharded on a few cores reserves its address range on *every* core's bank, which lowers every core's static-CB ceiling. The verifier reasoned the opposite.
- **Evidence:**
  - `verification_report.md:162`: "about 1.4 MB of CB space is free on workers, since port scratch never lands on workers".
  - Contradicted by the blind clash on "core range [0-1 - 10-9]" (worker cores).
  - The implementer only learned the rule in R1 (`changelog.md:74-75`) and applied it to the sharded path alone.
  - Grep of `.claude/references/` for lockstep / top-down / allocated-L1-buffer budgeting: none in `l1-footprint-discipline.md` or `ttnn-cb-memory-fundamentals.md`.
- **Recommendation:** Add to `l1-footprint-discipline.md` (budget section): "L1 buffers are allocated top-down at one address on every bank; any per-call L1 tensor, even one sharded on a few cores, lowers the CB ceiling of every core. Budget CBs against the allocator (`GetMemoryView`), never a fixed constant."
- **Confidence:** high.

### 3b. `ReduceFp32Mode::Accurate` with a 16-bit DEST silently yields inf
- **What:** The helper neither documents nor rejects Accurate + Float32 input CB + `fp32_dest_acc_en=False`. The SFPU path reads DEST as fp32, and every output was inf.
- **Evidence:**
  - `changelog.md:161`: "on a 16-bit DEST it reads the DEST as fp32 and every output came back inf".
  - The workaround is in `kernels/distributed_rms_norm_compute.cpp:87-89`.
  - The `ReduceFp32Mode` doc (`kernel_lib/reduce_helpers_common.hpp:11-17`) and `is_sfpu_reduce_path` (`:40-70`) never mention `DST_ACCUM_MODE`.
- **Recommendation:** Add `static_assert(fp32_mode != Accurate || DST_ACCUM_MODE, "Accurate fp32 reduce needs fp32_dest_acc_en")` in `reduce()`, or fall back to Fast. Add a doc line: "Accurate requires a 32-bit DEST."
- **Confidence:** high.

### 3c. `post_reduce_op` contract is silent on the packer reduce edge mask
- **What:** The reduce helper programs the pack edge mask, which keeps only the reduced row or column. A post-op that moves data, such as `transpose_dest`, has its result zeroed at pack. The design relied on that pattern.
- **Evidence:**
  - `ttnn-implementer_breadcrumbs.jsonl:9` ("post_reduce_op docs do not say the reduce helper programs the packer reduce edge mask").
  - Contract text: `reduce_helpers_compute.hpp:514-519`. Mask programmed at `reduce_helpers_compute.inl:982`.
  - The kernel had to call raw `llk_pack_reduce_mask_clear()` (`compute.cpp:91-98`).
- **Recommendation:** Add a POST-REDUCE OPERATIONS bullet: "The packer edge mask stays programmed for the call; a post-op that moves data out of the reduced row/col must clear it (`llk_pack_reduce_mask_clear`) or use a helper option." Consider adding that option.
- **Confidence:** medium-high.

### 3d. `blackhole-ccl-kernels.md` §7 omits the call-sized L1 landing zone and how it scales
- **What:** §7 offers two landing options:
  - DRAM, which needs no credits;
  - a fixed L1 ring, which needs credits.

  The design picked a third option it presented as §7-sanctioned: a per-call L1 buffer holding all G slots for the whole call. Its size grows with the payload, and that growth is exactly 2a(2).
- **Evidence:** `op_design.md:149`: "landing zone needs no credits (`blackhole-ccl-kernels.md` §7)". §7 says that only for DRAM landing (`blackhole-ccl-kernels.md:322-326`).
- **Recommendation:** Add to §7: "A call-sized L1 landing zone avoids credits, but its footprint scales with the payload; bound it with a predicate and route beyond the bound to DRAM landing or a credited ring."
- **Confidence:** medium.

### 3e. perf-measure skill reads the wrong CSV column
- **What:** The skill says column 19 is DEVICE KERNEL DURATION. In the current CSV it is FW duration.
- **Evidence:** `incremental-verifier_breadcrumbs.jsonl:2`. The text is still present at `.claude/skills/perf-measure/SKILL.md:144`.
- **Recommendation:** Select the column by header name.
- **Confidence:** medium.

#### Helper gaps (perf)
No trailing perf phase ran, so there is no `### Helper bypasses` table to audit. The one raw-LLK call in the kernels outside helpers (`llk_pack_reduce_mask_clear`, `compute.cpp:94,98`) comes from a queued refinement, not the perf phase, and is covered by 3c.

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| — | — | — | no perf phase | — |

## 4. Agent prompts

### 4a. Planner: size predicates on per-call L1 tensors were justified by "every INPUTS shape fits", with no fallback
- **What:** The design bounded `P_scratch` and the R1 regime only against golden INPUTS. Its failure mode is an allocator error, not a fallback regime or a `validate()` refusal.
  - `incremental-planner.md:193` requires a predicate-guarded dual path, but only for CBs.
  - The scratch is an L1 *tensor*, so the rule never applied to it.
- **Evidence:**
  - `op_design.md:160`: "The largest INPUTS case … is 512 KiB. | L1 allocation failure".
  - `op_design.md:171`: "every TARGET INPUTS shape fits one round".
- **Recommendation:** Extend `incremental-planner.md:193`/`:217` so they apply to every per-call L1 allocation (scratch tensors, inboxes), not just CBs. Every size predicate needs a fallback regime or a pre-allocation refusal. INPUTS' largest shape is not a bound on what callers pass.
- **Confidence:** high.

### 4b. Implementer (refinement): widened SUPPORTED onto known-failing cells and filed nothing
- **What:** Refinement 4 added `fp32_dest_acc_en=False`. That moved the DeepSeek prefill cells from xfail into SUPPORTED. The implementer saw them fail and wrote "not a regression". R5 repeated that. Neither added an `op_requirements.md` entry, so the queue drained with a known over-claim. Separately, the implementer also knew of a failure in an *already*-SUPPORTED region ((1,1,8192,3584), fp32 True) and filed nothing for it either.
- **Evidence:**
  - `changelog.md:170`: "Both were failing before this refinement (UnsupportedAxisValue), so they are not a regression".
  - `changelog.md:211`.
  - `op_requirements.md`: all entries `[x]`, none for L1 fit.
- **Recommendation:** Add to `ttnn-implementer.md`: "If a SUPPORTED widening newly exposes failing cells in any suite, or you find a failure in an already-SUPPORTED region, fix it or append a refinement entry to `op_requirements.md`. 'Previously xfail' is not 'not a regression'." This mirrors the verifier's supported_fail routing (`incremental-verifier.md:215`).
- **Confidence:** high.

### 4c. Verifier: L1-headroom review checked P_resident only, not CBs + per-call tensors
- **What:** The Phase-0 headroom section declared "no corpus or translated shape reaches `P_resident`". It missed the CB/scratch interplay and the scratch's growth in M. The translated prefill-8192/131072 cells were hidden as xfail at the time (fp32_dest_acc_en=False was unsupported).
- **Evidence:** `verification_report.md:159-162`.
- **Recommendation:** In `incremental-verifier.md`, require the L1-headroom review to cover all capacity predicates (CB + per-call L1 tensors). Evaluate them against translated shapes that are currently xfail only on a precision or other axis a queued refinement will add.
- **Confidence:** medium.

### 4d. `--dev`-exclusive mandate is unworkable for fabric (multi-device) ops
- **What:** Both agents report that `--dev` multi-test runs leave Ethernet cores dead, so the next mesh open fails. The mandated `--dev` acceptance run hung until the 10-minute tool timeout.
- **Evidence:** `incremental-verifier_breadcrumbs.jsonl:1`, `ttnn-implementer_breadcrumbs.jsonl:11`.
- **Recommendation:** In the implementer and verifier prompts, for fabric ops: run multi-test suites non-dev and use `--dev` one test per invocation. Separately, have `run_safe_pytest.sh` reset after a `--dev` fabric run.
- **Confidence:** medium.
