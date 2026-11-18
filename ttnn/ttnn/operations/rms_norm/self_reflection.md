# Self-Reflection: rms_norm

## Summary
Final blind: 5547 passed, 1367 xfail (expected), 1 failed, 33900 INVALID-skipped, 21 infeasible. Verifier: `supported_fail=1`, `xpass_drift=0`, `invalid_unexpected=2`. The only failure is an L1-OOM on one large WIDTH_SHARDED loose case. The main framework-level findings are process and doc gaps that agents logged as friction, not kernel correctness. No trailing perf tournament ran: there is no `## Perf` entry and no `### Helper bypasses` table. Refinements 3 and 6 were ordinary perf-type refinements. All refinements are `[x]`.

## 1. Golden coverage (feature_spec.py)
**Finding 1: no discriminating cluster, so nothing to widen.**
- **What:** There is one failure, so no cluster. Its shape `(13,777,1023)` is already pinned in golden at `feature_spec.py:417`, so golden did catch it. It is a singleton, not a crash or NaN, so I propose no new `LOOSE_CASES`.
- **Evidence:** `test_op_loose[13x777x1023-...-memory_layout=WIDTH_SHARDED-rank=3]`, "Statically allocated circular buffers ... clash with L1 buffers".
- **Recommendation:** None.
- **Confidence:** med.

**Finding 2 (axis-blind facet, low confidence):** WIDTH_SHARDED cells have no facet for per-core shard L1 footprint, which depends on the logical shape (`prod(batch)×H`) and the core grid. Only one case (`13x777x1023`) exercises large-footprint sharded W-split in golden. Consider promoting a "shard bytes per core" bucket to a tagger axis, or pinning one more mid-size WIDTH/BLOCK sharded case with fp32_dest_acc_en=False.
- **Evidence:** the OOM above (CB region ends at 403392 against an L1 buffer at 241664).
- **Confidence:** low.

## 2. SUPPORTED honesty
**Finding 1: `supported_fail` = 1 (over-claim, WIDTH_SHARDED, bf16, fp32_dest_acc_en=False, w_non_aligned, rank 3, TILE gamma).**
- **What:** The op declares WIDTH_SHARDED as supported, but the program descriptor statically allocates CBs beyond the available L1 for this large shape.
- **Evidence:** `failure_category: OOM`, `program.cpp:2691`.
- **Recommendation:** Prefer a fix. Cap or stream the CB depths in `rms_norm_program_descriptor.py` when the sharded slice is large (see `l1_ledger.md`). If that is not feasible, add a validate()-time L1 estimate that raises `ExcludedCell`, or demote this region to `EXCLUSIONS`. Do not demote all of WIDTH_SHARDED, because 5.5k other cells pass.
- **Confidence:** med.

**Finding 2: `invalid_unexpected` = 2.**
- **What:** `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w72_c2_nonaligned|w200_c3_nonaligned-bfloat8_b]` hit the INVALID combination bf8b ∧ w_non_aligned. They are translated tests, which do not skip INVALID cells, so they show as xfail.
- **Recommendation:** Treat this as a harness artifact. Have the verifier categorizer treat translated cells that match INVALID as expected, or mark those translated params skip. It is not an op bug.
- **Confidence:** med.

`xpass_drift` = 0: no promotions are needed.

## 3. Helper / reference docs
**Finding 1: the design's helper mapping was written without checking the helper contracts.**
- **What:** The design specified `DestReuseBinary` with Row broadcast for gamma, but the helper takes no `BroadcastDim`. It also listed a separate `cb_accum` although `Accumulate::at` uses the output CB. It used pre-DataflowBuffer signatures.
- **Evidence:** `agent_logs/ttnn-implementer_breadcrumbs.jsonl` (friction, `op_design.md:220 / chain.hpp DestReuseBinary`). Refinement 6 later lists "needs a new eltwise op for Row-bcast DestReuse" as the unfinished next step, so this is a real helper gap.
- **Recommendation:** Add a docstring line in `chain.hpp` `DestReuseBinary`: "no broadcast dim; Row/Col bcast with DEST reuse is unsupported". Also state "Accumulate target == output CB" at `Accumulate`. If a bcast variant is wanted, treat it as a `missing` helper feature. This is a single call site, so confidence is low.
- **Confidence:** med for the doc line, low for the feature.

**Finding 2: a blanket "UnpackToDestFp32 for all fp32 CBs" rule contradicts FPU-consumer limits.**
- **What:** The design said to set `UnpackToDestFp32` on every fp32 CB, but FPU consumers (square, reduce, bcast mul) cannot read such CBs.
- **Evidence:** breadcrumb `op_design.md:82 precision caps`.
- **Recommendation:** In the numeric-format reference, state that `UnpackToDestFp32` applies only to SFPU-consumed CBs.
- **Confidence:** med.

**Finding 3: bf16 DEST sequential accumulation of Σx².**
- **What:** It was the root cause of a precision bug, and the debugger was needed to find it.
- **Evidence:** commit `2a6c3a8c1a1` "bf16 DEST sequential accumulation of Sum x^2 is the root cause".
- **Recommendation:** Add a note to the reduce/numeric-formats references that a long sum with `fp32_dest_acc_en=False` accumulates in bf16 DEST and needs a fp32/L1 accumulate.
- **Confidence:** med.

#### Helper gaps (perf)
No perf tournament ran, so there is no `### Helper bypasses` table to audit. I did not grep the kernels for unrecorded raw-LLK bypasses. Refinement 6's note that the gamma multiply cannot be fused because "needs a new eltwise op for Row-bcast DestReuse" is the only perf-related helper gap in the record. It is covered in Finding 1 and is likely `missing`.

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| DestReuseBinary | no Row bcast | missing (unchecked in `.inl`) | op_requirements.md R6 Outcome; breadcrumb | add a BroadcastDim to the spec, or document its absence |

## 4. Agent prompts
**Finding 1: the entry-point contract for sharded TARGETs is unspecified.**
- **What:** The golden harness passes `memory_config=` on sharded cells, but no planner, op_template or verifier prompt says the entry point must accept it. 4786 cells failed as TypeError before the signature was fixed (commit `69750bd67f5` "memory_config kwarg fix").
- **Evidence:** the friction breadcrumb citing `eval/golden_tests/rms_norm/helpers.py:236`.
- **Recommendation:** Add a line to the planner/op_template prompt: "if `memory_layout` is a TARGET axis, the entry point must accept `memory_config` for output placement".
- **Confidence:** high.

**Finding 2: the golden run takes about 15 minutes, which exceeds the 10-minute bash maximum.**
- **What:** The verifier had to run it in the background, which conflicts with the guidance never to run safe_pytest in the background.
- **Evidence:** the same breadcrumb.
- **Recommendation:** Tell the verifier prompt to run the golden suite in chunks with `--precompile`, or document the expected duration.
- **Confidence:** med.

**Finding 3: the sandbox pre-commit hooks need network access and time out.**
- **What:** Agents had to bypass the hooks.
- **Evidence:** breadcrumb `op_design.md:82 precision caps`.
- **Recommendation:** Document the `--no-verify` policy or pre-install the hook environments.
- **Confidence:** low.
