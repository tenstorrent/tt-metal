# Self-Reflection: mhc_pre

## Summary
- **Blind pass: 206/206** (196 golden + 10 regression; there are no translated tests for this op). `verifier_report.json`: supported_pass 206, supported_fail 0, xpass_drift 0. SUPPORTED equals TARGET on every axis. The refinement trajectory was clean: the only red phase was R1, at 205/206 (the large-Sinkhorn-logits cell, fixed in R2). There were 2 perf rounds, and both graduated 2 of their 2 ideas.
- **Most important finding:** golden runs **one input draw per cell (seed 0)**, so it could not see this run's recurring bug class. Three separate CB-wait / source-reuse races were caught only by timing shifts or alternating seeds, never by golden. On top of that, a **SUPPORTED cell fails on seed 1**: T=1 decode, bf16 X / bf16 W, post rms 5.08e-4 against a 5e-4 limit. It was recorded in a perf README and never escalated.
- **Framework-level issues found:** the `mcast_pipe` Counter + loopback hang, the numerical-stability reference over-stating FPU matmul precision, and a contradiction in the verifier prompt. The op's own remaining risk is unexplained timing-dependent ~1-ulp output.

## 1. Golden coverage (`eval/golden_tests/mhc_pre/feature_spec.py`)

**G1. The input draw and stale L1 are an axis-blind facet: every golden cell uses seed 0** · confidence: high
- **What:** a race that depends on W/X landing order, or that stale L1 from the previous identically-shaped cell can hide, passes golden. The same is true of a precision cell that only misses on some draws. This run had 3 such races and 1 marginal cell, and golden flagged none of them on its own. The facet is in TARGET, but no tagger can capture it, and it should not become an axis because it is not a code path.
- **Evidence:**
  - `changelog.md` R3b: "Stress, 40 seeds at 256×24576: 1 failure before the fix".
  - `changelog.md` R5: "It worked only because W happened to land before the first X block … golden 202/206" (only once placement shifted timing).
  - `perf-part-optimizer_breadcrumbs.jsonl` 08:55: "fixed bf16-W missing cb_weight wait in pipelined project_block_pieces<1>".
  - `perf_experiments/cross_block_pipeline/README.md:44`: "1x7168 bx_bw on seed 1 … post rms is 5.08e-4 against a limit of 5e-4". It passes at seed 0 in the blind pass.
- **Recommendation:** `run_mhc_pre` already honours `extras["seed"]` (`helpers.py:271`), so the proposed fix costs no harness change. Add alternate-seed `LOOSE_CASES`, listed after the seed-0 cells of the same shape so that stale L1 cannot mask a missing wait:
  - `{"inputs": _case((1, 28672)), "dtype": ttnn.bfloat16, "weight_dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "fp32_dest_acc_en": True, "extras": {"seed": 1}}`. This one fails today, which is the point.
  - The same with `(1,1,640,7168)` fp32 X / bf16 W `seed: 1`, for the W column-broadcast path whose W wait was missing.
  - The same with `(1,1,256,24576)` bf16 X / fp32 W `seed: 1`, for the R3b race shape.
  - Do not add a tagger.

**G2. The doubly-stochastic row clause reports a precision miss as `severity="bug"`** · confidence: med
- **What:** `check_doubly_stochastic` raises one `severity="bug"` error (with pcc=0 and rms=inf) for any of its four clauses. The Phase-0/R1 failure was a 5e-5 absolute excess over the reference's own row error of 0.0672, which is a precision miss. It was classified as `numerical-bug`, and the verifier routing lets that category go to EXCLUSIONS or a demotion (`incremental-verifier.md:215`). The verifier routed it correctly by hand this time.
- **Evidence:**
  - `helpers.py:233-236` (`severity="bug"` for every clause).
  - `golden_refinement_1/test_results.json` `test_large_sinkhorn_logits[T64_nC4096]`: "numerical-bug … pcc=0.000000 rms=inf … max|rowsum-1|=0.0674 (reference 0.0672)".
- **Recommendation:** keep `bug` for the negative-entry and column-sum clauses. Raise the row-sum-excess clause as `severity="precision"`, with the excess recorded as the metric. Also consider making `ROW_SUM_SLACK` relative (for example 1e-3 × the reference row error): this gate alone drove R2, which first made the fp32 path +50% slower (378 → 567 µs).

**G3. Only one output's metrics reach `test_results.json`** · confidence: low
- **What:** the op returns y, post and comb, but each row carries a single pcc/rms, and it looks like the last `check_output` (comb). The output that actually came close to its limit (post, 5.08e-4) is invisible on the dashboard.
- **Evidence:**
  - `eval/metrics.py:419-425` records per call, and `metrics_plugin.py:65` says "uses the last value seen per key".
  - The bf16 blind rows show rms ≈ 2.9e-4, which is consistent with comb, not with bf16 y (~1.6e-3 per changelog R4).
- **Recommendation:** framework change, so propose it rather than apply it: record the metrics for each output, or the worst ratio of metric to limit across outputs.

## 2. SUPPORTED honesty (op file `SUPPORTED` / `EXCLUSIONS`)
The blind verifier shows **0 supported_fail and 0 xpass_drift**, and SUPPORTED == TARGET. There is no over-claim or under-claim in the measured universe. Two latent issues:

**S1. The bf16 X / bf16 W coefficient gate is marginal (hidden over-claim)** · confidence: med
- **What:** every bf16-stream cell sits at a comb rms of ~2.9e-4 against the 5e-4 gate. The FPU's in-tile accumulation floor, measured at ~2.5–2.9e-4, eats most of that budget. At T=1 (only 4 post values) a different draw goes over. This affects the whole bf16 X region, most at small T.
- **Evidence:**
  - Blind rows `X17x512 … BFLOAT16 … BFLOAT16` rms 2.907e-4 and `X1x1x512x10240 … BFLOAT16` 2.999e-4.
  - `helpers.py:202`: "set from host emulation, not yet calibrated on device".
  - Verifier breadcrumb: "The 5e-4 coeff gate is therefore marginal".
- **Recommendation:** do **not** demote. Decide between two options:
  - **Fix:** bring the exact-grid projection from R2 to the bf16-W path. This costs perf.
  - **Recalibrate:** set `("coeff", bf16)` on device, for example at 3× the measured floor.

  Add G1's seed-1 case either way, so the decision is visible.

**S2. Unexplained timing-dependent output in the shipped op** · confidence: low
- **What:** Phase 0 claims "Repeat calls are bitwise-identical", but Perf 2 found output that changes by ~1 ulp when only timing is perturbed. Separately, R5 left a deterministic failure unexplained.
- **Evidence:**
  - `perf-part-optimizer_breadcrumbs.jsonl` 10:57: "base itself changes identically when only timing is perturbed … pre-existing timing-dependent ~1ulp behaviour in the op".
  - The later 11:48 entry attributes the comb part of this to sfpi emitting fused vs unfused Newton MADs (now pinned in v6). The **y "1 bf16 ulp 1 row"** part is not explained.
  - `changelog.md` R5: "A wait placed later … still failed deterministically … I did not pin down the thread-level mechanism".
- **Recommendation:** open a small investigation refinement. Run a repeat-call bitwise check across alternating seeds and NoC placements on y. This is not a demotion.

## 3. Helper / reference docs

**D1. `mcast_pipe` Counter mode + loopback expects one ack too many, which hangs** · confidence: med
- **What:** on a loopback send, `SenderPipe::send` passes `num_dests_incl_` (receivers + 1 self-copy) to `signal_ready_`. The Counter path is a non-`INCL_SRC` `inc_multicast`, so the source never acks, and `fence_` then waits `async_atomic_barrier()` for one ack that never arrives. The run hit this hang and switched to Flag mode. It appears only as prose in the changelog, not in any bypass table or helper doc.
- **Evidence:**
  - `kernel_lib/mcast_pipe.inl:95-97`: `mcast_dests = loopback ? num_dests_incl_ : num_dests_excl_` → `signal_ready_(loopback, mcast_dests)`.
  - `:179`: `inc_multicast(..., mcast_dests)` with no loopback handling.
  - `:218-221`: the Counter path waits the atomic barrier.
  - `changelog.md` Perf 1: "A Counter hangs in the send's atomic barrier on the looped-back root copy".
  - Breadcrumb 08:36: "nohs Counter-mode hung (loopback atomic ack)".
- **Recommendation:** fix the helper (pass `num_dests_excl_` to the Counter increment, or do a local `inc` for self). If it is left as is, add a `static_assert` or a `mcast_pipe.hpp:57-58` doc line: "Counter is not supported with a loopback (in-box, src≠dst) sender."

**D2. The numerical-stability reference over-states FPU matmul precision** · confidence: med
- **What:** the reference says bf16 and tf32 operands enter srcA/srcB losslessly (10-bit tf32) and that fp32-DEST accumulation is "generally safe at any depth". The run measured otherwise, over about 15 probes:
  - The FPU keeps ~9 mantissa bits of an fp32/tf32 operand at HiFi4.
  - The in-tile 32-term dot product rounds to ~11 bits below its largest product, even for exact bf16 operands.
  - Only DEST accumulation *across* `matmul_tiles` calls is ~fp32.

  This silent rule cost R1 and R2 their design work, and it also misled the golden author (`helpers.py:46-49`, `:194-196`: "exact in the FPU source registers … carry ~fp32 accumulation noise only").
- **Evidence:**
  - `.claude/references/numerical_stability_analysis_reference.md:34,42,178`.
  - `changelog.md` R2 diagnosis: "The FPU's in-tile (32-long) matmul dot product rounds its sum to ~11 bits below the largest product".
  - Probes `ttnn/ttnn/bringup/mhc_pre_ttnn/tests/unit/probes/probe_009–015.py`.
- **Recommendation:** add to §2.1/§3.1 of the reference: "matmul_tiles' in-tile 32-term dot product is not fp32-exact (≈11 bits below the max product, RN), and fp32/tf32 operands keep ≈9 bits at HiFi4. For fp32-exact projections, use a hi/lo or exact-grid split (see mhc_pre R1/R2)." Then correct the `helpers.py` precision comment.

**D3. Absent rule: a raw replacement for a helper inherits the helper's `cb_wait_front`** · confidence: med
- **What:** two of the three races were a raw `matmul_tiles` path that skipped the in1 (`cb_weight`) wait that `matmul_block` does internally. The third published a CB page while a multicast was still reading it. No reference says "when you replace a helper call with raw LLK, you now own every wait/pop it did". `ttnn-implementer.md:196` covers only the opposite direction (policies that hand lifecycle back).
- **Evidence:**
  - `changelog.md` R5 "nothing waited for the resident `cb_weight` before the split projection's matmul read it".
  - Breadcrumb 08:55 (the same bug on the bf16-W pipelined path).
  - Now `mhc_pre_compute.cpp:1367,1396`.
- **Recommendation:** add one line to `.claude/references/cb-debugging-strategy.md` and to the raw-LLK deviation guidance: "Replacing a helper call with raw LLK moves its internal waits to you. List the helper's wait/pop per operand (see its `.inl`) and reproduce each one."

**D4. `InputTileMapping` Row/Col naming misled the planner** · confidence: low
- **What:** `op_design.md:343,363` picked `InputTileMapping::Col` for a `[1, n]` per-stream operand, next to `BroadcastDim::Col`. The correct choice was `Row` ("indexed by column only"). The implementer caught it.
- **Evidence:**
  - Implementer breadcrumb: "Col indexes by grid ROW (c), but the operand must be indexed by grid column (stream i) -> Row mapping".
  - The docs are at `kernel_lib/eltwise/api/chain.hpp:219-224`.
- **Recommendation:** add a note to `chain.hpp`: "The mapping names the operand's *shape* ([1,Wt] = Row), not the axis it varies along; it is independent of `BroadcastDim`, so `Row` mapping + `BroadcastDim::Col` is a common, valid pair."

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `matmul_block` (in0 tile base) | capability | **missing** | in1 has a per-K-block base shift (`matmul_block_helpers.hpp:223-229`, `In1BaseOffsetFn`), but in0 has none: `in0_index_subblock_offset = 0` (`.inl:293`), `In0SourceFn` swaps only the CB id (`hpp:216-221`), and `NoWaitNoPop` is static_asserted off for in0 (`.inl:117`). **The two ns are ~equal (104200 helper-schedule vs 104500 raw)**: nothing was gained, and the raw path then shipped a missing `cb_weight` wait (breadcrumb 08:55). | Add an `In0BaseOffsetFn` symmetric to in1, applied as an in0 index shift with the wait/pop lifecycle kept on the bound CB (wrap-aware, because the next block sits behind the resident one). |
| SFPU / sfpi (`SFPLOADMACRO`, pinned LREGs) | capability | **missing** | No kernel_lib or sfpi construct issues `SFPLOADMACRO`, pins LREGs, or programs `SFPCONFIG` templates. In-tree uses are private to LLK ckernels (`tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_recip.h`, `ckernel_sfpu_exp.h`, …). The win is real: 3490 (best sfpi) → 2730 ns. | Not a small doc fix. Propose a narrow LLK-level "macro pass" builder. Until then, the generated rule-checked schedule (`perf_experiments/sinkhorn_sfpu_fast/bench/gen_sinkhorn_lm.py`) is the reference pattern. |
| `mcast_pipe` (Counter + loopback) | *(unrecorded)* | **missing / bug** | The hang was recorded only in Perf 1 prose. The call site switched to Flag, so it is not a bypass and appears in no table (see D1, `mcast_pipe.inl:95,179,221`). | Fix the ack count. Until then, have the coordinator record helper *defects* that forced a mode change in the bypass table. |

No other unrecorded perf-phase raw LLK was found. The fast bias fill (writer word stores) has no helper counterpart, and all refinement-era raw paths are listed in `mhc_pre_compute.cpp:43-75`.

```cpp
// confidence: low (one call site, mhc_pre_compute.cpp:1394-1397)
struct NoIn0BaseOffset { ALWI uint32_t operator()(uint32_t /*block*/) const { return 0; } };
template <..., typename In0BaseOffsetFn = NoIn0BaseOffset, ...>
ALWI void matmul_block(..., In0BaseOffsetFn in0_base_offset_fn = {});
// call site (pipelined proj(b+1), X(b+1) sits extent tiles behind the front):
matmul_block<..., InputPolicy::WaitAndRetainOnLastBlock, ...>(
    cb_x_resident, cb_w_matmul, cb_partial, cb_partial, shape, ..., /*in0_base_offset_fn=*/
    [=](uint32_t) { return v.ring_offset; });   // helper waits front+offset+block, never pops past it
```

## 4. Agent prompts (`.claude/agents/*.md`)

**P1. A perf subagent found a failing SUPPORTED cell, and nothing escalated it** · confidence: high
- **What:** the part-optimizer recorded a base-level golden miss on an in-TARGET, SUPPORTED cell (seed 1) as "not a golden seed" in its README. The coordinator's changelog reports "bitwise identical to base" and omits it. There is no route from a perf-phase correctness observation into the refinement queue.
- **Evidence:**
  - `perf_experiments/cross_block_pipeline/README.md:44-45`.
  - Breadcrumb 09:02: "golden pass except pre-existing 1x7168 bf16/bf16 seed1 post rms 5.08e-4 (base same)".
  - `grep 5.08 changelog.md verification_report.md` finds nothing.
- **Recommendation:** add to `perf-part-optimizer.md` "Correctness is the only pass/fail": "Report any *base-level* correctness miss on a SUPPORTED cell (any seed) to the coordinator as a finding." Add to `perf-coordinator.md`: "Copy such findings into the changelog under `### Correctness findings (base)`."

**P2. The implementer's race check (dev vs non-dev) did not catch any of the 3 races** · confidence: med
- **What:** `ttnn-implementer.md:240-244` gives "If `--dev` passes but non-dev fails, a race condition exists" as the race detector. The races in this run were caught by alternating-seed stress and by timing perturbation (placement knobs, zones) instead. R5 and Perf 1 adopted "alternating seeds (stale L1 cannot mask a race)" on their own.
- **Evidence:** `changelog.md` R3b ("40 seeds"), R5 ("Alternating-seed stress (stale L1 cannot mask a race)"), and Perf 1 ("alternating seeds").
- **Recommendation:** add to the correctness-modes section: "Also run each multi-core / multicast path back-to-back with two different seeds in one process; a missing wait that stale L1 satisfies only fails when the data changes."

**P3. The verifier prompt contradicts itself on `supported_fail`** · confidence: high
- **What:** the routing row keeps precision and OOM `supported_fail` cells failing as refinements. The report template says they must be 0 to ship.
- **Evidence:**
  - `incremental-verifier.md:215` vs `:599`: "supported_fail: 0   (must be 0 to ship)".
  - Verifier breadcrumb: "contradictory for a precision-class regression test failure".
- **Recommendation:** change `:599` to "supported_fail: 0 (except `numerical-precision`/`OOM` cells named in a refinement's Done-when)".

**P4. The planner picked a chain enum from its name, not its doc** · confidence: low
- **What:** the D4 mis-mapping reached `op_design.md` unverified.
- **Evidence:** `op_design.md:363` cites `chain.hpp:232` (the bare enum) rather than the doc block at `:219-224`.
- **Recommendation:** add to `incremental-planner.md`: "For every `InputTileMapping` / `BroadcastDim` in the realization table, state the operand's tile shape ([1,Wt] / [Ht,1] / [Ht,Wt]) next to the enum."
