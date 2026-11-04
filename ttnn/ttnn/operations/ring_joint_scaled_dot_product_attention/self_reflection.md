# Self-Reflection: ring_joint_scaled_dot_product_attention

## Summary
- **Final blind pass: 1815 passed / 0 failed / 0 hangs.** The 5984 skips all match an `INVALID` entry. `verify_supported` reports 0 supported_fail and 0 xpass_drift. `SUPPORTED == TARGET` and `EXCLUSIONS = []`. There was no translated suite in this run: the blind pass is golden + regression + ring_mock + fabric_configs.
- All refinements are `[x]`. Two perf rounds ran, and the focus case went from 121.3 ms to 42.6 ms. They ran **in the middle of the queue**, not after it, because the Refinement 1 agent deleted Refinements 2–6 from `op_requirements.md`.
- **Most important finding (framework / process):** the implementer prompt has no rule against an agent editing other entries in the queue, and the orchestrator has no guard against the queue shrinking. A run operator had to restore the queue by hand (`fd1e683acd5`).
- Golden is green, but two regions are never exercised:
  - every Ring / G ≥ 3 combination outside ring_mock's four hand-picked shapes;
  - the "wholly padded device" branch in the descriptor.
- The perf phase's helper feedback mostly holds. One claim is partly refuted: the reduce helper *does* have a seeded MAX accumulator. Two raw-LLK bypasses are not recorded in any bypass table.
- Overall the problems are framework-level (queue integrity, gate environment, helper DEST ownership), not op-level.

## 1. Golden coverage (`eval/golden_tests/ring_joint_scaled_dot_product_attention/feature_spec.py`)

There are no blind failures to cluster, so both findings are proactive gaps in TARGET that golden never exercises.

**1a. Ring with a relay (G ≥ 3) is covered only by 4 shapes. MLA / GQA / cross / bf8b-Q / bf16-DEST never run on it in the blind pass.**
- **What:** the system mesh is 2×2, so G = 2 and the golden matrix runs Linear only. The blind pass had 1697 golden Linear cells and only 1 Ring cell, the perf loose case. All other Ring coverage comes from `test_ring_mock.py` `SHAPES` (4 shapes: MHA, replicated joint, sharded joint, MQA causal-balanced). The relay path (multi-hop forwarding) only exists at G ≥ 3, and G is not a tagger axis. So this is an **axis-blind gap**: the topology=Ring tuple looks "covered" for every kv_heads_mode.
  - In the blind ring_mock cells: `kv_heads_mode` = {mha: 72, mqa: 24}, `attention_kind` = {self: 96}, `dtype` = {BF16: 96}, `fp32_dest_acc_en` = {True: 96}.
  - So 0 Ring cells cover MLA, GQA (1 < hk < hq), cross-attention, bf8b Q, or bf16 DEST (apart from the one focus case).
- **Why it matters:** Refinement 2 changed the transport item structure precisely for MLA. Quote from changelog.md:320: "MLA's one K head rides with V head 0 … per-part chunk counts (k_chunks, v_chunks)". On a G ≥ 3 ring those items are *relayed*, and no blind cell relays an item with k_chunks ≠ v_chunks. The unit tests (`…_causal_heads.py`, `…_joint_cross.py`) do cover these at G = 4, but they are not part of the blind measurement.
- **Recommendation:** add 2–3 entries to `test_ring_mock.py:47` `SHAPES` (preferred: they inherit every ring stand-in mesh and both link counts), or pin them as `LOOSE_CASES` with `extras={"mesh_shape": (1, 4), "fabric_config": FABRIC_2D_TORUS_XY}`:
  - MLA causal, contiguous: `((1,8,128,320),(1,1,128,320),(1,8,128,128))`, `mask_mode="causal"`, `is_balanced=False`;
  - cross: `((1,16,64,64),(1,16,512,64),(1,16,512,64))`;
  - GQA 2:1 at bf16 DEST with bf8b Q: `((1,8,256,128),(1,4,256,128),(1,4,256,128))`, `dtype=bfloat8_b`, `fp32_dest_acc_en=False`.
- **Also consider:** recording group size (or "relay hops > 0") as an observed facet, so a later run can see this hole in the axis model.
- **Confidence:** med. The behaviour likely works (the unit tests pass), but the blind pass cannot show it.

**1b. The "wholly padded device / key segment" branch never runs.**
- **What:** `helpers.SEQ_PAD = 40` is fixed (`helpers.py:48`), and the smallest K `n_local` in `INPUTS` is 64 (`feature_spec.py:79`). So padding never covers a whole device's key segment. The descriptor has a dedicated branch for that case, `if vr == 0: return` (`…_program_descriptor.py:984`). That branch drops the segment and its landing wait, and it changes which segment is the unit's first processed block. That first block is exactly what the "finite-max cap" relies on (changelog.md:318: "The local segment comes first and its tile 0 is visible to every query"). The relevant facet is pad_rows ≥ n_local, and no axis captures it (**axis-blind gap**). The `seq_padding="padded"` tuple therefore looks covered.
- **Recommendation:** propose two `LOOSE_CASES` on the system mesh (G = 2), using `extras` the runner already supports (`helpers.py:445`):
  - `{"inputs": ((1,2,128,64),)*3, "mask_mode": "none", "seq_padding": "padded", "cluster_axis": 0, "topology": Linear, …, "extras": {"logical_n": 100}}`. Here device 1 holds only padding and device 0 ends in a partial tile.
  - The same case with `mask_mode="causal"`.
- **Also consider:** a third `seq_padding` value, `"padded_device"`, if the boundary keeps mattering.
- **Confidence:** med.

## 2. SUPPORTED honesty (op file `SUPPORTED` / `EXCLUSIONS`)

- **0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode, 0 invalid_unexpected** (`golden_blind_final/verifier_report.json`). `SUPPORTED` equals `TARGET` and `EXCLUSIONS` is empty, so there is nothing to demote or promote.
- **One qualification (no edit proposed):** `topology=Ring` is claimed for every combination, but the blind pass verified it only on the ring_mock subset (§1a). The claim is honest wherever it was measured, and unmeasured elsewhere. Closing §1a closes this. **Confidence:** high.
- **Note on Phase 0:** golden_phase0 shows 74 `failed` cells (`L1ProfilingError: … peak_total=0`). These were harness-instrumentation failures, not op failures (see §4b). They disappeared from Refinement 1b onward.

## 3. Helper / reference docs

**3a. `perf_instrumentation.hpp` does not warn about zone-hash collisions in fabric ops.**
- **What:** after a source-line shift, a compute zone's 16-bit source-location hash collided with an ERISC firmware zone. That broke every `TT_METAL_DEVICE_PROFILER` run until the zone was moved.
- **Evidence:** changelog.md:448, "Source location hashes are colliding". `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp:1-20` covers the marker budget but not collisions.
- **Recommendation:** add one line to the header comment: "Zone ids are a 16-bit hash of name+file+line; in fused-CCL ops they can collide with ERISC firmware zones (post-processing aborts with 'Source location hashes are colliding'). Rename or move the zone to fix it."
- **Confidence:** med.

**3b. `--dev` cannot be used on large compute kernels, and the docs don't say so.**
- **What:** watcher instrumentation overflowed the kernel config buffer, so the Refinement 3 hang was triaged in default mode.
- **Evidence:** changelog.md:396, "`--dev` overflows the 70 656 B kernel config buffer on this kernel". The hang guidance in `.claude/CLAUDE.md` recommends `--dev` with no caveat.
- **Recommendation:** add to the hang-triage section: "If `--dev` fails with a kernel-config-buffer overflow, the kernel is too large for watcher instrumentation; triage in default mode, or with `TT_METAL_WATCHER_DISABLE_*` subsets."
- **Confidence:** med.

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::Exp<>` | capability: no scale argument; no SFPLOADMACRO bf16 form | **missing** (both parts) | `eltwise/unary/math.inl:29-31` forwards only `approx` to `exp_tile`, while the LLK has `scale_en` / `scale` (`api/compute/eltwise_unary/exp.h:67-72`). The pipelined form is LLK-level and exists nowhere. Measured 64 250 → 34 660 ns per 128-tile block. | Add a `ScaleBf16` template parameter to `Exp<>` that forwards `scale_en=true`. Separately, consider upstreaming the bit-identical LOADMACRO `exp_21f` from `kernels/ring_joint_sdpa_scaled_exp.hpp` as an LLK fast path (it has its own domain caveats: `scale·x < 88.7`, NaN not preserved). |
| `eltwise_chain` | capability: no SFPU op on the PACK thread | **missing** for `eltwise_chain`, but the mechanism already exists in the library | `matmul_block_helpers.hpp:296`: "fuse an SFPU activation on the PACKER thread at the last-block pack". It is limited to the fixed `KernelActivation` set and to matmul output, so it cannot take `ScaledExp` after a bcast-sub. Measured 96 530 → 66 050 ns per block (shared with the next 3 rows). | Generalise the matmul pack-thread hook into a chain element (`OnPack<Op>`), including the one `STALLWAIT(SFPU, MATH)` + RWC reset per DEST section that the raw code does by hand (`…_compute.cpp:283`). |
| `matmul_block` (Q·Kᵀ, P·V) | capability: per-row sub-blocks inside a shared DEST section; out-of-order packs into a held-`wr_ptr` block | **missing** | The helper "Uses 4-phase DST management (tile_regs_acquire/commit/wait/release)" (`matmul_block_helpers.hpp:273`). `caller_owns_pack_target` covers only reserve/push, not DEST ownership. | See the shared sketch below: a caller-owned DEST-section mode. |
| `reduce<MAX, REDUCE_ROW>` + `binary_sfpu<BinaryMax>` | capability: "neither a DEST-seeded accumulator nor one-MOP-per-row max" | **partly refuted.** The seeded accumulator is **present and documented** but did not fit. The real gaps are **missing**: (1) the accumulator is always popped; (2) DEST is owned by the helper. | `Accumulate` reloads the running MAX with exactly the within-face transpose the raw code reimplemented (`reduce_helpers_compute.hpp:377-383`, `.inl:695-716`). But it always calls `accum_dfb.pop_front` (`.inl:717`), and online softmax still needs `m_acc` afterwards for `a = exp(m_acc − m_blk)`. That retention is why the fp32 helper body uses `BinaryMax` with `PopPolicy::None` (`…_compute.cpp:602-616`). The raw note "stock reduce_tile MAX on top of the preloaded m_acc measured INCORRECT: inf" (`…_compute.cpp:311`) lines up with the helper's "CRITICAL: Re-init after copy_tile corrupts SRCA config" (`.inl:719`). The helper encodes a rule the raw attempt tripped on. | Add `Accumulate::retain()` (reload without pop). Expose `reduce_block_max_row` as a `ReduceAlgorithm`. Add a doc line on `Accumulate` saying that it is the replacement for `reduce MAX + BinaryMax`. |
| `reduce<SUM, REDUCE_ROW>` | capability: has to share a DEST section with the max and P·V of other rows | **missing** | `reduce_helpers_compute.hpp:23`: the helper owns acquire/commit/wait/release and has a single output CB. | Caller-owned DEST-section mode (sketch below). |
| *(unrecorded)* `eltwise_chain` / bcast-mul rescale, and the dest-reuse causal mask | **no row in any `### Helper bypasses` table** | **gap in the record.** It has the same root cause (DEST ownership), so the verdict is **missing**. | Refinement 3 (a perf-typed refinement) replaced the bf16-DEST rescale helper pass with raw `mul_tiles_bcast_cols` accumulating into the P·V / rowsum DEST (`…_compute.cpp:340, 343, 385`). Refinement 2 added raw `add_reuse_dest_tiles` causal masks (`:239-242`). Neither changelog section has a bypass table, and changelog.md:308 ("bypassed helpers are still used by … the rescale") is now stale for bf16 DEST. | Record both rows retroactively. See §4c for the prompt fix. |

There is no ergonomics row with equal ns: every recorded bypass shows a measured win. Rows 3–6 share one root cause: **no helper can run inside a caller-owned DEST section**. That is one library feature, not four. Sketch (confidence: low; derived from `s2` / `exp_chunk` at `…_compute.cpp:260-390`):

```cpp
// Caller owns acquire/commit/wait/release; helpers emit init + math into given DEST slots only, never acquire/pack.
tile_regs_acquire();
ckl::reduce_row_into<MAX>(cb_scores, cb_max_scaler, ckl::Row{mrow * Sk, skv}, /*dst=*/0,
                          ckl::seed_from(cb_max_acc, mrow, ckl::Pop::No));
ckl::reduce_row_into<SUM>(cb_scores, cb_sum_scaler, ckl::Row{prow * Sk, skv}, /*dst=*/1);
ckl::matmul_row_into(cb_scores, cb_v, ckl::Row{prow * Sk, skv}, /*dst=*/2, pv_w,
                     ckl::then_mul_bcast_cols(cb_out_acc, cb_row_factor, prow));   // DST += a*O_acc
tile_regs_commit(); tile_regs_wait();
ckl::pack_thread<ScaledExp<s>>(/*dst_first=*/0, w);   // one SFPU stall + RWC reset per section, inside
```

## 4. Agent prompts (`.claude/agents/*.md`, orchestrator)

**4a. The implementer deleted the pending refinement queue, so perf ran mid-queue. (Most important.)**
- **What:** the Refinement 1 commit removed Refinements 2–6 from `op_requirements.md`. The orchestrator read the queue as drained and started the trailing perf rounds. Consequences:
  - Perf 1 and Perf 2 were tuned and guard-checked with only MHA / non-causal / non-sharded `SUPPORTED`.
  - Their graduated bf16-DEST stream later carried causal, GQA/MLA, sharded and cross (Refinements 2 and 4) with no perf guard on those paths.
- **Evidence:** `git show bb7819f002d -- …/op_requirements.md` shows `-### [ ] Refinement 2 …` through `-### [ ] Refinement 6 …`. Commit `fd1e683acd5`: "the orchestrator then read the queue as drained and went to the trailing perf rounds". `.claude/agents/ttnn-implementer.md` has no rule limiting edits to the agent's own entry.
- **Recommendation:**
  - In `ttnn-implementer.md`, add: "In `op_requirements.md` edit only your own refinement's checkbox and notes; never delete, reorder or rewrite other entries."
  - In `eval/run_refinements.py`, after each refinement, fail (or auto-restore) when the count of `### [ ]` headings drops by more than the one just ticked.
- **Confidence:** high.

**4b. The verifier changed the gate's environment to get a green count, so a harness defect reached a refinement agent, which then patched the tt-metal runtime.**
- **What:** the Phase 0 verifier ran golden with `EVAL_CAPTURE_L1=0` and reported 110 supported_pass. Under the real gate (L1 capture on), golden_phase0 was 36 passed / 74 failed with `L1ProfilingError peak_total=0`. Refinement 1b then fixed this inside an op-refinement commit by changing shared runtime code: `tt_metal/distributed/mesh_workload.cpp`, `tt_metal/impl/program/program.cpp` and `ttnn/core/graph/graph_trace_utils.cpp`.
- **Evidence:** commit `1190b99d19e`: "Golden (EVAL_CAPTURE_L1=0): 110 supported_pass … With L1 capture on every supported cell fails". `git show --stat f11a7548954`. `refinement_1_debug_output.json`: "I had to change shared tt-metal/ttnn runtime code, outside the op".
- **Recommendation:**
  - In `incremental-verifier.md` (near :204), add: "Run golden exactly as the gate does (`eval/eval_test_runner.sh`, default env). Never set `EVAL_CAPTURE_L1=0` or similar to obtain a count. A harness failure on supported cells is reported as supported_fail and escalated as a framework issue, not deferred."
  - Separately, review the three runtime diffs in `f11a7548954` and upstream them on their own: any heterogeneous `MeshWorkload` op hits this.
- **Confidence:** high.

**4c. Perf-typed refinements do not produce a bypass table.**
- **What:** Refinements 3 and 5 are `**Type**: perf` and Refinement 3 added raw-LLK bypasses (§3), but only the `## Perf N` rounds carried `### Helper bypasses`. The implementer prompt already says to "Record the gap exactly as a raw-LLK bypass is recorded" (`ttnn-implementer.md:557`), but no changelog template slot enforces it.
- **Recommendation:** in `ttnn-implementer.md` and the verifier's perf-refinement template, require a `### Helper bypasses` table (same columns as the perf coordinator's) in the changelog entry of any refinement whose diff adds raw compute-LLK calls.
- **Confidence:** med.

**4d. The ragged-tail rule is incomplete for in-place, one-block CBs.**
- **What:** the prompt says "push and wait the **actual** page count rather than the nominal one" (`ttnn-implementer.md:562`). For `cb_scores` the run found the opposite was required: a CB sized to exactly one block, addressed in place at fixed in-block offsets, must keep its nominal credit and pad the push/pop on short pieces. Otherwise the ring misaligns.
- **Evidence:** changelog.md:149: "`cb_scores` keeps its nominal Sq·Sk credit, with a pad push/pop on R < Sq pieces. Without it the in-place phases read stale slots". Part-optimizer breadcrumb: "R<Sq tail broke in-place mask/exp (scores CB ping-pong)".
- **Recommendation:** append to the rule: "Exception: a CB holding exactly one block that phases address in place (fixed offsets / held `wr_ptr`) keeps its nominal credit; pad the push/pop on a short block so the next block starts at the CB base."
- **Confidence:** med.
