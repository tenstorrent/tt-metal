# Port Report — reduction/generic op family

Second test-drive of the Metal 2.0 op-porting recipe. The audit cleared GREEN; the plan was written for all seven port variants in the family (W, H interleaved, H width-sharded, HW single-core, Welford W, Welford H, Welford HW). **The port itself did not proceed**: a single shared cross-op writer kernel sits in every factory's data flow except Welford HW, and the recipe's guidance for handling it is internally inconsistent (see Handoff §1 below). Stopping at this point is the correct outcome — the port cannot proceed without an upstream decision.

## Handoff points

### 1. Cross-op writer kernel modification — primary blocker

Six of the seven port variants reach the same external writer kernel:

- **Five variants** route output through `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (W, H interleaved, HW single-core, Welford W, Welford H). The kernel has **33 consumer .cpp files** in the tree (grep `writer_unary_interleaved_start_id\.cpp ttnn/cpp/ttnn/`).
- **One variant** (H width-sharded) routes output through `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — **13 consumer .cpp files**.

Both kernels use the legacy `get_compile_time_arg_val(N)`, `get_arg_val<uint32_t>(N)`, and `TensorAccessorArgs<N>()` idioms. Porting the reduction op to Metal 2.0 requires the writer to use `dfb::output` and `ta::output` instead. Either the cross-op kernel changes its signature (breaking 33 + 13 consumers still on the legacy positional-CTA path), or a fork is introduced.

**Where the docs disagree with themselves.**

[`metal2_port_patterns.md` → Caution: Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel) at line 285 instructs:
> **Decision**: Modify in place under the assumption that all consumers will co-migrate to Metal 2.0 together.

The same Caution at line 291 says:
> **The fork path is not used today.** ... If a port hits a case where it does fail, that's a porting problem to record in the report, not a fork to take on the porter's own initiative.

These two directives are in direct tension when **the co-migration assumption visibly fails today** — and it does, because the reduction op is among the first in-tree TTNN ops being ported and 46 other consumer ports haven't even started. Following "modify in place" generates a non-building tree. Following "do not fork on initiative" leaves no way to write any of W, H, HW, Welford W, or Welford H. The Welford HW variant is the only one that escapes the dilemma (it uses `writer_welford_hw.cpp`, local to the op directory).

The first test-drive's resolution (per the `Phase C: W reduction factory port` commit on `akertesz/porting-recipe-1st-test`) was to **fork** the writer into `writer_unary_interleaved_start_id_metal2.cpp` in the reduction op directory. That commit pre-dates the current Caution text — when it landed, the "fork path" was an explicit option in the doc. The fork was removed by `cbdbc75fdd2` (Catalog: scope today-vs-tomorrow; drop Quasar overclaim) and surrounding revisions; the test-drive subagent friction notes record the fork as the answer that worked. The current doc state has no answer.

**Suggested resolutions (for the user to pick):**

- **(a) Accept the fork path back into the catalog** — explicitly permit cross-op kernel forks during the bulk-port window, with a sunset clause once all consumers migrate. This was the first test-drive's answer; it works mechanically.
- **(b) Stage the cross-op writer migration** — pre-port the writer with a `#ifdef METAL2` (or named-CTA-gated) compatibility shim so both consumer styles work from one source. Risks: every consumer port must touch the shim's CTA conventions; the shim ages poorly.
- **(c) Treat the cross-op writer as a kernel-lib boundary crossing** — move the writer into `ttnn/cpp/ttnn/kernel_lib/` and apply the lib refactor strategy (rename, retype, implicit conversion at the boundary). Likely the correct long-term home but a much larger change of scope than this port.
- **(d) Defer reduction op port until ≥1 sibling op completes its migration with whichever path is picked** — kicks the can but lets bigger ops (matmul, layernorm) drive the resolution.

This entry tagged **API/recipe: requires upstream decision before port can proceed**.

### 2. Test infrastructure parity for fork path verification

If the resolution to §1 is the fork path, the existing pytest suite at `tests/ttnn/unit_tests/operations/reduction/` exercises the public TTNN API — it does not care whether the underlying kernel is `writer_unary_interleaved_start_id.cpp` or its `_metal2` fork. So the same tests verify the port without modification. The first test-drive confirmed this (W-reduce + program-cache tests passed against the forked kernels). If the resolution is path (b) or (c), test harness changes are likely too.

This entry tagged **Testing: scoped to upstream decision in §1**.

### 3. ReduceMultiCoreWProgramFactory header/impl mismatch in clean-slate state

`reduce_op_device_operation.hpp:39-44` declares `ReduceMultiCoreWProgramFactory::create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`, while `reduce_op_multi_core_w_program_factory.cpp:17` defines `create_descriptor` returning `ProgramDescriptor`. The clean-slate commit (`418c81b71b8`) notes "W factory restored to its pre-Phase-C state from commit 0ed8e1440f0", but the corresponding header revert wasn't made. The tree as-given does not compile.

Resolution required: revert the W declaration to `create_descriptor` (matches `0ed8e1440f0`) or, after §1 resolves, the W port lands and the mismatch becomes the canonical post-port state. Either way, addressed by completing the port.

This entry tagged **Tree hygiene: pre-existing inconsistency, surface for the user**.

## Vindications

### V1. Audit's borrowed-memory DFB classification holds up

[`port_op_to_metal2_audit.md` → Dynamic CircularBuffer LANDED entry](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_audit.md#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed) directed me to recognize the H factory's `use_width_sharding` branch as a borrowed-memory DFB use case, not a RED gate. The signal was `CBDescriptor::buffer = a.buffer()` at `reduce_op_multi_core_h_program_factory.cpp:111` and `:148`. Without the LANDED tier and the inline rule for it, my reflex would have been to classify dynamic CB as UNSUPPORTED (per the prior test-drive's audit, before iteration round 1 flipped the entry).

[`port_op_to_metal2_audit.md` → Causal-link gate](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_audit.md#step-03--tensoraccessor-in-use-for-every-tensor-read) saved me from flagging `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` as RED ("no TensorAccessor"); the reader genuinely doesn't need one because it reads from a borrowed CB. The gate's explicit "Don't classify under any of the cases below" was the right shape — without it, the reflex would be to recommend a `TensorAccessor` conversion as Check 3 RED.

### V2. Plan template's per-variant breakdown carried the audit's signal forward

[`port_op_to_metal2_recipe.md` → Appendix A](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_recipe.md#appendix-a--metal2_port_planmd-template)'s "Multi-variant ops" callout for the inventory section is what made writing seven variant blocks (W / H interleaved / H sharded / HW / Welford W / Welford H / Welford HW) feel structured rather than overwhelming. The inventory and Planned Spec Shape sections sort cleanly per-variant; the shared sections (Cross-op kernels, Flags, Dropped Plumbing) live above them.

### V3. Dropped Plumbing table forced the cross-op writer issue to surface during planning

Working through `METAL2_PORT_PLAN.md`'s Dropped Plumbing section — enumerating every legacy RTA / CTA that should disappear — was where the cross-op writer's blocker became unmissable. The writer's `compile_time_args = {output_cb_index, TensorAccessorArgs(*dst)}` and `runtime_args = {dst_buffer, num_tiles, start_id}` each appear in the table with their Metal 2.0 replacements (`DFBBinding`, `TensorBinding`, named RTAs). Asking "where do these replacements get applied?" surfaced the answer: in `writer_unary_interleaved_start_id.cpp`, a cross-op file with 33 consumers.

Without the enumeration step, the issue would have surfaced during the construction step when I tried to actually edit the writer — much later, with more partial code in flight. The recipe's framing of Dropped Plumbing as "the gate against builder-pattern carry-over" works in this direction too: the enumeration shines a light on the kernel-side work that the host-side plumbing implies.

### V4. The audit's distinction between op-directory and cross-op kernels was load-bearing

The audit's Check 2 yellow-tier guidance ("fold isolated holdovers into port-time cleanup") gave the right shape for the in-directory kernels' `get_tile_size(cb_id)` holdovers. But the cross-op writer's `get_local_cb_interface(cb_id_out).fifo_page_size` (line 18 of the eltwise/unary writer) sits in a kernel I cannot touch without first resolving Handoff §1. The audit correctly placed the cross-op-writer-touch in the report under the [shared-dataflow-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel), and the recipe correctly told me to flag it at the inventory step. The flagging mechanism worked; it's the *decision* about cross-op kernels that the docs don't pin down.

## Friction

### Gaps

#### G1. Recipe lacks a path forward for "co-migration assumption fails" today

[`port_op_to_metal2_patterns.md` → Caution: Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel) tells me to modify in place under the co-migration assumption. The recipe at line 291 tells me "if the assumption fails, record in the report, not a fork on initiative." Both directives apply to my situation; following one breaks 33 builds, following the other halts the port entirely.

What would help: the catalog Caution to grow a section "When co-migration assumption visibly fails (≥ N consumers, none yet ported)" with a decision tree — e.g., "if the consumer count is small enough that you can co-port them all in this PR, do so; if not, escalate to the doc maintainer." Right now both paths route to "stop and report" without distinguishing which subcase you're in.

This is the friction the test-drive surfaces most strongly. The reduction op family is **the** test case for the cross-op-kernel cohort — the writer is widely shared, and reduction is the first family bulk-port wave will touch.

#### G2. The recipe's Construction step lacks a worked example for borrowed-memory DFB run-params

The H factory's width-sharded branch declares input and output as borrowed-memory DFBs via `borrowed_from = INPUT` / `borrowed_from = OUTPUT`. The [`metal2_migration_guide.md` → DataflowBufferSpec](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md#dataflowbufferspec) section covers the spec field. The [`program_run_params.hpp:96-98` comment](/localdev/akertesz/tt-metal-2nd-port-test/tt_metal/api/tt-metalium/experimental/metal2_host_api/program_run_params.hpp) says "borrowed-memory DFBs update their backing L1 SRAM address from the corresponding tensor_arg." So the implicit contract is: no `dfb_run_params` entry needed for borrowed DFBs; the `TensorArg` carries the runtime address.

But the recipe's Construction step doesn't say this in one place — it has to be assembled from the migration guide + the run-params header. A short worked example in the recipe ("for borrowed-memory DFBs, build the `DataflowBufferSpec` with `borrowed_from = X` and do NOT add a `DFBRunParams` entry — the TensorArg drives the address resolution") would close the gap.

(I didn't actually try to construct one because the port stopped at G1; this is a forecast of friction I'd hit in the construction step.)

#### G3. No in-tree example of `ProgramSpecFactoryConcept` for a TTNN op

A grep of `ttnn/cpp/ttnn/operations/` for `create_program_spec` and `ProgramArtifacts` returns only the reduction op's own pre-port mismatch (the W factory's stale header from the clean-slate revert) plus the CCL ops (which use the pattern but in a CCL-specific shape). The migration guide's Example 1 and 2 show standalone code, not the full TTNN device-op shape. The recipe's TTNN Framework Integration section gives the factory shape skeleton but doesn't show a complete worked-end-to-end port for a non-CCL TTNN op.

The first test-drive's W factory port (`3a6aac1d23a`) is the only concrete example in the project's history — and it was forked-writer-dependent (i.e., Handoff §1-blocked) and the corresponding commit was reverted.

This is a gap that will close itself as soon as one factory ports cleanly. The reduction family was supposed to be that example; G1 blocks it.

### Confusion

#### C1. "Modify in place" decision text vs. "fork path is not used today" note

Already covered under G1 — calling out here because it's the doc structure that produced the confusion, not just the underlying issue. The Caution has its Decision sentence at line 285 and its "fork path is not used today" guidance at line 291. A first-time reader (me) takes ~30 seconds to spot that the two sentences point in different directions; an AI doing the port mechanically might miss the conflict entirely and just modify in place, producing a non-building tree.

Suggested fix: move the "fork path is not used today" guidance to the **top** of the Decision block, gating the "modify in place" instruction behind a co-migration check. Something like: "If you can confirm all current consumers have already migrated to Metal 2.0, modify in place. If they haven't, stop and report — the co-migration is not yet achieved." (And then point to G1's decision tree.)

#### C2. The Scope-boundary preamble doesn't list cross-op writers as an example of the boundary-crossing case

[`port_op_to_metal2_recipe.md` → Read this first](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_recipe.md#read-this-first)'s Scope boundary subsection introduces the porter's writeable surface as "the op's own directory" and lists exceptions for cross-op kernel files (peer ops, modifiable with caution) and framework primitives. The H factory's width-sharded writer goes to `data_movement/sharded/...` — definitely a peer op. The reduction factories' writers go to `eltwise/unary/...` — also a peer op.

But the **second exception** (framework primitives) gives examples like `noc.async_read` and the `TensorAccessor` constructor — primitives the porter *uses*, not *modifies*. The first exception (cross-op kernels) gives the example `writer_unary_interleaved_start_id.cpp` reused by many ops, but does not call out the 30+ count or the "co-migration likely fails" risk. Reading the preamble, I assumed cross-op writers were the easy case; the Caution further into the patterns catalog is where the real risk lives.

Suggested fix: the Scope boundary preamble's cross-op-kernels bullet to grow a one-line cautionary tag — e.g., "modifying these may break sibling ops still on the legacy API; see Caution for the decision shape." Right now the preamble reads as if cross-op modification is routine.

#### C3. The recipe's "stop and report" framing applies to *recipe-discovered* issues, not *Caution-acknowledged* ones

The recipe's "stop and report" language is used most prominently in two places:

- Audit and Plan stop signals: "if you find yourself constructing a clever workaround, stop and report" — directed at the porter avoiding anti-patterns.
- Boundary-rule preamble: "if a call site outside the op directory requires `sem::name` or `ta::name`, stop and record in METAL2_PORT_REPORT.md."

The cross-op writer issue fits *neither* shape — it's not a clever workaround I'm contemplating, and it's not the named-handle-crossing assumption violation. It IS the case the Caution already acknowledges. So when the Caution-acknowledged issue actually fires, the doc has no "stop and report" anchor specifically for it; the closest hooks are the Open Items for Downstream section (mentioned for *successful* ports as a downstream signal, not for *blocked* ones) and the Handoff Points section (intended for boundary-rule violations, not for Caution-acknowledged consequences).

This report invents a new use of Handoff Points — flagging the Caution-acknowledged consequence as a primary blocker. The doc would benefit from a explicit "Handoff Points covers both boundary-rule violations *and* Caution-acknowledged consequences that block the port" framing, with examples for each.

#### C4. Audit's "Result" line shape doesn't have a row for "GREEN but downstream-blocked"

The audit Result was GREEN — every signal was clean. The block isn't a feature gap; it's a doc-vs-reality gap on cross-op kernel modification. There's no audit Result shape that captures "the op itself is portable but a coordinated change is required first." The recipe loaded with the GREEN audit and the planning step uncovered the block.

This is mostly a documentation hygiene issue: a small note in the audit's Path Forward template that says "if the cross-op kernel inventory step in the recipe surfaces a co-migration block, the audit's GREEN is contingent — surface as a recipe-step blocker, not an audit re-classification" would prepare the porter for the timing.

## Open items for downstream

### Cross-op kernel touches

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — referenced by W, H interleaved, HW, Welford W, Welford H factories. Consumer count today (May 2026): 33 in-tree `.cpp` files outside this op. The first consumer to actually port will pay the migration cost (and need permission to either modify in place + co-port 33 others, or fork + sunset the fork). Recording here so the next porter sees the coordination signal.
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — referenced by H width-sharded path. Consumer count: 13. Same coordination signal.

### Per-op carry-over

- The same audit and plan shape applies to every TTNN op that uses the cross-op writer. Once the resolution to Handoff §1 lands, the audit will need an entry calling out the cross-op kernel rules explicitly (right now the audit's Appendix A is feature-shaped, not kernel-touch-shaped) and the patterns catalog Caution needs the decision-tree language from G1.

### Doc-evolution suggestions

- See G1 for the principal suggestion (Caution's decision tree).
- See C1 / C2 / C3 / C4 for the targeted edits.
- Welford HW is the only fully-portable variant in the reduction family with current docs (its writer is local). A subagent test-drive scoped to "port only the Welford HW variant" would be a useful followup once §1 is unblocked enough to attempt the W factory — Welford HW exercises the writer↔compute 2-way DFB flow (c_21 partial, c_22 combined), which neither the first test-drive nor any existing in-tree code currently demonstrates.

### Test coverage notes

- The pytest suite at `tests/ttnn/unit_tests/operations/reduction/` is API-level; it exercises every variant in the family without caring about kernel implementation. Coverage is adequate for verifying the port once unblocked.
- Per-variant test coverage (by parallelization strategy) is implicit: the test parameterizes over input shape, which selects the strategy. No explicit per-factory unit tests exist below the API layer. Sufficient for correctness verification; may not pin down which variant regressed if a future port breaks one of W / H / HW.
