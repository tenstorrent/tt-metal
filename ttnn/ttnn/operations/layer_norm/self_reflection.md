# Self-Reflection: layer_norm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm` |
| Pipeline Phases Executed | Phase 0 (Discovery), Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels), Phase 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer, ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd (4 stages) |
| Total Git Commits | 8 (1 analyzer + 1 analyzer breadcrumbs + 1 architect + 1 builder + 4 kernel writer stages) + 1 report |
| Total Pipeline Duration | ~34 minutes (10:02 to 10:36 UTC, Mar 9 2026) |
| Overall Result | SUCCESS |
| Pipeline Iteration | Run 4 (previous runs: Run 1 Mar 2, Run 2 Mar 4 08:23, Run 3 Mar 4 10:32) |

---

## 0. Context: Four Pipeline Runs

This is the **fourth** time layer_norm was built. The previous self-reflection (commit `a44288712d8`, Mar 9 09:46) analyzed Runs 1-3 extensively. This report focuses on Run 4, which is the current state on the branch.

| Run | Date | Duration | Analyzer Ref | Kernel Writer | Outcome | TDD Failures |
|-----|------|----------|--------------|---------------|---------|--------------|
| 1 | Mar 2, 14:06-15:21 | ~75m | moreh_group_norm | ttnn-kernel-writer (non-TDD) | PASS + post-fix | CT args fix needed |
| 2 | Mar 4, 08:23-09:17 | ~54m | tilize, untilize, softmax | ttnn-kernel-writer-tdd | PASS | 1 hang, 3 compile errors |
| 3 | Mar 4, 10:32-11:44 | ~72m | moreh_norm_w, softmax_general | ttnn-kernel-writer (non-TDD) | CLEAN PASS | 0 failures |
| **4** | **Mar 9, 10:02-10:36** | **~34m** | **softmax_w_small** | **ttnn-kernel-writer-tdd** | **PASS** | **1 hard attempt (scale_shift)** |

Run 4 is a major efficiency improvement over all previous runs: 34 minutes vs 54-75 minutes.

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Start | End | Duration | Status | Key Observations |
|-------|----------|-------|-----|----------|--------|------------------|
| 0: Discovery | orchestrator | ~10:00 | ~10:02 | ~2m | Done | Selected softmax_w_small as sole reference |
| 1: Analysis | ttnn-operation-analyzer | 10:02:43 | 10:06:18 | ~3m 35s | Done | Single analyzer; produced `softmax_w_small_analysis.md` |
| 2: Design | ttnn-operation-architect | 10:08:00 | 10:11:06 | ~3m 6s | Done | Derivative mode; produced `op_design.md` + `.tdd_state.json` |
| 3: Build | ttnn-generic-op-builder | ~10:12 | 10:25:46 | ~13m | Done | 10 CBs, 3 stub kernels, 7 integration tests, 4 TDD test files |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | 10:27:37 | 10:35:26 | ~7m 49s | Done | 4 stages; 3 clean passes + 1 retry on scale_shift |
| 5: Report | orchestrator | ~10:35 | 10:36:30 | ~1m | Done | REPORT.md generated |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer | 10:02:43 | 10:06:18 | 3m 35s | 0 | All productive |
| ttnn-operation-architect | 10:08:00 | 10:11:06 | 3m 6s | 0 | All productive |
| ttnn-generic-op-builder | ~10:12 | 10:25:46 | ~13m | 0 | All productive |
| ttnn-kernel-writer-tdd | 10:27:37 | 10:35:26 | 7m 49s | 1 (stage 4) | ~6m productive, ~2m debugging InterleavedAddrGenFast |

**Duration calculation method**: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps for analyzer and architect. Git commit timestamps for builder (no breadcrumbs). Breadcrumb timestamps for kernel writer (start event at 10:27:37, last test_run at 10:35:01). Report commit timestamp for Phase 5.

### Duration Visualization

```
Phase 0  |#|                                         (~2m)
Phase 1  |####|                                      (~4m)  1 analyzer
Phase 2       |###|                                  (~3m)
Phase 3            |#############|                   (~13m)
Phase 4                          |########|          (~8m)  4 TDD stages
Phase 5                                   |#|        (~1m)
         0    5    10   15   20   25   30   35 min

Longest phase: Phase 3 (~13m) -- builder produced 721 lines across 9 files
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~6m | 18% | 1 analyzer, single reference |
| Design (Phase 2) | ~3m | 9% | Derivative mode from softmax_w_small |
| Build (Phase 3) | ~13m | 38% | 9 files, 721 lines, includes integration tests |
| Kernel implementation (Phase 4) | ~8m | 24% | 4 TDD stages |
| -- Productive coding | ~6m | 18% | Stages 1-3 clean, stage 4 first pass |
| -- Debugging/retries | ~2m | 6% | InterleavedAddrGenFast .data_format fix |
| Reporting (Phase 5) | ~1m | 3% | |
| **Total** | **~34m** | **100%** | |

---

## 2. What Went Well

### 1. Remarkable TDD execution speed: 4 stages in under 8 minutes

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: From breadcrumbs:
- data_pipeline: start 10:27:37 -> pass 10:29:18 (1m 41s)
- subtract_mean: pass 10:30:42 (1m 24s from previous)
- normalize: pass 10:32:11 (1m 29s from previous)
- scale_shift: pass 10:35:01 (2m 50s from previous, includes 1 retry)

First 3 stages passed with `hard_attempts: 0, free_retries: 0`. Total kernel implementation: 7m 49s for a 6-phase normalization compute kernel with 133 lines of C++.

**Why it worked**: The architect's design document was exceptionally detailed. It specified exact helper function calls with template parameters, CB input/output policies, manual pop instructions, and CB reuse patterns. The kernel writer could effectively transcribe the design rather than invent the implementation. The `op_design.md` Part 2 section reads like pseudocode, with exact function signatures for all 6 compute phases.

### 2. Single, focused reference analysis produced superior results

**Phase/Agent**: Phase 1 / ttnn-operation-analyzer
**Evidence**: The analyzer produced a 490-line `softmax_w_small_analysis.md` covering: CB layout (9 CBs), 4 compute phases with exact code snippets, multi-pass data reuse patterns, reduce helper API signatures, broadcast semantics, and a dedicated "Relevance to Layer Norm" section mapping softmax patterns to layer_norm requirements.

**Why it worked**: Using softmax_w_small as the sole reference (rather than the 2-3 references in Runs 2-3) gave the analyzer a tighter focus. The reference shares the exact same algorithmic pattern: W-dimension row-reduce with multi-pass CB reuse. The analysis document explicitly mapped softmax phases to layer_norm phases (mean via `reduce<SUM>` with 1/W scaler, subtract via `sub_tiles_bcast<COL>`, rsqrt as post-reduce op analogous to recip).

### 3. CB layout was correct from design through implementation with zero CB bugs

**Phase/Agent**: Phase 2 (Architect) through Phase 4 (Kernel Writer)
**Evidence**: The architect designed 10 CBs. The builder allocated exactly these 10 CBs in the program descriptor. The kernel writer used all 10 with the correct page counts and lifetime semantics. Zero CB-related compilation errors, hangs, or numerical bugs across all 4 TDD stages. CB dual-use (c_24 as mean then rstd, c_27 as diff_sq then temp_norm) was correctly designed, built, and implemented.

**Why it worked**: The architect's "CB Allocation (final, validated against helpers)" table and "Binary Op Broadcast Verification" table left no ambiguity about which CBs are consumed/produced at each phase. The builder translated this 1:1 into `CBDescriptor` objects.

### 4. Overall pipeline duration cut by more than 50% vs any previous run

**Phase/Agent**: Entire pipeline
**Evidence**: Run 4 took ~34 minutes. Run 1: ~75m, Run 2: ~54m, Run 3: ~72m. This is a 2.1x speedup vs Run 2 (the fastest previous run).

**Why it worked**: Multiple factors: (a) single focused reference instead of multiple scattered references, (b) TDD kernel writer with clear stage boundaries, (c) the architect produced a near-complete implementation specification, (d) no device hangs (which consumed significant time in Run 2).

---

## 3. Issues Found

### Issue 1: InterleavedAddrGenFast missing .data_format for gamma/beta reads

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- scale_shift (stage 4) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~2 minutes |

**Problem**: The scale_shift stage initially failed with a numerical mismatch (max diff: 11.6875, recorded in `.tdd_state.json` failure_history). The root cause was that the `InterleavedAddrGenFast` struct for gamma and beta reads was missing the `.data_format` field. Without it, address calculations were incorrect for tensors wider than a single tile (Wt > 1), causing gamma/beta tiles to be read from wrong addresses. The fix was adding `.data_format = get_dataformat(cb_gamma)` (and matching for beta) in `layer_norm_reader.cpp`.

Git diff (`3c0ff69..732064a`):
```cpp
const auto gamma_accessor = InterleavedAddrGenFast</*is_dram=*/true>{
    .bank_base_address = gamma_addr,
    .page_size = tile_bytes,
+   .data_format = get_dataformat(cb_gamma),
};
```

**Root Cause**: The `InterleavedAddrGenFast` struct has `.data_format` as a member that is needed for correct bank offset computation when `get_noc_addr(page_id, accessor)` is called. Without it, the default (likely zero/uninitialized) value produces wrong offsets for pages beyond the first. This is a well-known footgun in the codebase. The architect's design document did not specify `InterleavedAddrGenFast` usage -- it described gamma/beta reads generically. The kernel writer independently chose `InterleavedAddrGenFast` (as noted in REPORT.md: "Used InterleavedAddrGenFast instead of TensorAccessor since gamma/beta don't have TensorAccessorArgs in the program descriptor").

**Fix for agents**:
- **ttnn-operation-architect**: When designing reader kernels that access secondary tensors (gamma, beta, mask) not covered by TensorAccessor, explicitly specify the read method and required struct fields. Example: "Use `InterleavedAddrGenFast<true>` with `.bank_base_address`, `.page_size`, and `.data_format = get_dataformat(cb_gamma)`."
- **ttnn-kernel-writer-tdd**: Add a checklist item: "Every `InterleavedAddrGenFast` initialization must include `.data_format`. Missing it causes silent address calculation errors for multi-tile reads."

### Issue 2: No execution log files produced by any agent

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | All phases |
| Agent | All agents |
| Retries Consumed | 0 |
| Time Cost | 0 (but reduces post-hoc analysis quality) |

**Problem**: The `agent_logs/` directory contains only breadcrumb JSONL files for 3 agents (analyzer, architect, kernel-writer-tdd). No `*_execution_log.md` files were produced. The builder has no breadcrumbs at all. This means the structured execution summaries (input interpretation, execution timeline, recovery tables, deviations, handoff notes, instruction improvement recommendations) are not available for analysis.

**Root Cause**: Execution log generation is apparently not part of the current agent prompts, or is optional and was not triggered. Builder breadcrumbs may not be implemented.

**Fix for agents**:
- **All agents**: Require execution log generation as a mandatory final step before the agent completes. The execution log template should be part of each agent's prompt.
- **ttnn-generic-op-builder**: Add breadcrumb generation (start, completion, key decisions) to match other agents.

### Issue 3: Builder phase is disproportionately long at 38% of total pipeline time

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 |
| Time Cost | ~13 minutes (38% of 34m total) |

**Problem**: The builder took ~13 minutes to produce the Python infrastructure (entry point, program descriptor, stub kernels, test files). This is longer than the entire kernel implementation phase (4 TDD stages in ~8 minutes). The builder's output is 721 lines across 9 files, which is substantial but formulaic -- the CB descriptors, kernel descriptors, and test boilerplate follow predictable patterns from the architect's design.

**Root Cause**: The builder likely runs on a less capable model (Sonnet, per pipeline-improvements.md issue #6) and spends time navigating ProgramDescriptor API quirks. The 10-CB setup with conditional gamma/beta handling adds complexity. No breadcrumbs exist to diagnose where time was spent within the builder.

**Fix for agents**:
- **Pipeline orchestrator**: Consider running the builder on Opus (same model as architect/kernel-writer) per existing pipeline-improvements.md issue #6.
- **ttnn-generic-op-builder**: Add breadcrumbs to identify whether time is spent on code generation, API lookup, test creation, or integration testing.

### Issue 4: Design document contains visible deliberation/revision text

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2 |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 |
| Time Cost | 0 directly, but adds cognitive load for downstream agents |

**Problem**: The `op_design.md` Part 2 section contains inline deliberation visible to downstream agents. Examples:
- Lines 271-296: "The post-reduce lambda for fusing eps addition and rsqrt is complex... Instead, we use a two-step approach:" followed by "Revise: use cb_24..."
- Lines 325-344: Extended deliberation about where P5 should output ("Requires reading output back... Better approach... Actually, P5 with PopAtEnd..."), including crossed-out alternatives and a final "Final revised flow" section.

While the final answer is correct, these "thinking out loud" sections risk confusing the kernel writer if it reads them literally rather than skipping to the final answer.

**Root Cause**: The architect model streams its reasoning directly into the design document without a revision pass. There is no "clean up deliberation" step.

**Fix for agents**:
- **ttnn-operation-architect**: Add a final cleanup pass instruction: "Before writing the design document, resolve all deliberation internally. The document should present only final decisions, not the reasoning path. If alternatives were considered and rejected, do not include them."

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | 1m 41s | 0 free, 0 hard | PASS | Clean -- identity passthrough with copy_tile loop |
| subtract_mean | 1m 24s | 0 free, 0 hard | PASS | Clean -- reduce + sub with correct NoPop policies |
| normalize | 1m 29s | 0 free, 0 hard | PASS | Clean -- 4 compute phases (square, reduce, add+rsqrt, mul) |
| scale_shift | 2m 50s | 0 free, 1 hard | PASS | InterleavedAddrGenFast .data_format missing |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Build phase | ttnn-generic-op-builder | ~13m | 38% | Creating 9 files with 721 lines | 0 | Builder model/API complexity |
| 2 | scale_shift retry | ttnn-kernel-writer-tdd | ~1m | 3% | InterleavedAddrGenFast fix | 1 hard | Missing .data_format field |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | First attempt at scale_shift stage (gamma/beta reads without .data_format) | Missing struct field caused incorrect address calculation, producing 11.6875 max diff | Architect should specify exact InterleavedAddrGenFast initialization; kernel writer should have checklist for required struct fields |

**Note on cross-run waste**: The previous self-reflection documented ~3.5 hours wasted across Runs 1-3. Run 4 (this run) started from scratch again rather than building on Run 3's proven kernel code. However, Run 4 produced a better result (more detailed design, cleaner kernel code, better analysis document) in less time. The fresh start was partially justified by the improved pipeline tooling (TDD orchestrator improvements between runs).

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | `softmax_w_small_analysis.md` (490 lines) |
| Quality | GOOD |
| Issues | None significant. The "Relevance to Layer Norm" section at the end explicitly mapped each softmax phase to the corresponding layer_norm operation. |
| Downstream Impact | The architect's breadcrumb shows `mode_detection: Derivative` and `design_decision: USE HELPERS` with all 7 helper operations explicitly listed, citing the analysis. Design was completed in 3 minutes. |
| Suggestion | Continue the practice of including a dedicated "Relevance to [target op]" mapping section in analysis documents. This directly accelerates architect design time. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md` Part 1 (Architecture section, ~136 lines) |
| Quality | GOOD |
| Issues | The deliberation text in Part 2 (intended for the kernel writer) is also visible to the builder. The builder correctly ignored it and only used Part 1's CB table and kernel argument specifications. |
| Downstream Impact | Builder correctly produced all 10 CBs with correct page counts, all kernel descriptors with correct compile-time and runtime args. Zero builder-related bugs downstream. |
| Suggestion | Consider splitting Part 1 and Part 2 into separate files if deliberation text in Part 2 is a concern. |

### Handoff 3: ttnn-operation-architect + ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifacts Passed | `op_design.md` Part 2 (kernel implementation, ~250 lines), builder's stub kernels and program descriptor |
| Quality | ADEQUATE |
| Issues | (1) The design specified gamma/beta reads generically ("If has_gamma, read Wt tiles into cb_gamma") without specifying the read mechanism (`InterleavedAddrGenFast` vs `TensorAccessor`). The kernel writer had to independently determine that `InterleavedAddrGenFast` was needed because gamma/beta lack `TensorAccessorArgs` in the program descriptor. (2) The builder allocated gamma/beta CBs but did not add `TensorAccessorArgs` for these tensors to the reader's compile-time args, forcing the kernel writer to use an alternative read method. |
| Downstream Impact | The kernel writer's choice to use `InterleavedAddrGenFast` without `.data_format` caused the only failure in the pipeline (1 hard attempt on scale_shift). |
| Suggestion | Architect should specify the exact read mechanism for every tensor. Builder should either add `TensorAccessorArgs` for secondary tensors or explicitly document in comments that gamma/beta require `InterleavedAddrGenFast`. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | Self-reflection | Specify exact read mechanism (TensorAccessor vs InterleavedAddrGenFast) for every tensor in the reader kernel design, including required struct fields | H | H |
| ttnn-operation-architect | Self-reflection | Remove deliberation text from final design doc; present only resolved decisions | M | M |
| ttnn-kernel-writer-tdd | Self-reflection | Add checklist: "InterleavedAddrGenFast requires .data_format field" | H | H |
| ttnn-generic-op-builder | Self-reflection | Add breadcrumb logging (start, key decisions, completion) | M | M |
| All agents | Self-reflection | Generate execution_log.md as mandatory completion step | M | L |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Build | Builder consumes 38% of pipeline time despite producing formulaic output | Investigate builder model upgrade (Opus) or template-based generation for CB descriptors | M |
| TDD | 4 stages completed in 8 minutes with only 1 failure -- TDD is working well | No change needed; current stage design is effective | -- |
| Analysis | Single focused reference (softmax_w_small) outperformed multi-reference approaches from previous runs | Prefer single, highly-relevant references over multiple partial-match references | M |
| Design | Architect design quality directly correlates with TDD speed (3m design -> 8m kernel implementation) | Continue investing in detailed design documents with pseudocode-level helper specifications | H |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Numerical debugging burns context | NO | Only 1 retry, resolved quickly (~2m). The numerical mismatch was large (11.6875) which made the bug easy to identify as an address calculation error rather than a subtle numerical drift. |
| 3 | `.tdd_state.json` fragility | NO | TDD state file worked correctly through all 4 stages. |
| 6 | Builder runs on Sonnet | POSSIBLY | Builder was the slowest phase (38% of total). No direct evidence of model choice, but the disproportionate time is consistent with this known issue. |
| 9 | No architect/builder cross-validation | NO | CB allocation matched perfectly between design and builder. However, this was fortunate -- no automated validation actually runs. |
| 12 | No cross-run learning | YES | Run 4 started from scratch despite 3 prior runs. However, the improved pipeline tooling made the fresh start fast enough (~34m) that the cost was acceptable this time. |
| 13 | Broadcast type validation | NO | All broadcast types were correct in this run. The architect's "Binary Op Broadcast Verification" table caught this at design time. |
| 14 | Reference layout gate | NO | softmax_w_small was a well-matched TILE_LAYOUT reference. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| InterleavedAddrGenFast .data_format footgun | Missing `.data_format` field in `InterleavedAddrGenFast` initialization silently produces wrong addresses for multi-tile reads. Neither architect nor kernel writer had a checklist for this. | MEDIUM |
| Builder lacks breadcrumb logging | The builder is the only pipeline agent without breadcrumb generation, making it impossible to diagnose where its ~13 minutes are spent. | LOW |
| Architect designs secondary tensor reads too generically | When gamma/beta (or similar secondary tensors) need to be read by the reader kernel but lack TensorAccessorArgs, the design document should explicitly specify the read method and struct initialization, not leave it to the kernel writer. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Add InterleavedAddrGenFast .data_format to architect and kernel writer checklists

- **Type**: instruction_change
- **Target**: Architect agent prompt, Kernel writer agent prompt
- **Change**: Architect: when specifying reader reads for secondary tensors (gamma, beta, mask, etc.) that use `InterleavedAddrGenFast`, always include the full struct initialization including `.data_format = get_dataformat(cb_xxx)`. Kernel writer: add a validation checklist item "Every `InterleavedAddrGenFast` must have `.bank_base_address`, `.page_size`, AND `.data_format`."
- **Expected Benefit**: Eliminates the most common single-retry failure pattern for operations with secondary tensor reads.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add breadcrumb logging to the builder agent

- **Type**: tool_improvement
- **Target**: ttnn-generic-op-builder agent
- **Change**: Add breadcrumb JSONL output at: start, after CB generation, after kernel descriptor generation, after test generation, and at completion. Include timing for each sub-phase.
- **Expected Benefit**: Enables diagnosis of builder performance bottlenecks (13m in this run, 38% of total).
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Prefer single highly-relevant references over multiple partial references

- **Type**: pipeline_change
- **Target**: Discovery phase (Phase 0) / orchestrator
- **Change**: When a single reference covers all needed patterns (compute, dataflow, CB layout), prefer it over combining 2-3 partial references. Add a scoring mechanism that weights "coverage completeness" over "number of aspects covered by different references."
- **Expected Benefit**: Run 4 (single reference: softmax_w_small) took 34m with 1 failure. Run 2 (3 references: tilize, untilize, softmax) took 54m with 4 failures. Single focused references produce better analysis documents and more coherent designs.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 4: Architect should eliminate deliberation text from design documents

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add instruction: "The design document must present only final decisions. Do not include deliberation, alternatives considered, or revision history. Resolve all design choices before writing. If the CB assignment changes during design, update the tables to reflect only the final state."
- **Expected Benefit**: Reduces cognitive load on downstream agents (builder and kernel writer). Prevents potential confusion if an agent reads intermediate deliberation text as instructions.
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 5: Builder should add TensorAccessorArgs for secondary tensors or document the gap

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent prompt
- **Change**: When the architect's design includes secondary tensors (gamma, beta) read by the reader kernel, the builder should either: (a) add `TensorAccessorArgs` for these tensors to the reader's compile-time args (preferred), or (b) add explicit comments in the program descriptor noting "gamma/beta require InterleavedAddrGenFast reads; TensorAccessor not available for these tensors."
- **Expected Benefit**: Prevents the kernel writer from having to independently discover the read mechanism mismatch, which was the root cause of the only failure in this run.
- **Priority**: MEDIUM
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Excellent reference selection (softmax_w_small). Single point deduction: no automated scoring of reference quality. |
| Analysis quality | 5/5 | Outstanding. 490-line analysis with explicit layer_norm mapping section. Covered all needed patterns. |
| Design completeness | 4/5 | Near-complete pseudocode-level design. One gap: secondary tensor read mechanism not fully specified. Deliberation text is a minor issue. |
| Build correctness | 5/5 | All 10 CBs correct, all kernel descriptors correct, all args correct. Zero builder-related bugs. |
| Kernel implementation | 5/5 | 4 stages in 8 minutes with only 1 retry. Clean CB reuse, correct helper invocations, proper push/pop balance. |
| Inter-agent communication | 4/5 | Excellent analyzer->architect and architect->builder handoffs. One gap: architect->kernel_writer handoff for secondary tensor reads was underspecified. |
| Logging/observability | 2/5 | Breadcrumbs exist for 3 of 4 agents but are minimal (5-6 entries each). No execution logs for any agent. Builder has no logging at all. Timeline reconstruction required git commit timestamps as primary source. |

### Top 3 Things to Fix

1. **Add InterleavedAddrGenFast .data_format to checklists** -- This was the only failure in an otherwise flawless run. A single checklist item in the architect and kernel writer prompts eliminates it permanently.
2. **Add builder breadcrumbs** -- The builder is a black box consuming 38% of pipeline time. Without logging, we cannot diagnose or optimize it.
3. **Eliminate design document deliberation text** -- The architect's thinking-out-loud style risks confusing downstream agents, even though it did not cause problems in this run.

### What Worked Best

The **analysis quality** was the standout success of this pipeline run. The analyzer's `softmax_w_small_analysis.md` was so thorough -- with exact code snippets, helper API signatures, CB lifecycle tracking, and an explicit "Relevance to Layer Norm" mapping -- that the architect completed the design in 3 minutes and the kernel writer implemented all 4 stages in under 8 minutes. The single focused reference strategy (one excellent reference instead of multiple partial ones) drove the entire pipeline's efficiency improvement from ~60-75 minutes in previous runs down to 34 minutes. This validates the principle: a deep analysis of one highly-relevant reference is worth more than shallow analyses of several partially-relevant ones.
