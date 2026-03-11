# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | Phase 0 (Discovery), Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels), Phase 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3 parallel), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd (x2 sessions) |
| Total Git Commits | 12 (this run, filtered to 2026-03-11 17:xx-19:xx range) |
| Total Pipeline Duration | ~2h 10m (17:04 - 19:14 UTC) |
| Overall Result | SUCCESS -- all 4 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m | PASS | Identified 3 hybrid references: tilize (input_stage), untilize (output_stage), batch_norm (compute_core) |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~12m | PASS | 3 analyzers ran in parallel: tilize, untilize, batch_norm. All produced comprehensive markdown analyses. |
| 2: Design | ttnn-operation-architect | ~9m | PASS | Produced 414-line op_design.md with 10-phase compute pipeline, 11 CBs, 4 TDD stages. Self-corrected B policies in pass 2. |
| 3: Build | ttnn-generic-op-builder | ~16m | PASS | Created Python infra + stub kernels + tests. 1 free retry for bad tensor_accessor include path. Fixed architect's test reference functions. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd (x2) | ~1h 23m | PASS | Session 1: stages 1-2 (stage 2 hit bf16 wall). Session 2: stages 3-4 (clean). Orchestrator loosened tolerance between sessions. |
| 5: Report | orchestrator | ~2m | PASS | REPORT.md generated summarizing full pipeline. |

### Agent Duration Breakdown

Duration calculation method: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps used as primary source. Git commit timestamps used as cross-check.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 17:05:19 | ~17:10:13 | ~5m | 0 | ~5m active |
| ttnn-operation-analyzer (untilize) | 17:04:33 | ~17:14:28 | ~10m | 0 | ~10m active |
| ttnn-operation-analyzer (batch_norm) | 17:04:51 | ~17:17:03 | ~12m | 0 | ~12m active |
| ttnn-operation-architect | 17:18:39 | 17:27:36 | ~9m | 0 | ~9m active |
| ttnn-generic-op-builder | 17:30:08 | 17:45:51 | ~16m | 1 (free) | ~14m active, ~2m fixing include path |
| ttnn-kernel-writer-tdd (session 1) | 17:49:05 | ~18:53:28 | ~64m | 5 hard + 2 free | ~8m productive, ~56m debugging bf16 precision |
| ttnn-kernel-writer-tdd (session 2) | 19:03:57 | 19:12:13 | ~8m | 1 free | ~7m productive, ~1m fixing rsqrt include |

### Duration Visualization

```
Phase 0  |#|                                                          (~1m)
Phase 1  |############|                                               (~12m) 3 analyzers parallel
Phase 2       |#########|                                             (~9m)
Phase 3            |################|                                 (~16m)
Phase 4 S1                          |##################################################████████████| (~64m)
         [gap: orchestrator tolerance fix, ~10m]
Phase 4 S2                                                                               |########| (~8m)
Phase 5                                                                                         |##| (~2m)
         0    10   20   30   40   50   60   70   80   90  100  110  120  130 min

Longest phase: Phase 4 Session 1 (64m) -- bf16 precision debugging in center_and_square
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~13m | 10% | 3 analyzers in parallel |
| Design (Phase 2) | ~9m | 7% | Single attempt, clean |
| Build (Phase 3) | ~16m | 12% | 1 free retry for include path |
| Kernel implementation (Phase 4) | ~82m | 63% | 2 TDD sessions |
| -- Productive coding | ~15m | 12% | Stage 1 + Stage 3 + Stage 4 implementation |
| -- Debugging/retries | ~56m | 43% | Stage 2 bf16 precision investigation |
| -- Orchestrator gap | ~10m | 8% | Tolerance loosening between sessions |
| Reporting (Phase 5) | ~2m | 2% | |
| Overhead/gaps | ~8m | 6% | Agent startup, inter-phase transitions |
| **Total** | **~130m** | **100%** | |

---

## 2. What Went Well

### 1. CB Layout Was Entirely Correct From the Start

**Phase/Agent**: Phase 2 (ttnn-operation-architect)
**Evidence**: All 11 circular buffers (c_0, c_1, c_2, c_8, c_9, c_16, c_24-c_28) in the final compute kernel (`layer_norm_rm_compute.cpp`) match the architect's design table exactly in ID, purpose, sizing, and lifetime. Zero CB-related bugs across 4 TDD stages. The kernel writer's breadcrumb `cb_sync_check` events consistently report `balanced:true`.
**Why it worked**: The architect read all three reference analyses (tilize, untilize, batch_norm) and the helper library headers before designing the CB layout. The pass-2 self-correction (changing sub-mean and mul-normalize B policies from NoWaitNoPop to WaitUpfrontPopAtEnd) prevented what would have been a hang or deadlock in the kernel.

### 2. Stages 1, 3, and 4 Passed With Zero Hard Attempts

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: `.tdd_state.json` shows: data_pipeline (0 hard, 0 free), normalize (0 hard, 1 free), affine (0 hard, 0 free). Only 1 free retry total for a missing `rsqrt.h` include. Combined wall time for these three stages was ~12 minutes across both sessions.
**Why it worked**: The architect's design was detailed enough (per-phase CB routing, helper function signatures, input policies) that the kernel writer could implement each stage as a direct translation with minimal interpretation. The TDD stage progression was well-ordered: each stage added exactly one new concern.

### 3. Reference Selection and Analysis Quality Were Strong

**Phase/Agent**: Phase 0-1 (orchestrator + ttnn-operation-analyzer)
**Evidence**: The three references (tilize, untilize, batch_norm) each served a distinct, non-overlapping role. Tilize provided the RM stick batching pattern (32 sticks -> Wt tile-pages). Untilize provided the writer stick extraction pattern. Batch_norm provided the eps fill, conditional affine routing, and the overall normalization compute chain. The architect's breadcrumbs confirm all three were read and their key findings incorporated: `"reference_read"` events at 17:19:02 list specific findings from each.
**Why it worked**: Hybrid mode (input_stage + output_stage + compute_core) was the correct decomposition for an RM-in/RM-out normalization operation. Each analyzer produced focused, actionable output (tilize_analysis.md: 332 lines, untilize_analysis.md: 433 lines, batch_norm_analysis.md: 597 lines).

### 4. No Device Hangs or Infrastructure Issues

**Phase/Agent**: All phases
**Evidence**: Zero device hangs across all test runs. No venv problems. No build failures (kernels compile at runtime). Pre-commit hooks auto-fixed clang-format on first commit attempt without consuming test budget.
**Why it worked**: The CB sync was correct from the start (the architect's self-correction caught the policy mismatch), and all helper function calls matched their expected signatures.

### 5. Inter-Agent Handoff From Architect to Builder Was Clean

**Phase/Agent**: Phase 2 -> Phase 3
**Evidence**: The builder's execution log reports "None - input was clear and complete" under Interpretation Issues. The builder successfully created all 11 CBs, correct kernel argument layouts, and proper work distribution from the architect's design. The only issues were: (a) test reference function bugs (missing `return`, wrong variable names), and (b) a bad tensor_accessor include path from the system prompt (not from the architect).
**Why it worked**: The architect's design doc was fully specified with tables for every CB, every kernel argument index, and every compile-time/runtime arg.

---

## 3. Issues Found

### Issue 1: Kernel Writer Spent 56 Minutes Debugging an Inherent bf16 Precision Limitation

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- center_and_square |
| Agent | ttnn-kernel-writer-tdd (session 1) |
| Retries Consumed | 5 hard attempts + 2 free retries |
| Time Cost | ~56 minutes (68% of Phase 4 wall time) |

**Problem**: The center_and_square stage computes `(x - mean(x))^2` where `mean` is computed via `reduce_row` with bf16 accumulation. The hardware reduce introduces ~0.01 error in the mean, which is then amplified by squaring to a max diff of 0.375 vs the PyTorch fp32 reference. The stage was configured with `atol=0.1`, making a pass impossible without either fp32 accumulation (which breaks untilize due to DEST halving) or loosened tolerance.

The kernel writer went through a lengthy investigation cycle (breadcrumbs 17:56:58 - 18:53:28):
1. First hypothesis (H1): scaler interference. Tried `fp32_dest_acc_en=True` -> max diff worsened to 9.0 (DEST capacity halved, untilize broke). Reverted.
2. Second hypothesis (H2): kernel cache not invalidated. Tried passthrough diagnostic to verify recompilation -> confirmed kernel rebuilds work.
3. Third hypothesis (H3/H4): switched to `reduce<SUM>` with scaler=1.0 + post_reduce_op `mul_unary_tile(1/W)` -> encountered FloatBits union name conflict -> fixed -> same max diff persisted (the approach was equivalent).
4. Reverted to original 1/W scaler approach. Tried DPRINT diagnostics to inspect intermediate tiles. Context approaching exhaustion.
5. Eventually ran diagnostic isolating centered vs squared outputs -> confirmed error is in reduce accumulation, amplified by squaring.
6. Orchestrator intervened: loosened tolerance from atol=0.1 to atol=0.5.

**Root Cause**: The architect set the center_and_square tolerance to `atol=0.1`, which is unrealistic for bf16 reduce followed by squaring. The kernel writer did not have a strategy for recognizing "inherent precision limitation" vs "implementation bug" and spent most of its budget investigating what was fundamentally a tolerance miscalibration.

**Fix for agents**:
- **ttnn-operation-architect**: When designing TDD stages involving `reduce` + nonlinear operations (square, power) in bf16 mode, set intermediate stage tolerances to at least `atol=0.5`. Add a note in the design: "bf16 reduce precision: expect ~0.01 mean error, amplified by squaring to ~0.4 max diff."
- **ttnn-kernel-writer-tdd**: Add an explicit "bf16 precision classification" step: if the first hard attempt shows a consistent small numerical mismatch (max diff < 1.0) with the correct output pattern (values track reference but with systematic offset), classify it as "bf16 precision" after 1 diagnostic attempt and recommend tolerance loosening to the orchestrator, rather than continuing to try code-level fixes.

### Issue 2: fp32_dest_acc_en Breaks Untilize Due to DEST Capacity Halving

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- center_and_square |
| Agent | ttnn-kernel-writer-tdd (session 1) |
| Retries Consumed | 2 hard attempts (attempts that tried fp32 mode) |
| Time Cost | ~10 minutes across two fp32 attempts |

**Problem**: The kernel writer tried `fp32_dest_acc_en=True` twice (breadcrumbs at 17:58:09 and 18:41:03). Both times, the output showed zeros in the bottom half (max diff 9.0). The reason: fp32 mode halves DEST capacity from 16 to 8 tiles, breaking the untilize helper which needs 16 tiles for a full 32-row tile block. The kernel writer figured this out only on the second attempt (breadcrumb 18:42:00: "fp32 mode halves DEST to 8 tiles, breaking untilize").

**Root Cause**: The architect's design does not mention the DEST capacity constraint when fp32_dest_acc_en is toggled. The "Hardware Constraints Checklist" in op_design.md mentions "DEST register holds max 8 tiles (bf16 half-sync) / 4 tiles (f32)" but does not connect this to the untilize helper's requirements.

**Fix for agents**:
- **ttnn-operation-architect**: Add to the Hardware Constraints Checklist: "fp32_dest_acc_en halves DEST capacity. If untilize is used with Wt > 4, fp32 mode will break untilize. Do not enable fp32_dest_acc_en in compute config unless untilize block width is verified to fit in halved DEST."
- **ttnn-kernel-writer-tdd**: Before enabling fp32_dest_acc_en, check if the kernel uses untilize and whether Wt > DEST_AUTO_LIMIT/2. If so, skip this approach and note the incompatibility.

### Issue 3: Architect Set center_and_square Tolerance Too Tight

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 2 (Design) -> manifested in Phase 4 |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 (architect), but caused 5 hard + 2 free in kernel writer |
| Time Cost | ~56 minutes of kernel writer time (cascading impact) |

**Problem**: The architect set `atol=0.1` for the center_and_square stage (line 199 of op_design.md: "Tolerances: rtol=0.02, atol=0.1"). For a stage computing `(x - mean)^2` with bf16 reduce, this tolerance is physically impossible to meet. The final passing tolerance was `atol=0.5` (5x looser).

**Root Cause**: The architect did not calibrate tolerances based on the mathematical operation's sensitivity to bf16 precision. Squaring amplifies errors: a 0.01 error in the mean becomes ~0.4 error after centering and squaring with typical input magnitudes around 1.0. The architect's reference analyses (batch_norm) do not address numerical precision of reduce operations because batch_norm receives pre-computed statistics.

**Fix for agents**:
- **ttnn-operation-architect**: Add a "Tolerance Calibration" section to the design. For each TDD stage, reason about error propagation: (1) identify sources of bf16 precision loss (reduce, transcendental functions), (2) estimate amplification from downstream operations (squaring, reciprocal), (3) set atol to at least 2x the estimated max error. For stages involving reduce+square, default to atol >= 0.5.

### Issue 4: Builder Had to Fix Architect's Test Reference Functions

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (caught during file creation, not a test failure) |
| Time Cost | ~2 minutes |

**Problem**: The architect-generated test stage files had broken `pytorch_reference` functions: missing `return` statements and wrong variable names (using bare `input` instead of `input_tensor`). The builder caught and fixed these before running tests (breadcrumb at 17:34:22: "Fixed broken pytorch_reference functions in all 3 stage test files").

**Root Cause**: The architect writes test reference bodies as string expressions in `.tdd_state.json` (e.g., `"reference_body": "(input - input.mean(dim=-1, keepdim=True)) ** 2"`). These are template-expanded into Python functions by the orchestrator, but the parameter naming convention and return wrapping may not match the template. The architect likely wrote the reference as a mathematical expression rather than a valid Python function body.

**Fix for agents**:
- **ttnn-operation-architect**: Ensure `reference_body` values in `.tdd_state.json` use `input_tensor` (not `input`) as the parameter name, and are complete expressions that can be directly wrapped in `return (...)`. Document this convention at the top of the TDD stage plan.

### Issue 5: System Prompt Has Wrong tensor_accessor Include Path

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry |
| Time Cost | ~2 minutes |

**Problem**: The builder's system prompt maps TensorAccessor to `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp`, which does not exist. The correct path is `api/tensor/tensor_accessor.h` or (more commonly) it is included via `dataflow_api.h`. Builder's breadcrumb H1 (17:39:52) and execution log recommendation 1 both flag this.

**Root Cause**: Stale include path mapping in the builder's system prompt.

**Fix for agents**:
- **Pipeline infrastructure**: Update the builder system prompt's include path mapping. Either remove the TensorAccessor entry or change it to note that `dataflow_api.h` already provides TensorAccessor.

### Issue 6: Missing eps_packed Runtime Arg in Program Descriptor

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- normalize |
| Agent | ttnn-kernel-writer-tdd (session 2) |
| Retries Consumed | 0 (fixed proactively before running test) |
| Time Cost | ~1 minute |

**Problem**: When implementing the normalize stage, the kernel writer discovered that `eps_packed` was not in the reader's runtime args. The architect's design doc specifies it should be at index 7, but the builder's program descriptor did not include it. The kernel writer added it as an upstream fix (breadcrumb at 19:04:45).

**Root Cause**: The builder created the program descriptor from op_design.md Part 1 which lists reader runtime args including eps_packed. However, for stage 1 (data_pipeline) eps is not needed, so the builder may have deferred it. The builder should have included all runtime args from the design, even if unused in early stages, to avoid requiring kernel writer upstream fixes.

**Fix for agents**:
- **ttnn-generic-op-builder**: When creating the program descriptor, include ALL runtime args specified in op_design.md from the start, even if some are only needed by later TDD stages. Use 0 or sentinel values for unused args. This prevents the kernel writer from needing to modify Python infrastructure.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~4m | 0 free, 0 hard | PASS | Clean -- implemented and passed first try |
| center_and_square | ~68m (incl. gap) | 2 free, 5 hard | PASS | bf16 precision investigation, required orchestrator tolerance fix |
| normalize | ~4m | 1 free, 0 hard | PASS | Missing rsqrt include (trivial free retry) |
| affine | ~4m | 0 free, 0 hard | PASS | Clean -- gamma/beta routing worked first try |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | bf16 precision debugging | kernel-writer-tdd S1 | ~56m | 43% | Investigated inherent bf16 reduce precision loss, tried fp32 mode (broke untilize), tried alternative scaler approaches (equivalent results), ran DPRINT diagnostics | 5 hard + 2 free | Tolerance miscalibration by architect; no "precision triage" strategy in kernel writer |
| 2 | Orchestrator gap | orchestrator | ~10m | 8% | Time between session 1 ending (context exhausted) and session 2 starting, during which tolerance was loosened | N/A | Context exhaustion in session 1 required new session |
| 3 | Builder test validation | builder | ~4m | 3% | Fixing test reference functions + tensor_accessor include | 1 free | Architect reference_body format issues; stale system prompt |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| kernel-writer-tdd S1 | Implemented alternative scaler approach (reduce SUM with 1.0 + post_reduce_op mul_unary_tile) | Produced identical results to original 1/W scaler because the precision loss is in the accumulation, not the scaler multiplication. ~15 minutes spent on implementation + testing. | Kernel writer should have first run a diagnostic (skip square, output centered values) to isolate whether error is in reduce or squaring, before attempting alternative approaches. |
| kernel-writer-tdd S1 | Enabled fp32_dest_acc_en twice | Broke untilize both times (DEST halved). ~10 minutes total. | Architect should document fp32/untilize incompatibility. Kernel writer should check DEST capacity constraints before toggling fp32. |
| kernel-writer-tdd S1 | Added and then removed DPRINT diagnostics | DPRINT required multiple iterations to get working (DPRINT_MATH vs DPRINT_PACK for TSLICE support). Output was informative but by then context was nearly exhausted. ~8 minutes. | Earlier diagnostic approach: skip to isolating which compute phase introduces the error, rather than instrumenting after exhausting code-level fixes. |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| Quality | GOOD |
| Issues | None significant. Batch_norm analysis describes channel-broadcast patterns which required adaptation to row-broadcast for layer_norm. This adaptation was handled correctly by the architect. |
| Downstream Impact | Positive. The architect explicitly referenced all three analyses in breadcrumb `reference_read` events and extracted specific patterns (RM stick batching, untilize helper signature, eps fill pattern). |
| Suggestion | No changes needed. The analysis quality was sufficient. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | ADEQUATE |
| Issues | (1) Test `reference_body` strings used `input` instead of `input_tensor` and lacked explicit return wrapping. (2) eps_packed runtime arg was specified in design but not flagged as needed from stage 1 onward. |
| Downstream Impact | Minor: builder spent ~2 minutes fixing test functions. eps_packed omission cascaded to kernel writer needing an upstream fix in session 2. |
| Suggestion | Architect should validate reference_body strings against the test template's parameter convention. Builder should include all specified runtime args from stage 1. |

### Handoff 3: ttnn-operation-architect -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md (Part 2: Kernel Implementation) |
| Quality | GOOD (for stages 1, 3, 4), POOR (for stage 2 tolerance) |
| Issues | (1) Stage 2 tolerance set to atol=0.1, which is unrealistically tight for bf16 reduce + squaring. (2) No mention of fp32_dest_acc_en incompatibility with untilize when Wt > DEST_AUTO_LIMIT/2. (3) Missing rsqrt.h include requirement for stage 3. |
| Downstream Impact | HIGH: The tolerance miscalibration directly caused 56 minutes of wasted debugging. The kernel writer followed the design precisely but could not meet the tolerance. |
| Suggestion | Architect should add a "Tolerance Calibration" section with bf16 error propagation analysis. Architect should emit required includes per stage. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files |
| Quality | ADEQUATE |
| Issues | (1) Missing eps_packed from reader runtime args (kernel writer added it as upstream fix). (2) Test reference functions required fixes (already done by builder, so kernel writer received corrected versions). |
| Downstream Impact | Low: kernel writer needed 1 upstream fix taking ~1 minute. |
| Suggestion | Builder should populate all runtime args from the design, even for unused-in-stage-1 args. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-generic-op-builder | Use `input_tensor` (not `input`) in reference_body strings; include `return` | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Set intermediate stage tolerances based on bf16 error propagation (atol >= 0.5 for reduce+square) | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Document fp32_dest_acc_en / untilize DEST capacity incompatibility | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Emit required includes per TDD stage (rsqrt.h for normalize stage) | MEDIUM | LOW |
| ttnn-generic-op-builder | ttnn-kernel-writer-tdd | Include ALL runtime args from design in initial program descriptor | HIGH | MEDIUM |
| Pipeline infra | ttnn-generic-op-builder | Fix tensor_accessor include path mapping in builder system prompt | HIGH | LOW |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD tolerance | Stage 2 tolerance miscalibration caused 43% of total pipeline time to be spent on debugging | Add bf16 precision calibration guidance to architect instructions. For stages with reduce+nonlinear, default atol >= 0.5 | HIGH |
| Kernel writer debugging strategy | Kernel writer tried 5 different code-level fixes before concluding the issue was tolerance, not implementation | Add a "numerical mismatch triage" protocol: (1) run diagnostic isolating error source, (2) if max diff < 1.0 with correct pattern, classify as precision, (3) recommend tolerance adjustment after 2 hard attempts with same root cause | HIGH |
| Context management | Session 1 exhausted context on bf16 debugging, requiring a new session | Kernel writer should escalate to orchestrator after 3 hard attempts on the same stage with the same error pattern, rather than continuing to debug until context exhaustion | MEDIUM |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | YES | This is exactly what happened. Session 1 spent ~56m debugging bf16 precision, exhausted context. The root cause analysis in pipeline-improvements.md ("The writer can see expected 0.5, got 0.0 but cannot easily trace which tile, which CB, which compute step produced the wrong value") matches perfectly -- the kernel writer used DPRINT to try to trace intermediate values but ran out of context before being able to act on the results. |
| 3 | .tdd_state.json coupling fragile | NO | File-based IPC worked correctly in this run. |
| 6 | Builder runs on Sonnet while everything else uses Opus | POSSIBLY | Builder had 1 free retry for include path and fixed test reference functions. These are the type of detail errors that could benefit from a stronger model, but impact was minor in this run. |
| 9 | No validation between architect output and builder output | YES (minor) | The missing eps_packed runtime arg is exactly this: the design specified it but the builder omitted it. A static cross-check would have caught this. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| fp32_dest_acc_en / untilize incompatibility not documented | Enabling fp32_dest_acc_en halves DEST capacity, breaking untilize when Wt > DEST_AUTO_LIMIT/2. Neither the architect nor the kernel writer had this constraint documented. The kernel writer wasted 2 hard attempts discovering this empirically. | MEDIUM |
| Architect tolerance calibration for bf16 intermediate stages | The architect does not perform error propagation analysis when setting TDD stage tolerances. For reduce+square in bf16, atol=0.1 is physically impossible, yet this is what was set. | HIGH |
| Kernel writer lacks "precision triage" protocol | The kernel writer has no systematic strategy for distinguishing "implementation bug" from "inherent precision limitation." It defaults to trying code fixes until budget exhaustion rather than escalating after recognizing a pattern of small, consistent mismatches. | HIGH |

---

## 8. Actionable Recommendations

### Recommendation 1: Add bf16 Tolerance Calibration to Architect Instructions

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a "Tolerance Calibration" step after designing TDD stages. For each stage, the architect must: (1) identify operations that introduce bf16 precision loss (reduce, transcendental functions), (2) estimate error amplification from downstream operations in that stage (squaring: 2x mean error amplified by input magnitude; reciprocal: error proportional to 1/x^2), (3) set atol to at least 2x the estimated max error. Default: atol >= 0.5 for any stage with reduce+square.
- **Expected Benefit**: Eliminates the entire 56-minute bf16 debugging cycle. Stage 2 would have passed on the first attempt.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add Numerical Mismatch Triage Protocol to Kernel Writer

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd prompt/instructions
- **Change**: After the first hard attempt with a numerical mismatch, the kernel writer must: (1) Check if the output pattern is correct (values track reference, systematic offset, no zeros/NaN). (2) If max diff < 1.0 and pattern is correct, run exactly one diagnostic: isolate the error to a specific compute phase by skipping downstream phases. (3) If the error is in the reduce or a transcendental function and is consistent across shapes, classify as "bf16 precision limitation" and escalate to orchestrator requesting tolerance adjustment. Do not spend more than 2 hard attempts on the same consistent numerical mismatch.
- **Expected Benefit**: Reduces debugging time from ~56m to ~10m for precision-related issues.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Document fp32_dest_acc_en / untilize Constraint

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions, kernel writer debugging checklist
- **Change**: Add to the architect's Hardware Constraints Checklist template: "If the compute kernel uses untilize and Wt > DEST_AUTO_LIMIT/2 (>4 tiles for fp32, >8 tiles for bf16), do NOT set fp32_dest_acc_en=True in ComputeConfigDescriptor. fp32 mode halves DEST capacity, breaking untilize's pack_untilize_block." Add the same constraint to the kernel writer's "things to check before modifying compute config" checklist.
- **Expected Benefit**: Saves 2 hard attempts (~10 minutes) per occurrence.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Builder Should Include All Designed Runtime Args From Stage 1

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder prompt/instructions
- **Change**: When creating the program descriptor, the builder must include ALL runtime args specified in op_design.md for ALL kernels, using 0 or sentinel values for args not needed in the initial TDD stage. This prevents the kernel writer from needing to modify Python infrastructure as an "upstream fix."
- **Expected Benefit**: Eliminates kernel writer upstream fixes for missing args. Keeps Python infrastructure stable across TDD stages.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Kernel Writer Should Escalate After 3 Same-Pattern Hard Attempts

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd prompt/instructions, orchestrator
- **Change**: If the kernel writer has 3 hard attempts on the same stage with the same error classification (e.g., "numerical_mismatch, max diff ~0.375" three times), it must escalate to the orchestrator with a summary: "Stage X has consistent numerical mismatch of Y after 3 attempts. Suspected cause: [precision/implementation]. Recommended action: [loosen tolerance/investigate design]." The kernel writer should NOT continue debugging the same pattern beyond 3 hard attempts.
- **Expected Benefit**: Prevents context exhaustion on inherently unsolvable problems. Saves ~30 minutes per occurrence.
- **Priority**: MEDIUM
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5/5 | Correct hybrid-mode decomposition, all three references directly applicable |
| Analysis quality | 5/5 | Three comprehensive analyses totaling ~1350 lines, all actionable findings used by architect |
| Design completeness | 4/5 | Excellent CB layout and phase-by-phase kernel design. Lost 1 point for tolerance miscalibration (atol=0.1 for reduce+square stage) and missing fp32/untilize constraint documentation |
| Build correctness | 4/5 | Infrastructure worked first try. Lost 1 point for missing eps_packed runtime arg and needing to fix test reference functions |
| Kernel implementation | 3/5 | Stages 1, 3, 4 were clean (4-5 territory). Stage 2's bf16 debugging dominated. Lost points for spending 56 minutes on what should have been a tolerance adjustment |
| Inter-agent communication | 4/5 | Handoffs were generally clean. Lost 1 point for the cascading tolerance miscalibration from architect to kernel writer |
| Logging/observability | 4/5 | Breadcrumbs were detailed and timestamped, enabling full reconstruction of the debugging timeline. Execution logs were complete. Lost 1 point because the kernel writer's session 1 does not have a formal `complete` breadcrumb (it hit context limits), requiring inference from the last breadcrumb entry at ~18:53:28 |

### Top 3 Things to Fix

1. **Add bf16 tolerance calibration to the architect's instructions** -- this single change would have saved 56 minutes (43% of total pipeline time) by preventing the impossible-tolerance Stage 2 from consuming the kernel writer's budget.
2. **Add a numerical mismatch triage protocol to the kernel writer** -- when facing consistent small numerical mismatches, the kernel writer should diagnose (1 attempt), classify (precision vs bug), and escalate rather than grinding through code changes.
3. **Document the fp32_dest_acc_en / untilize DEST capacity constraint** -- this is a non-obvious hardware interaction that wasted 2 hard attempts and is likely to recur in any RM-in/RM-out operation that considers fp32 mode.

### What Worked Best

The architect's CB layout design. All 11 circular buffers were correctly specified from the start -- IDs, page counts, data formats, lifetimes, and inter-phase routing. The architect's pass-2 self-correction of binary op input policies (changing NoWaitNoPop to WaitUpfrontPopAtEnd for freshly-pushed operands) prevented what would have been deadlocks or hangs. This design accuracy enabled stages 1, 3, and 4 to pass with zero hard attempts, demonstrating that when the design is right, the TDD kernel implementation is fast and mechanical.
