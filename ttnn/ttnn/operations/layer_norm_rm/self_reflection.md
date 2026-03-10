# Self-Reflection: layer_norm_rm (Run 2)

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 12 (for this run) |
| Total Pipeline Duration | ~80 minutes (11:53 - 13:14 UTC) |
| Overall Result | SUCCESS -- All 5 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~2m (pre-11:53) | Completed | Selected 3 references: tilize, reduce_w, untilize. Switch from batch_norm (run1) to reduce_w as compute reference was a significant improvement. |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~26m (11:53 - 12:19) | Completed | 3 analyzers ran sequentially (tilize: ~10m, reduce_w: ~9m, untilize: ~5m). Produced 3 analysis files totaling ~25KB each. |
| 2: Design | ttnn-operation-architect | ~7m (12:21 - 12:28) | Completed | Single pass, no revisions needed. Produced comprehensive 466-line op_design.md with 15 CB layout and 10 compute phases. |
| 3: Build | ttnn-generic-op-builder | ~11m (12:31 - 12:42) | Completed | 1 compile failure (include paths), then 7/7 tests pass. Created 12 files. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~29m (12:44 - 13:13) | Completed | 5/5 stages passed. 1 hard attempt consumed (reduce_mean numerical mismatch). |
| 5: Report | orchestrator | ~2m (13:12 - 13:14) | Completed | REPORT.md generated. |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 11:53:18 | 12:03:42 | ~10m 24s | 0 | ~10m active (read + DeepWiki + write) |
| ttnn-operation-analyzer (reduce_w) | 12:04:46 | 12:13:27 | ~8m 41s | 0 | ~8m active (read + DeepWiki + write) |
| ttnn-operation-analyzer (untilize) | 12:14:13 | ~12:19:24 | ~5m 11s | 0 | ~5m active |
| ttnn-operation-architect | 12:21:03 | 12:28:19 | ~7m 16s | 0 | ~7m active (read analyses + design + write) |
| ttnn-generic-op-builder | 12:31:15 | 12:42:04 | ~10m 49s | 1 (include fix) | ~8m active, ~3m on include path fix |
| ttnn-kernel-writer-tdd | 12:44:26 | ~13:11:24 | ~27m | 1 (reduce_mean) | ~20m productive, ~7m debugging reduce_mean |

**Duration calculation method**: Used breadcrumb `"event":"start"` and `"event":"complete"` timestamps where available. For kernel-writer-tdd stages 2-5, breadcrumb timestamps are unreliable (placeholder values like `12:00:00Z`, `12:01:00Z`); git commit timestamps were used as fallback for those stages. Git commit for data_pipeline at 12:48:52 and affine at 13:11:24 bracket the kernel-writer TDD phase.

### Duration Visualization

```
Phase 0  |##|                                                        (~2m)
Phase 1  |#########################|                                 (~26m) 3 analyzers sequential
Phase 2                             |######|                         (~7m)
Phase 3                                     |##########|             (~11m)
Phase 4                                                |############################| (~29m)
Phase 5                                                                              |##| (~2m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75 80 min

Longest phase: Phase 4 (~29m) -- 5 TDD stages with 10 compute phases each, 1 numerical debugging cycle
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~28m | 35% | 3 analyzers, sequential execution |
| Design (Phase 2) | ~7m | 9% | Single pass, no revisions |
| Build (Phase 3) | ~11m | 14% | 1 include path retry |
| Kernel implementation (Phase 4) | ~29m | 36% | 5 TDD stages |
| -- Productive coding | ~22m | 28% | Writing kernel code that passed |
| -- Debugging/retries | ~7m | 8% | reduce_mean numerical mismatch investigation |
| Reporting (Phase 5) | ~2m | 3% | |
| **Total** | **~80m** | **100%** | |

---

## 2. What Went Well

### 1. Switching from batch_norm to reduce_w as compute reference

**Phase/Agent**: Phase 0 (Discovery) / orchestrator
**Evidence**: Run 1 used `batch_norm` as compute_core reference and required 3 hard attempts in the kernel-writer. Run 2 used `reduce_w` which directly demonstrates the exact REDUCE_ROW pattern needed, resulting in only 1 hard attempt. The reduce_w analysis file provided precise helper signatures (`reduce<SUM, REDUCE_ROW>`, scaler CB format, `WaitAndPopPerTile` policy) that the architect consumed directly.
**Why it worked**: `reduce_w` is a more targeted reference than `batch_norm` for the row-reduction pattern. It showed the exact reduction helper API, scaler preparation, and data format reconfiguration that layer_norm_rm needs.

### 2. Four out of five TDD stages passed on first attempt

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `attempts: 0` for stages data_pipeline, subtract_mean, variance_inv_std, and affine. Only reduce_mean required 1 hard attempt. Zero free retries consumed across all 5 stages.
**Why it worked**: The architect's design document was exceptionally detailed, with explicit CB state tables after each phase, exact helper template parameters, manual pop instructions, and broadcast verification tables. The kernel writer could translate the design almost directly into code.

### 3. CB layout designed correctly on first pass with zero CB-related bugs

**Phase/Agent**: Phase 2 (Design) / ttnn-operation-architect
**Evidence**: 15 CBs were allocated in the design. The kernel-writer produced the final `layer_norm_rm_compute.cpp` with all 15 CBs matching the design exactly (IDs 0-7, 16, 24-29). No CB sizing errors, no page count mismatches, no data format issues emerged during TDD. The binary op broadcast verification table (Section: Binary Op Broadcast Verification in op_design.md) correctly identified COL broadcast for mean/inv_std and ROW broadcast for gamma/beta.
**Why it worked**: The architect validated CB decisions against helper library requirements during design, as evidenced by `helper_analysis` breadcrumb entries for all 6 helper files read before design decisions were made.

### 4. Clean multi-core work distribution handled by kernel-writer

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: The kernel-writer proactively split the compute kernel into two `KernelDescriptor` instances for `core_group_1` and `core_group_2` during stage data_pipeline (breadcrumb: `upstream_fix` at 12:47:44). This fix was applied before the first test run, preventing a potential hang or incorrect behavior on cliff cores.
**Why it worked**: The kernel-writer recognized that `nblocks_per_core` is a compile-time arg and must differ between core groups. This shows good understanding of the hardware programming model.

### 5. No device hangs throughout entire TDD pipeline

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: All 5 stages completed without any device hangs. No `tt-smi -r` resets were needed. All test runs produced results (pass or numerical mismatch), never a timeout.
**Why it worked**: The CB synchronization was correct from the start. The design's explicit CB state tables after each phase (Phases 2, 3, 4, 7 in op_design.md) made push/pop balance verification straightforward. The kernel-writer performed `cb_sync_check` breadcrumb entries for every stage.

---

## 3. Issues Found

### Issue 1: Reduce mean numerical mismatch due to REDUCE_ROW output format misunderstanding

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- reduce_mean (Stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~7 minutes |

**Problem**: After implementing reduce_mean, the test showed a numerical mismatch with max_diff=0.263. The kernel writer's breadcrumb (H1 at 12:53:27) diagnosed it: "REDUCE_ROW output tile has valid data only in Col0 of each face. After untilize, only element[0] of each 32-element stick has the mean value, rest are zeros." The test expected the mean broadcast across all 32 columns, but REDUCE_ROW only fills Col0.

**Root Cause**: The stage 2 test reference was `x.mean(dim=-1, keepdim=True).expand(x.shape[0], x.shape[1], x.shape[2], 32)` -- expecting the mean replicated across all 32 columns of the output. However, the kernel's REDUCE_ROW output naturally only fills Col0 of the tile. The architect's design noted "output is `x.mean(dim=-1, keepdim=True)` with output shape `(B,C,H,1)` padded to tile alignment `(B,C,H,32)`" but did not specify whether the padded columns should contain the mean or zeros. The test and kernel disagreed on this point.

The fix was creative: the kernel writer generated a zeros tile in c_24 and used `add<COL>(zeros, mean)` to broadcast the Col0 mean values across all columns before untilizing. This worked but added complexity that was later removed in stage 3 (where the mean is consumed by `sub<COL>` directly, which handles the Col0-only format natively).

**Fix for agents**:
- **ttnn-operation-architect**: For intermediate TDD stages that output reduced tiles (REDUCE_ROW, REDUCE_COL), explicitly document whether the test reference expects valid-region-only output or broadcast output. Include a note like "REDUCE_ROW output: only Col0 valid. Test reference should compare against Col0 extraction, not broadcast."
- **ttnn-kernel-writer-tdd**: Before implementing a reduce-output stage, check whether the test reference expects broadcast values or valid-region-only values, and choose the simpler approach (modifying the test reference to match kernel output is cheaper than adding broadcast logic to the kernel).

### Issue 2: Builder used wrong include paths from architect's design

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry (compile error) |
| Time Cost | ~3 minutes |

**Problem**: The builder's first test run failed with `fatal error: ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp: No such file or directory`. The correct device-side path is `api/tensor/tensor_accessor.h`. The builder also had to comment out `binary_op_helpers.hpp` which was referenced in the design but not installed in the device build directory.

**Root Cause**: The architect's design document uses host-side include paths (or references headers that exist in source but are not installed for device compilation). The builder's execution log explicitly calls this out as upstream feedback: "Include paths in design use host-side conventions" (Section 1, Upstream Feedback table).

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation step: for every `#include` mentioned in the design, verify the path exists under `build_Release/libexec/tt-metalium/`. Alternatively, include a "Known Device-Side Include Paths" reference table.
- **Pipeline**: Add a static include-path validator that the builder runs before the first compilation attempt.

### Issue 3: Kernel-writer breadcrumb timestamps are unreliable for stages 2-5

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- all TDD stages after data_pipeline |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 (observability issue, not runtime) |
| Time Cost | 0 (but degrades self-reflection analysis quality) |

**Problem**: The kernel-writer breadcrumbs for stages reduce_mean through affine have placeholder-like timestamps (`2026-03-10T12:00:00Z`, `12:01:00Z`, `12:02:00Z`, etc.). These are clearly not real wall-clock times -- they increment by exactly 1 minute and several predate the agent's own start time (12:44:26). This makes it impossible to compute accurate per-stage durations from breadcrumbs alone.

By contrast, the data_pipeline stage breadcrumbs (entries 3-10) have real timestamps (12:45:43, 12:46:34, 12:47:05, etc.) that correlate with the git commit at 12:48:52.

**Root Cause**: The kernel-writer agent likely switched to a different breadcrumb emission method after the first stage, or the breadcrumb helper was called with synthetic timestamps for batch-logging events after the fact rather than in real-time.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Always emit breadcrumbs in real-time with actual `Date.now()` / shell `date -Iseconds` timestamps. Never batch-emit breadcrumbs with synthetic timestamps after a stage completes.
- **Pipeline**: Add a breadcrumb validation step that checks for monotonically increasing timestamps and flags entries where `ts < agent_start_ts`.

### Issue 4: Architect design contains deliberation text that could confuse the kernel writer

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2 -- Design |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 (did not cause a failure this run) |
| Time Cost | 0 |

**Problem**: The op_design.md Phase 5 section contains visible deliberation/revision text that was not cleaned up:

Lines 336-348 of op_design.md:
```
    // Actually: add_tiles reads from DEST[dst_idx] and CB[c_7, tile 0]
    // Correction: We cannot add_tiles from DEST. Need different approach.

**REVISED Phase 5 approach**: The post_reduce_op runs while result is in DEST, but `add_tiles` requires both operands in CBs. Instead, we split into: reduce -> pack to temp -> add_eps -> rsqrt -> pack to c_6.

**Actually, the simplest approach**: Use the reduce with scaler=1/W...
```

This "thinking out loud" text showing the architect reconsidering approaches is preserved in the final design document. While the kernel writer in this run correctly identified the final approach (Phase 5 reduce + Phase 6 add+rsqrt separately), leaving deliberation text in the design document increases confusion risk.

**Fix for agents**:
- **ttnn-operation-architect**: After completing the design, perform a cleanup pass that removes any deliberation text (lines starting with "Actually", "Correction", "REVISED approach"). The final document should present only the chosen approach, not the journey to get there.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~4m (12:44 - 12:49) | 0 free, 0 hard | PASS | Clean -- reader/compute/writer all implemented correctly on first pass. Proactive upstream fix (dual compute kernels for core groups). |
| reduce_mean | ~9m (12:49 - ~12:58) | 0 free, 1 hard | PASS | Numerical mismatch on REDUCE_ROW Col0-only output. Fixed by broadcasting mean via add<COL> with zeros tile. |
| subtract_mean | ~5m (12:58 - 13:02) | 0 free, 0 hard | PASS | Clean -- added sub<COL> phase, removed zeros broadcast workaround from stage 2, restored full-width output. |
| variance_inv_std | ~5m (13:02 - 13:07) | 0 free, 0 hard | PASS | Clean -- added 5 compute phases (square, reduce_var, add_eps+rsqrt, mul_inv_std) in one pass. |
| affine | ~4m (13:07 - 13:11) | 0 free, 0 hard | PASS | Clean -- added gamma/beta read/tilize/mul/add phases. |

Note: Stage durations estimated from git commit timestamps since kernel-writer breadcrumbs have unreliable timestamps for stages 2-5.

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Analysis phase | ttnn-operation-analyzer | ~26m | 33% | Three sequential analyzer runs, each spending significant time on DeepWiki queries and doc reads | 0 | Analyzers ran sequentially not in parallel; each analyzer took 5-10 minutes |
| 2 | reduce_mean debugging | ttnn-kernel-writer-tdd | ~7m | 9% | Numerical mismatch investigation and fix for REDUCE_ROW output format | 1 hard | Architect did not clearly specify REDUCE_ROW valid region behavior in test reference |
| 3 | Builder include fix | ttnn-generic-op-builder | ~3m | 4% | Fixing host-side to device-side include paths | 1 free | Known issue (pipeline-improvements.md does not track this specifically, but builder execution log recommends fixing) |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | Stage 2: Generated zeros tile in c_24 and implemented add<COL> broadcast for mean | This broadcast logic was removed in Stage 3, where sub<COL> consumes the Col0-only mean directly | Architect should design stage 2 test to compare against Col0-only output, avoiding the need for broadcast |
| ttnn-kernel-writer-tdd | Stage 2: Modified output shape to (B,C,H,32) and separated input/output Wt | These upstream fixes were reverted in Stage 3 when output returned to full input width | This is inherent to incremental TDD -- intermediate stages may need temporary infrastructure that gets reverted. Could be reduced by designing stage 2 to output full-width data (use sub<COL>(zeros, mean) as identity + mean broadcast) |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer --> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | `tilize_analysis.md`, `reduce_w_analysis.md`, `untilize_analysis.md` |
| Quality | GOOD |
| Issues | Analyses averaged ~25KB each, which is larger than necessary. The architect distilled them into a few key findings per reference. |
| Downstream Impact | Minimal -- architect successfully extracted what it needed. |
| Suggestion | Consider enforcing a structured output format with explicit sections (CB layout, helper API signatures, key patterns) and a strict length limit (~10KB). |

### Handoff 2: ttnn-operation-architect --> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md`, `.tdd_state.json` |
| Quality | ADEQUATE |
| Issues | 1) Include paths used host-side conventions (caused 1 compile failure). 2) Deliberation text left in design document. 3) Test templates generated by tdd_orchestrator.py had quality issues (noted in architect's execution log Recommendation 1). |
| Downstream Impact | Builder spent ~3 minutes fixing include paths. Builder also had to rewrite test files from architect's auto-generated templates. |
| Suggestion | Architect should validate include paths against device build directory. Architect should clean up deliberation text before committing. |

### Handoff 3: ttnn-generic-op-builder --> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files |
| Quality | GOOD |
| Issues | The builder's handoff notes (execution log Section 6) explicitly warned about `binary_op_helpers.hpp` not being available, correct include paths, and CB reuse in Phase 6. These warnings were helpful and actionable. |
| Downstream Impact | The kernel-writer was able to implement all 5 stages with only 1 numerical debugging cycle. The builder's dual-core-group split was proactively fixed by the kernel-writer in stage 1 (before first test). |
| Suggestion | The builder should also validate that its stub kernels compile against the device build, not just against the generic_op infrastructure. |

### Handoff 4: ttnn-operation-architect --> ttnn-kernel-writer-tdd (design doc)

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md` Part 2 (Kernel Implementation) |
| Quality | GOOD |
| Issues | 1) Deliberation text in Phase 5 section (see Issue 4). 2) REDUCE_ROW output valid region not explicitly tied to stage 2 test expectations. 3) The design mentions `calculate_and_prepare_reduce_scaler<c_2, SUM, REDUCE_ROW, 32, W>()` but the kernel writer used `prepare_reduce_scaler<cb_reduce_scaler>(1.0f/W)` -- a simpler approach that worked. |
| Downstream Impact | The kernel writer correctly chose the simpler scaler preparation approach. The deliberation text did not cause confusion this time. |
| Suggestion | Design should present only the final chosen approach per phase. For helper functions, prefer the simplest viable API call. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-generic-op-builder | Use device-side include paths (e.g., `api/tensor/tensor_accessor.h`) not host-side paths | HIGH | HIGH |
| ttnn-operation-architect | ttnn-generic-op-builder | Verify binary_op_helpers.hpp availability in device build before referencing | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-operation-architect | Fix auto-generated test templates (missing returns, wrong variable names, missing commas) | HIGH | HIGH |
| ttnn-operation-architect | ttnn-operation-architect | Clarify reduce scaler setup: SUM with 1/W scaler == AVG | MEDIUM | LOW |
| ttnn-kernel-writer-tdd | ttnn-kernel-writer-tdd | For reduce-output stages, check test reference expectations before adding broadcast logic | HIGH | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Analysis | 3 analyzers ran sequentially (~26m total). Could be parallelized. | Run analyzer instances in parallel using backgrounded subagent calls. Would save ~15-20m. | HIGH |
| Logging | Kernel-writer breadcrumb timestamps are unreliable after stage 1 (placeholder values) | Add breadcrumb timestamp validation; require real-time emission | MEDIUM |
| Build | Include path errors are recurring (seen in builder execution log and known from prior runs) | Create a static include-path reference table that agents consult, or add a pre-compilation validation step | MEDIUM |
| TDD | Stage 2 (reduce_mean) test/kernel mismatch on REDUCE_ROW valid region | Architect should specify exact output format per stage, including valid tile regions | MEDIUM |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | PARTIALLY | Only 1 numerical mismatch consuming ~7 minutes. Far less severe than described in the issue (which mentions hour-long sessions). The simpler reduce_w reference and high-quality design doc prevented extended debugging. |
| 2 | Too many planning stages (long leash) | NO (DONE) | Architect agent successfully merged planner+designer. Single design pass worked well. |
| 3 | `.tdd_state.json` coupling fragility | NO | No schema mismatches observed. File was consumed correctly by builder and kernel-writer. |
| 4 | No fast path for simple operations | N/A | layer_norm_rm is not a simple operation (10 compute phases, 15 CBs). Full pipeline was appropriate. |
| 6 | Builder runs on Sonnet | POSSIBLY | Builder had 1 include path error but otherwise performed well. Cannot determine model from logs. |
| 7 | Discovery keyword matching | NO | Discovery correctly identified tilize/reduce_w/untilize references. Switching from batch_norm to reduce_w between runs was the right call. |
| 9 | No architect/builder cross-validation | YES | Include path mismatch between architect's design and device build environment was not caught until builder tried to compile. A cross-validation step would have caught this. |
| 11 | No incremental re-run capability | NO | Full pipeline completed without needing to resume. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Kernel-writer breadcrumb timestamps become unreliable after first TDD stage | Stages 2-5 breadcrumbs use synthetic/placeholder timestamps (12:00:00Z, 12:01:00Z, ...) that predate the agent's own start time. Makes per-stage timing analysis impossible from breadcrumbs alone. | MEDIUM |
| Architect leaves deliberation text in design document | op_design.md Phase 5 contains "Actually", "Correction", "REVISED approach" text showing the architect's reasoning process rather than the final decision. Could confuse downstream agents. | LOW |
| Analyzer sequential execution wastes ~15-20 minutes | Three analyzers ran one after another (~26m total) when they could have run in parallel (~10m). | HIGH |

---

## 8. Actionable Recommendations

### Recommendation 1: Parallelize analyzer execution

- **Type**: pipeline_change
- **Target**: Orchestrator / Phase 1 execution logic
- **Change**: Launch all 3 analyzer instances concurrently using backgrounded subagent calls. Wait for all to complete before starting Phase 2.
- **Expected Benefit**: Reduce Phase 1 from ~26m to ~10m (bounded by the slowest analyzer, tilize at ~10m). Saves ~16m per pipeline run.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add device-side include path reference table for architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: Add a table mapping common components to their device-side include paths: `TensorAccessor -> api/tensor/tensor_accessor.h`, `dataflow_api -> api/dataflow/dataflow_api.h`, `compute_hw_startup -> api/compute/compute_kernel_hw_startup.h`. Instruct architect to use only device-side paths in the design document.
- **Expected Benefit**: Eliminate include-path compilation failures in builder phase. Saves ~3m per run.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Require architect to specify test output format for reduce stages

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions, TDD stage registration
- **Change**: For any TDD stage that outputs a reduced tensor (REDUCE_ROW, REDUCE_COL), the architect must explicitly specify: (a) the valid tile region of the output, (b) whether the test reference should compare against the valid region only or broadcast values, and (c) the recommended test reference expression. Example: "Stage 2 output: Col0 valid only. Test reference: `x.mean(-1, keepdim=True)` compared against `output[:,:,:,0:1]`."
- **Expected Benefit**: Prevent the reduce_mean numerical mismatch pattern. Saves 1 hard attempt and ~7m per layer_norm-class operation.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Enforce real-time breadcrumb emission in kernel-writer

- **Type**: tool_improvement
- **Target**: `append_breadcrumb.sh` or kernel-writer agent instructions
- **Change**: Modify `append_breadcrumb.sh` to always generate timestamps server-side (using `date -Iseconds`) rather than accepting them from the caller. Alternatively, add a validation rule: if `ts` in the JSON is older than the most recent breadcrumb's `ts`, reject or warn.
- **Expected Benefit**: Reliable per-stage timing data for self-reflection analysis.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Architect cleanup pass to remove deliberation text

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: Add a final step before committing op_design.md: "Review the document for any deliberation text (lines containing 'Actually', 'Correction', 'REVISED', 'Note:', or strikethrough patterns). Remove all such text, keeping only the final chosen approach for each phase."
- **Expected Benefit**: Cleaner design documents that present only decisions, not the reasoning journey. Reduces confusion risk for kernel-writer.
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5 | Correctly identified tilize/reduce_w/untilize. The switch from batch_norm to reduce_w (vs run 1) was a major improvement. |
| Analysis quality | 4 | Analyses were comprehensive and useful. Slightly too verbose (~25KB each). reduce_w analysis provided the exact helper signatures needed. |
| Design completeness | 4 | 466-line design with explicit CB state tables, broadcast verification, and 10-phase compute plan. Deducted for deliberation text and imprecise reduce stage output specification. |
| Build correctness | 4 | 12 files created correctly. One include-path error (quickly fixed). Dual core-group compute kernels not initially handled (fixed by kernel-writer). |
| Kernel implementation | 5 | 4/5 stages on first attempt. Only 1 hard attempt total. Clean compute kernel matching design. No hangs. |
| Inter-agent communication | 4 | Good handoff notes from builder. Architect's design was highly usable. Include path mismatch is the main gap. |
| Logging/observability | 3 | Breadcrumbs present for all agents but kernel-writer timestamps are unreliable for stages 2-5. Execution logs well-structured for architect and builder. No execution log for kernel-writer. |

### Top 3 Things to Fix

1. **Parallelize analyzer execution** -- Saves ~16 minutes per run with minimal effort. Currently the single largest time sink after TDD implementation.
2. **Add device-side include path reference table** -- Recurring issue across runs. Include path errors are a known, easily preventable failure mode.
3. **Specify reduce stage test output format explicitly** -- Prevents the most common numerical mismatch pattern (REDUCE_ROW Col0-only valid region vs broadcast test reference).

### What Worked Best

The architect's design document quality was the standout strength of this run. The 466-line op_design.md with explicit CB state tables after each compute phase, binary op broadcast verification, TDD stage plan with delta-from-previous descriptions, and critical notes section enabled the kernel-writer to implement a 10-phase, 15-CB compute kernel with only 1 numerical debugging cycle. Four out of five TDD stages passed on the first attempt, and the final kernel code is a clean, almost literal translation of the design document. This validates that investing heavily in design quality pays off in kernel implementation speed.
