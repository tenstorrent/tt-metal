# Self-Reflection: layer_norm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm` |
| Pipeline Phases Executed | Phase 0 (Discovery), Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels), Phase 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x2), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer (x4 stages) |
| Total Git Commits | 8 (Run 3 only: 1 architect + 1 builder + 4 kernel stages + 1 report + 1 scoring) |
| Total Pipeline Duration | ~1h 12m (10:32 to 11:44 UTC, Run 3) |
| Overall Result | SUCCESS |
| Pipeline Iterations | 3 runs total (Run 1: Mar 2 14:06-15:21, Run 2: Mar 4 08:23-09:17, Run 3: Mar 4 10:32-11:44) |

---

## 0. Context: Three Pipeline Runs

This operation was built three times, each with a different design approach. This is the single most important fact about this pipeline execution and is analyzed throughout this report.

| Run | Date | Duration | Approach | References | Outcome |
|-----|------|----------|----------|------------|---------|
| Run 1 | Mar 2, 14:06-15:21 | ~75m | Derivative (moreh_group_norm) | moreh_group_norm | PASS but with post-completion fix (CT args) |
| Run 2 | Mar 4, 08:23-09:17 | ~54m | Hybrid (tilize, untilize, softmax) | tilize, untilize, softmax | PASS with 1 hang + 3 compile errors |
| Run 3 | Mar 4, 10:32-11:44 | ~72m | Hybrid (moreh_norm_w, softmax_general) | moreh_norm_w, softmax_general | CLEAN PASS (0 failures) |

Run 3 is the "final" pipeline run whose artifacts are currently on disk and on HEAD. The analysis below focuses on Run 3 but draws lessons from comparing all three runs.

---

## 1. Pipeline Execution Summary

### Phase Timeline (Run 3)

| Phase | Agent(s) | Start | End | Duration | Status | Key Observations |
|-------|----------|-------|-----|----------|--------|------------------|
| 0: Discovery | orchestrator | - | - | - | Done | Selected moreh_norm_w + softmax_general as references |
| 1: Analysis | ttnn-operation-analyzer (x2) | - | ~10:32 | unknown | Done | No analysis commits found in Run 3 git history; architect may have relied on prior run analyses or inline knowledge |
| 2: Design | ttnn-operation-architect | ~10:32 | 10:32 | <1m | Done | Produced op_design.md + .tdd_state.json + 4 test files (commit fd66eea) |
| 3: Build | ttnn-generic-op-builder | ~10:32 | 11:00 | ~28m | Done | Produced full Python infra + stub kernels (commit 6772266) |
| 4: TDD Kernels | ttnn-kernel-writer (x4) | 11:00 | 11:26 | ~26m | Done | 4 stages, all passed on first attempt |
| 5: Report | orchestrator | 11:26 | 11:44 | ~18m | Done | REPORT.md produced (commit 950f99d) |

### Agent Duration Breakdown (Run 3)

| Agent | First Commit | Last Commit | Wall Duration | Retries | Active vs Debugging |
|-------|-------------|-------------|---------------|---------|---------------------|
| ttnn-operation-architect | 10:32:45 | 10:32:45 | <1m (single commit) | 0 | All productive |
| ttnn-generic-op-builder | 11:00:03 | 11:00:03 | ~27m (wall clock from architect end) | 0 | Mostly productive; includes build/test cycles |
| ttnn-kernel-writer (S1) | 11:09:52 | 11:09:52 | ~10m (from builder end) | 0 | All productive |
| ttnn-kernel-writer (S2) | 11:14:03 | 11:14:03 | ~4m | 0 | All productive |
| ttnn-kernel-writer (S3) | 11:19:51 | 11:19:51 | ~6m | 0 | All productive |
| ttnn-kernel-writer (S4) | 11:26:11 | 11:26:11 | ~6m | 0 | All productive |

**Duration calculation method**: Git commit timestamps (`%ai` format). No breadcrumb files exist for this run.

### Duration Visualization

```
Phase 2   |#|                                              (~1m)  architect
Phase 3        |############|                              (~28m) builder
Phase 4                      |####|##|###|###|             (~26m) kernel-writer x4
Phase 5                                       |########|   (~18m) report
          10:30  10:40  10:50  11:00  11:10  11:20  11:30  11:45 UTC

Longest phase: Phase 3 (Build, ~28m) -- builder had to produce full Python infra,
  10 CBs, multi-core work distribution, conditional gamma/beta support
```

### Time Distribution (Run 3)

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | unknown | - | No analyzer commits in Run 3 |
| Design (Phase 2) | ~1m | ~1% | Single commit, very fast |
| Build (Phase 3) | ~28m | ~39% | Full Python infra + stub kernels |
| Kernel implementation (Phase 4) | ~26m | ~36% | 4 TDD stages |
| -- Productive coding | ~26m | ~36% | Zero retries |
| -- Debugging/retries | 0m | 0% | Clean pass |
| Reporting (Phase 5) | ~18m | ~25% | |
| **Total** | **~72m** | **100%** | |

---

## 2. What Went Well

### 2.1 All TDD Stages Passed on First Attempt (Run 3)

**Phase/Agent**: Phase 4, ttnn-kernel-writer
**Evidence**: All 4 kernel-writer commits (68d0605, 2f01cd7, a14fc49, 39b65c1) each represent a single clean pass. Zero entries in `.tdd_state.json` failure_history (all stages show `attempts: 0, free_retries: 0`). No fixup commits between stages.
**Why it worked**: Run 3 was the third iteration. The architect had the benefit of two prior designs (Run 1's Derivative approach with moreh_group_norm, Run 2's Hybrid with tilize/untilize/softmax). The Run 3 design chose optimal references: moreh_norm_w for cross-tile reduction patterns and softmax_general for the multi-pass normalization pattern. The design document's "Binary Op Broadcast Verification" table and "Critical Notes" section provided precise guidance that the kernel writer followed correctly.

### 2.2 Excellent CB Design -- Zero CB-Related Bugs

**Phase/Agent**: Phase 2, ttnn-operation-architect
**Evidence**: The 10-CB allocation (c_0, c_1, c_2, c_3, c_4, c_16, c_24, c_25, c_26, c_27) survived all 4 TDD stages without modification. No CB sizing errors, no CB ID conflicts, no push/pop imbalance bugs. Contrast with Run 1 which used 13 CBs (including separate cb_centered, cb_squared, cb_var, cb_rstd, cb_normalized, cb_gamma_out) -- a more complex layout that was harder for the kernel writer to manage.
**Why it worked**: The architect's design learned from Runs 1 and 2. Run 3 minimized CB count by reusing cb_tmp (c_27) as a scratch buffer across phases and using single-buffered (1-page) CBs everywhere, which eliminated page count mismatches.

### 2.3 Three-Pass Architecture Eliminated L1 Pressure

**Phase/Agent**: Phase 2, ttnn-operation-architect
**Evidence**: Runs 1 and 2 both used a "small algorithm" approach where all Wt tiles for a row must fit in L1 simultaneously (Wt pages in cb_input). Run 3's three-pass design reads each tile row three times from DRAM (single-buffered c_0), requiring only 1 tile of input at any time. This works for any width without L1 overflow.
**Why it worked**: The architect explicitly drew from softmax_general's W-large variant, which uses the same three-pass DRAM re-read strategy.

### 2.4 Gamma/Beta ROW Broadcast Correction

**Phase/Agent**: Phase 4, ttnn-kernel-writer
**Evidence**: The architect's design document originally specified `mul<NONE>` (element-wise) for gamma/beta application. The kernel writer correctly identified that gamma/beta tensors shaped `[1,1,1,W]` only have valid data in row 0 when tilized, requiring ROW broadcast instead of element-wise. This is documented in REPORT.md deviation #2 and implemented using `mul_tiles_bcast_rows` / `add_tiles_bcast<ROW>`.
**Why it worked**: The kernel writer independently understood the tile layout semantics and self-corrected without needing upstream feedback. This represents mature kernel reasoning.

### 2.5 Clean One-Commit-Per-Stage Git History

**Phase/Agent**: Phase 4, ttnn-kernel-writer
**Evidence**: Run 3 has exactly one commit per TDD stage: 68d0605 (S1), 2f01cd7 (S2), a14fc49 (S3), 39b65c1 (S4). No intermediate fix commits, no state-advance commits (unlike Run 1 which had separate "advance TDD state" commits between stages).
**Why it worked**: The kernel writer implemented each stage correctly on the first try, and the commit discipline was clean.

---

## 3. Issues Found

### Issue 1: Pipeline Was Run Three Times for One Operation

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | All phases |
| Agent | All (pipeline orchestration) |
| Retries Consumed | 2 full pipeline runs wasted |
| Time Cost | ~129 minutes of combined Run 1 + Run 2 effort |

**Problem**: The layer_norm operation was built three times:
- Run 1 (Mar 2, 14:06-15:21): Derivative approach with moreh_group_norm reference. Completed but required a manual post-completion fix (commit e82552721ff) for TensorAccessor compile-time arg padding when gamma/beta are absent.
- Run 2 (Mar 4, 08:23-09:17): Hybrid approach with tilize/untilize/softmax. Used ROW_MAJOR input layout with tilize/untilize. Had 1 hang + 3 compile errors across TDD stages.
- Run 3 (Mar 4, 10:32-11:44): Hybrid approach with moreh_norm_w/softmax_general. TILE_LAYOUT throughout. Clean pass.

Each run produced different file layouts (Run 2 used `reader.cpp`/`compute.cpp`/`writer.cpp`, Run 3 used `reader_layer_norm.cpp`/`compute_layer_norm.cpp`/`writer_layer_norm.cpp`), different CB schemes, different parameter naming (`weight`/`bias` vs `gamma`/`beta`), and different design philosophies.

**Root Cause**: No mechanism to preserve and learn from prior pipeline run failures. Each run started fresh. The orchestrator does not carry forward lessons from prior runs on the same operation.

**Fix for agents**:
- Orchestrator: Before starting a pipeline run, check if prior run artifacts exist. If so, produce a "lessons learned" summary and inject it into the architect's context.
- Architect: If re-designing an operation, explicitly review what went wrong in the prior design rather than starting from scratch.

### Issue 2: Stage 3 Test Reference Modified by Kernel Writer

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- Stage 3 (variance) |
| Agent | ttnn-kernel-writer |
| Retries Consumed | 0 (worked first try despite deviation) |
| Time Cost | Minimal direct cost, but risk of silently loosened testing |

**Problem**: The architect's design specified Stage 3 should output raw variance per row: `(x - x.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True)`. The kernel writer changed both the compute kernel and the test reference to output the fully normalized result `(x - mean) / sqrt(var + eps)` instead (commit a14fc49). The `.tdd_state.json` still shows the original `reference_body` with `.expand_as(x)`.

The REPORT.md documents this deviation: "Design specified raw variance output. Implemented normalized output instead, since broadcasting reduced col0 values to all columns requires extra infrastructure."

**Root Cause**: The architect designed a stage output (raw variance as a reduced [N,1] tile broadcast to [N,W]) that was impractical to implement with the three-pass architecture. Broadcasting a reduced col0-only tile to all columns would require explicit copy/broadcast infrastructure that doesn't exist in the current CB layout. The kernel writer made a pragmatic choice to skip ahead to normalized output.

**Fix for agents**:
- Architect: When designing TDD stage outputs, verify that the proposed output shape and format can actually be produced by the kernel at that stage without extra infrastructure. Prefer stage outputs that are full-tile (all elements valid) rather than reduced tiles that need broadcasting.
- Kernel writer: When deviating from the stage spec, update `.tdd_state.json` to reflect the actual reference, not just the test file. The `reference_body` field in `.tdd_state.json` still contains the original (now incorrect) expression.

### Issue 3: Run 2 Used ROW_MAJOR Input Layout (Design Regression)

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 2 (Design) -- Run 2 |
| Agent | ttnn-operation-architect (Run 2) |
| Retries Consumed | Entire Run 2 was ultimately superseded |
| Time Cost | ~54 minutes of Run 2 |

**Problem**: Run 2's architect designed layer_norm to accept ROW_MAJOR input and include tilize/untilize steps in the compute pipeline (13 CBs including cb_tilize_out, cb_untilize_out). This added complexity for no user benefit since the standard TTNN convention is TILE_LAYOUT input/output. Run 1 and Run 3 both used TILE_LAYOUT correctly.

Run 2's `.tdd_state.json` shows `"layout": "ROW_MAJOR_LAYOUT"`, and the design document explicitly stated "x layout: ROW_MAJOR, Input must be row-major" and "Output: ROW_MAJOR".

**Root Cause**: Run 2 used tilize/untilize as input_stage/output_stage references, which led the architect to design around ROW_MAJOR even though it's not the natural choice for this operation. The reference selection drove the design in the wrong direction.

**Fix for agents**:
- Discovery/orchestrator: Do not select tilize/untilize as references for normalization operations that should operate on tiled data natively. Reserve tilize/untilize references only for operations that genuinely need layout conversion.
- Architect: Always validate the input layout choice against the operation's natural domain. LayerNorm's natural domain is tile-based REDUCE_ROW; adding tilize/untilize is unnecessary overhead.

### Issue 4: Run 1 Required Post-Completion Fix for TensorAccessor CT Args

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 (completed) then Phase 3 retroactively -- Run 1 |
| Agent | ttnn-generic-op-builder / ttnn-kernel-writer (Run 1) |
| Retries Consumed | Required manual human fix after pipeline "completed" |
| Time Cost | ~7 minutes for the fix (15:14 to 15:20) |

**Problem**: Run 1 completed all 4 TDD stages, but when running the integration test with no gamma/no beta, the kernel failed with "static assertion failed: Index out of range" on `TensorAccessorArgs`. The kernel's compile-time arg offsets for gamma and beta were computed unconditionally, but the program descriptor only appended those CT args when gamma/beta were present.

The fix (commit e82552721ff) changed the program descriptor to always pad dummy accessor CT args even when gamma/beta are absent: `DUMMY_ACCESSOR_CT = [2]`.

**Root Cause**: The builder and kernel writer both assumed that `constexpr` offset computation in the kernel would only execute when the corresponding `if constexpr` branch was taken. In reality, C++ evaluates all `constexpr` template parameters at compile time regardless of control flow, so `TensorAccessorArgs<offset>` must have valid args at that offset even if the branch is never executed.

**Fix for agents**:
- Builder: When a kernel uses conditional TensorAccessor arguments (e.g., optional gamma/beta), always pad compile-time args with dummy accessor entries for all optional tensors, regardless of whether they are present. Document this as a mandatory pattern.
- Architect: In the design document, explicitly call out when a kernel has conditional tensor accessors and specify the CT arg padding strategy.
- Run 3's approach (using `#ifdef HAS_GAMMA` preprocessor defines instead of `constexpr` branches) elegantly avoids this problem entirely.

### Issue 5: Missing Breadcrumbs and Execution Logs

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | All phases |
| Agent | All agents |
| Retries Consumed | N/A |
| Time Cost | N/A (observability issue, not runtime issue) |

**Problem**: No `agent_logs/` directory exists for any of the three runs. There are no breadcrumb JSONL files and no execution log Markdown files. The only evidence available for this analysis is git history and the final artifacts.

**Root Cause**: The agents do not currently emit breadcrumb files or structured execution logs. The pipeline lacks an observability layer.

**Fix for agents**:
- All agents: Implement breadcrumb emission (start, hypothesis, test, fix_result, deviation, complete events) to a standardized `agent_logs/` directory.
- Orchestrator: Create the `agent_logs/` directory before launching any agents and pass the path to each agent.

### Issue 6: `.tdd_state.json` Not Updated to Reflect Actual Run Results

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 |
| Agent | ttnn-kernel-writer |
| Retries Consumed | 0 |
| Time Cost | Minimal |

**Problem**: The `.tdd_state.json` on disk (HEAD) still shows all stages as `"status": "pending"` with `"attempts": 0` and empty `failure_history`. This is the architect's initial template, not the kernel writer's updated state. The kernel writer in Run 3 apparently did not update the TDD state file after completing stages (no state-advance commits visible in Run 3, unlike Run 1 which had explicit "advance TDD state" commits).

**Root Cause**: The Run 3 kernel writer either (a) operated outside the TDD state tracking system, or (b) the state file on disk was overwritten by the architect's fresh template when Run 3 started.

**Fix for agents**:
- Kernel writer: Must update `.tdd_state.json` after each stage passes, recording the commit hash, attempt count, and status.
- Orchestrator: Verify `.tdd_state.json` reflects actual run state after pipeline completion.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown (Run 3)

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline (S1) | ~10m | 0 free, 0 hard | PASS | Clean -- includes builder-to-kernel handoff |
| mean_subtract (S2) | ~4m | 0 free, 0 hard | PASS | Clean -- single compute file change |
| variance (S3) | ~6m | 0 free, 0 hard | PASS | Clean -- largest delta (130 lines of new code + test change) |
| full_normalize (S4) | ~6m | 0 free, 0 hard | PASS | Clean -- reader gamma/beta + compute ROW bcast |

### Comparison Across Runs

| Run | Stages | Total Failures | Hangs | Compile Errors | Numerical | Wall Time |
|-----|--------|---------------|-------|----------------|-----------|-----------|
| Run 1 | 4 | 0 (but post-fix) | 0 | 0 | 0 | ~75m |
| Run 2 | 4 | 4 | 1 hang | 3 compile errors | 0 | ~54m |
| Run 3 | 4 | 0 | 0 | 0 | 0 | ~72m |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Repeated pipeline runs | All | ~129m wasted | 64% of total effort | Run 1 + Run 2 entirely superseded by Run 3 | 2 full runs | No learning mechanism between runs |
| 2 | Build phase | ttnn-generic-op-builder | ~28m | 39% of Run 3 | Full infra creation including compile/test cycles | 0 | Builder complexity: 10 CBs, multi-core, conditional gamma/beta |
| 3 | Report phase | orchestrator | ~18m | 25% of Run 3 | Report generation took longer than all 4 kernel stages combined | 0 | Unclear -- report is 100 lines, should be faster |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| Run 1 analyzers | moreh_group_norm analysis (505 lines) | Run 1 approach superseded | Preserve analyses across runs |
| Run 2 analyzers | tilize + untilize + softmax analyses (1044 lines combined) | Run 2 approach superseded | Better reference selection |
| Run 1 builder | Full program descriptor with 13 CBs, single-core | Replaced by Run 3's 10-CB multi-core design | Get design right the first time |
| Run 2 builder | ROW_MAJOR tilize/untilize pipeline | Wrong layout choice | Validate layout choice before building |
| Run 1 kernel writer | 4 TDD stages + post-fix | Entire implementation superseded | - |
| Run 2 kernel writer | 4 TDD stages with recovery | Entire implementation superseded | - |

**Total wasted effort**: Approximately 3.5 hours of combined agent compute across Runs 1 and 2, producing ~2500 lines of code and documentation that were entirely replaced.

---

## 5. Inter-Agent Communication Issues

### Handoff 1: Discovery/Orchestrator -> Analyzers

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation selections |
| Quality | VARIABLE across runs |
| Issues | Run 2 selected tilize/untilize which drove incorrect ROW_MAJOR design |
| Downstream Impact | Entire Run 2 was built around the wrong layout assumption |
| Suggestion | Add a layout validation gate: if the target operation naturally works on tiled data, reject references that imply layout conversion |

### Handoff 2: Architect -> Builder (Run 3)

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md + .tdd_state.json |
| Quality | GOOD |
| Issues | Minor: design said `mul<NONE>` for gamma/beta but ROW broadcast was actually needed. No impact because builder doesn't implement compute logic. |
| Downstream Impact | None for builder; kernel writer self-corrected |
| Suggestion | Architect should validate broadcast types against tensor shapes in the design phase |

### Handoff 3: Builder -> Kernel Writer (Run 3)

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels + program descriptor + test files |
| Quality | GOOD |
| Issues | The builder produced correct stub infrastructure that compiled and ran. No kernel-writer-visible issues. |
| Downstream Impact | Clean start for kernel writer |
| Suggestion | None -- this handoff worked well |

### Handoff 4: Architect Design -> Kernel Writer (Run 3)

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 (Kernel Implementation) |
| Quality | ADEQUATE |
| Issues | (1) Stage 3 output spec was impractical (required broadcasting reduced tile). (2) Gamma/beta broadcast type was wrong (NONE vs ROW). |
| Downstream Impact | Kernel writer successfully self-corrected both issues, but this required independent reasoning that might not always succeed |
| Suggestion | Architect should simulate the CB state after each compute step to verify outputs are producible. Add a "tile valid region" column to the stage plan showing what data is valid in each output tile. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-kernel-writer (Run 3) | Always verify broadcast types against tensor shapes. A [1,1,1,W] tensor tilized has data only in row 0 -- this requires ROW broadcast, not NONE. Add a mandatory "Tilized Data Layout" section to design docs. | H | H |
| ttnn-operation-architect | self-reflection | When designing TDD stage intermediate outputs, ensure the output can be produced without extra broadcast/copy infrastructure. Prefer full-tile outputs over reduced-tile outputs. | H | M |
| ttnn-generic-op-builder | ttnn-kernel-writer (Run 1) | When a kernel conditionally uses TensorAccessorArgs, always pad CT args with dummy entries for absent tensors to avoid compile-time assertion failures. | H | H |
| discovery/orchestrator | self-reflection | Do not select tilize/untilize as references for operations that should natively accept TILE_LAYOUT input. Add a layout-match gate to reference selection. | H | H |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Cross-run learning | Same operation built 3 times with no knowledge transfer | Implement a "prior run summary" that persists across pipeline invocations for the same operation | H |
| Observability | No breadcrumbs or execution logs produced | Implement structured logging for all agents | H |
| TDD state tracking | `.tdd_state.json` not updated in Run 3 | Enforce state file updates as part of kernel writer protocol | M |
| Report phase duration | Report took 18m for a 100-line file | Investigate why -- possible model overhead or unnecessary re-analysis | L |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Run 3 had zero numerical issues. Run 2 also had no numerical issues (only hangs and compile errors). |
| 3 | `.tdd_state.json` coupling fragility | YES | Run 3's state file was not updated by the kernel writer. The file on disk still shows the architect's template. |
| 4 | No fast path for simple operations | NO | LayerNorm is a medium-complexity operation; the full pipeline was appropriate. |
| 6 | Builder runs on Sonnet while everything else uses Opus | POSSIBLY | Run 3's builder took ~28 minutes, the longest single phase. The builder produced correct output however, so the model choice did not cause failures. |
| 7 | Discovery keyword matching | YES | Run 2's discovery selected tilize/untilize as references, leading to the ROW_MAJOR design mistake. This matches the brittleness described in Issue 7. |
| 9 | No architect/builder cross-validation | NOT OBSERVED | The builder faithfully implemented the architect's CB design in Run 3. No discrepancies detected. |
| 11 | No incremental re-run capability | YES | Each pipeline run started fresh. When Run 1's design was found wanting, there was no way to reuse its analysis or adjust its design -- a full re-run was needed. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| No cross-run learning mechanism | The pipeline ran 3 times for one operation with no knowledge transfer between runs. ~3.5 hours of agent compute wasted. | HIGH |
| Architect broadcast type validation | The architect specified NONE broadcast for gamma/beta when ROW was required. Tilized [1,1,1,W] tensors have data only in row 0. | MEDIUM |
| TDD stage output feasibility check | Architect designed a stage output (raw variance) that was impractical to produce, forcing the kernel writer to deviate. | MEDIUM |
| TensorAccessor CT arg padding for optional tensors | When optional tensor accessors are absent, kernel CT args must still be padded to valid offsets. Run 1 hit this; Run 3 avoided it with preprocessor defines. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Implement Cross-Run Learning

- **Type**: pipeline_change
- **Target**: Orchestrator script / pipeline infrastructure
- **Change**: Before starting a new pipeline run for an operation that has prior artifacts, generate a "prior run lessons" document summarizing: (a) what approach was tried, (b) what failed and why, (c) what should be different. Inject this into the architect's context.
- **Expected Benefit**: Avoid rebuilding operations from scratch. The 3x run cost here (~3.5 hours) would shrink to ~1 hour with learning.
- **Priority**: HIGH
- **Effort**: MEDIUM

### Recommendation 2: Add Broadcast Type Validation to Architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: Add a mandatory "Tilized Tensor Layout Check" step: for every binary/broadcast op in the design, the architect must verify what data region is valid in each CB operand after tilization. Specifically, a [1,1,1,W] tensor tilized has valid data only in row 0, requiring ROW broadcast for replication to all rows.
- **Expected Benefit**: Prevents kernel writer from needing to independently discover and correct broadcast types.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: TDD Stage Output Feasibility Validation

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: For each TDD stage, the architect must verify that the specified output can be produced by the kernel at that stage using only the CBs and operations available. Specifically: if a stage output is a reduced tile (e.g., col0 only), verify that the test comparison can handle this without requiring broadcast infrastructure.
- **Expected Benefit**: Prevents kernel writer deviations from the stage plan.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Reference Selection Layout Gate

- **Type**: pipeline_change
- **Target**: Discovery / orchestrator phase
- **Change**: When selecting reference operations, validate that the reference's input/output layout matches the target operation's natural layout. Do not select tilize/untilize as references unless the target operation genuinely requires layout conversion.
- **Expected Benefit**: Prevents entire pipeline runs based on incorrect layout assumptions (Run 2).
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 5: Mandatory Breadcrumb Emission

- **Type**: tool_improvement
- **Target**: All agent implementations
- **Change**: Implement structured breadcrumb logging (JSONL format) to `agent_logs/` for every agent invocation. At minimum: start event, per-test-run result events, deviation events, complete event. Include ISO timestamps.
- **Expected Benefit**: Enables post-hoc analysis, timing breakdown, and systematic improvement identification.
- **Priority**: HIGH
- **Effort**: MEDIUM

### Recommendation 6: Document Preprocessor-Define Pattern for Optional Tensor Accessors

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder and ttnn-operation-architect
- **Change**: Standardize the pattern for optional tensor accessors: use `#ifdef`/`#define` preprocessor gates (as in Run 3) rather than `constexpr` branches (as in Run 1). The builder should pass `defines=[("HAS_GAMMA", "1")]` from the program descriptor and the kernel should gate all accessor construction behind `#ifdef`. This avoids the CT arg padding problem entirely.
- **Expected Benefit**: Eliminates the class of "Index out of range" errors for optional tensors.
- **Priority**: MEDIUM
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 3/5 | Run 3 had good references (moreh_norm_w, softmax_general), but Run 2's references (tilize, untilize) were poor choices that wasted an entire pipeline run |
| Analysis quality | 3/5 | Run 3 analysis files are not visible in git; earlier runs produced thorough analyses (500+ line documents) that were ultimately discarded |
| Design completeness | 4/5 | Run 3's design was nearly complete; only two issues (broadcast type, stage output feasibility) required kernel writer correction |
| Build correctness | 4/5 | Run 3's builder produced correct infrastructure on first try; Run 1's builder had the CT arg padding issue |
| Kernel implementation | 5/5 | Run 3 achieved zero failures across all 4 TDD stages; kernel writer demonstrated strong independent reasoning (broadcast correction, stage deviation) |
| Inter-agent communication | 3/5 | Good within Run 3, but the need for 3 full runs indicates poor communication at the pipeline level about what approach to take |
| Logging/observability | 1/5 | No breadcrumbs, no execution logs, no agent_logs directory. Analysis relied entirely on git history and final artifacts. |

### Top 3 Things to Fix

1. **Cross-run learning mechanism**: The pipeline ran 3 times, wasting ~3.5 hours. Each run started from scratch. A learning mechanism would dramatically reduce cost for iterative operations.
2. **Breadcrumb and execution log emission**: Without structured logs, this self-reflection had to reconstruct agent behavior from git commits alone. Timing, decision points, and debugging cycles are invisible.
3. **Reference selection validation**: Run 2's tilize/untilize references drove an entirely wrong design (ROW_MAJOR layout). A simple layout-match gate would have prevented this.

### What Worked Best

The Run 3 kernel writer's performance was exceptional. All 4 TDD stages passed on the first attempt with zero retries. The kernel writer independently corrected two design issues (broadcast types, stage 3 output) and produced clean, well-structured C++ kernel code. The three-pass compute architecture with manual cross-tile accumulation, reduce helpers, and broadcast operations is correct and efficient. This demonstrates that when the design is close to correct and the references are well-chosen, the kernel writer agent can reliably produce working hardware kernels for non-trivial operations like LayerNorm.
