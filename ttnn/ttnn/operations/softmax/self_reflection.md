# Self-Reflection: softmax

## Metadata
| Field | Value |
|-------|-------|
| Operation | `softmax` |
| Operation Path | `ttnn/ttnn/operations/softmax` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x2), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd, orchestrator |
| Total Git Commits | 12 (within this run: 8757a610 through 2593b642) |
| Total Pipeline Duration | ~53 minutes (08:58 - 09:51 UTC) |
| Overall Result | SUCCESS -- All 6 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | Orchestrator | ~1 min | PASS | Identified moreh_sum_w and moreh_sum_h as references for W and H reduction patterns |
| 1: Analysis | ttnn-operation-analyzer (x2) | ~10m (parallel) | PASS | Two analyzers ran in parallel: moreh_sum_w (08:58:58-09:09:26) and moreh_sum_h (08:58:55-09:09:17). Produced 538-line and ~500-line analyses. |
| 2: Design | ttnn-operation-architect | ~4m (09:07-09:11) | PASS | Produced unified op_design.md with 6-CB layout and 6 TDD stages. Hybrid mode using reduce, binary_op, copy_tile helpers. |
| 3: Build | ttnn-generic-op-builder | ~9m (09:17-09:25) | PASS | Created 16 files. One compilation failure (bad tensor_accessor.hpp include) fixed quickly. 5/5 integration tests passed. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~20m (09:29-09:49) | PASS | 6 stages, 7 hard attempts total (budget 36). Only 1 numerical failure on stage 5. No device hangs. |
| 5: Report | Orchestrator | ~2m (09:49-09:51) | PASS | REPORT.md generated |
| **Total** | | **~53m** | **PASS** | Clean end-to-end run. Phase 4 consumed ~38% of total time. |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (moreh_sum_h) | 08:58:55 | 09:09:17 | 10m 22s | 0 | ~10m active (reading code, querying DeepWiki) |
| ttnn-operation-analyzer (moreh_sum_w) | 08:58:58 | 09:09:26 | 10m 28s | 0 | ~10m active (same pattern) |
| ttnn-operation-architect | 09:07:24 | 09:11:38 | 4m 14s | 0 | ~4m active (design authoring) |
| ttnn-generic-op-builder | 09:17:13 | 09:25:50 | 8m 37s | 1 | ~7m active, ~1.5m fixing tensor_accessor.hpp include |
| ttnn-kernel-writer-tdd | 09:29:31 | 09:49:20 | 19m 49s | 1 | ~17m productive, ~3m debugging dim=-2 REDUCE_ROW vs REDUCE_COL |

**Duration calculation method**: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps were used as primary source. Git commit timestamps were used as cross-validation. Both sources are consistent.

### Duration Visualization

```
Phase 0  |#|                                                   (~1m)
Phase 1  |####################|                                (~10m) 2 analyzers in parallel
Phase 2       |########|                                       (~4m)
Phase 3                  |#################|                    (~9m)
Phase 4                                     |####################################| (~20m)
Phase 5                                                                           |###| (~2m)
         0    5    10   15   20   25   30   35   40   45   50 min

Longest phase: Phase 4 (20m) -- 6 TDD stages with one numerical failure requiring hypothesis/fix cycle
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~11m | 21% | 2 analyzers in parallel, queried DeepWiki 4 times |
| Design (Phase 2) | ~4m | 8% | Clean single-pass design |
| Build (Phase 3) | ~9m | 17% | 1 compilation retry |
| Kernel implementation (Phase 4) | ~20m | 38% | 6 TDD stages |
| -- Productive coding | ~17m | 32% | Writing kernel code that passed |
| -- Debugging/retries | ~3m | 6% | dim=-2 REDUCE_ROW vs REDUCE_COL hypothesis cycle |
| Reporting (Phase 5) | ~2m | 4% | |
| Inter-phase gaps | ~7m | 13% | Time between agent completion and next agent start |
| **Total** | **~53m** | **100%** | |

---

## 2. What Went Well

### 1. Exceptional TDD Pass Rate (6/6 first-attempt or near-first-attempt)

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: Stages 0-3 and 5 all passed on the first attempt (1 hard attempt each). Stage 4 (softmax_h_stable) required 2 hard attempts. Total: 7/36 budget = 19% utilization. Zero free retries consumed. Zero device hangs.
**Why it worked**: The architect's design document was exceptionally detailed -- it specified exact helper function calls with template parameters, CB state tables after each phase, and explicit manual pop requirements. The kernel writer could almost transcribe the design directly into code.

### 2. Helper Library Usage Was Comprehensive and Correct

**Phase/Agent**: Phase 2 (Design) + Phase 4 (Implementation)
**Evidence**: All 4 compute phases use helpers: `reduce<>()` for MAX and SUM, `sub<>()` and `mul<>()` for broadcast operations, `copy_tiles<>()` for passthrough/exp. `prepare_reduce_scaler<>()` used in reader. No raw tile register management in any kernel. All CB synchronization policies were correct on first implementation.
**Why it worked**: The architect mapped every compute phase to a specific helper call with precise template parameters. The kernel writer followed the mappings exactly.

### 3. Forward-Looking Implementation in Stage 3

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: The kernel writer implemented both the stable and unstable softmax paths during Stage 3 (softmax_w_stable), even though Stage 4 (softmax_w_unstable) was a separate test. Stage 4 required zero code changes and passed immediately. This also applied to Stage 6 (softmax_h_unstable) which required zero code changes after Stage 5.
**Why it worked**: Good engineering judgment from the kernel writer. The `if constexpr (numeric_stable)` branch was natural to implement alongside the stable path.

### 4. CB Layout Was Correct From the Start

**Phase/Agent**: Phase 2 (Design) + Phase 3 (Build)
**Evidence**: 6 CBs were designed, built, and used without any CB-related bugs across all 6 TDD stages. CB page counts (R for input/exp, 2 for output, 1 for scaler/max/recip_sum) were all correct. No CB sizing or synchronization issues.
**Why it worked**: The architect's CB requirements table with producer/consumer/pages/lifetime annotations gave the builder and kernel writer unambiguous guidance.

### 5. Clean Writer Double-Buffer Handling

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd (Stage 0)
**Evidence**: The writer was initially implemented with `cb_wait_front(cb_out, R)` (bulk wait), which would deadlock when R > 2 because cb_out only has 2 pages. The kernel writer caught this proactively (breadcrumb at 09:31:25) and fixed to per-tile wait/pop before even running the test. Stage 0 passed on first attempt.
**Why it worked**: The kernel writer performed CB sync checking before test runs (documented in `cb_sync_check` breadcrumbs), catching the mismatch between compute's PerTile push and a hypothetical bulk wait.

---

## 3. Issues Found

### Issue 1: Missing is_dim_h Compile-Time Arg for Compute Kernel

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- softmax_h_stable (Stage 5) |
| Agent | ttnn-operation-architect (root cause) / ttnn-kernel-writer-tdd (discovered) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~3 minutes (09:40:51 test fail to 09:43:50 fix confirmed) |

**Problem**: The original design document specified "Both dim=-1 and dim=-2 use REDUCE_ROW in the compute kernel" and "Unified Compute Design" stating the difference is "only in dataflow." This was incorrect. When dim=-2, the reduce operation must use REDUCE_COL (not REDUCE_ROW) and broadcast must use ROW (not COL). The kernel writer discovered this when the non-square shape `(1,1,32,256)` failed numerically -- the 32x32 single-tile case passed accidentally because REDUCE_ROW and REDUCE_COL produce the same result on a square tile.

The kernel writer had to form hypothesis H2 (HIGH confidence) and add `is_dim_h` as compile-time arg index 9 to the compute kernel, then parametrize all dimension-dependent operations (reduce_dim, bcast_dim, reduce_block, binary_block).

**Root Cause**: The architect's "Unified Compute Design" section (op_design.md lines 59-68) incorrectly stated that both dimensions use REDUCE_ROW. This is a fundamental mathematical error: reducing along dim=-2 (height) requires REDUCE_COL within tiles, not REDUCE_ROW. The architect appears to have conflated the dataflow-level "virtual row" abstraction with the tile-level reduction direction.

**Fix for agents**:
- **ttnn-operation-architect**: Add a mandatory validation step: "For each reduce dimension, verify the tile-level reduce direction. REDUCE_ROW reduces across columns within a tile (W dimension). REDUCE_COL reduces across rows within a tile (H dimension). If the softmax dimension is H, the tile-level reduce must be REDUCE_COL, not REDUCE_ROW."
- **ttnn-operation-architect**: The design document should include `is_dim_h` as a compute compile-time arg from the start, not leave it only in reader/writer args.

### Issue 2: tensor_accessor.hpp Include Path Does Not Exist

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry (compilation error, easy fix) |
| Time Cost | ~1.5 minutes |

**Problem**: The builder's kernel stubs included `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` which does not exist. The `TensorAccessorArgs` type is provided through `api/dataflow/dataflow_api.h`. Kernel compilation failed with `fatal error: tensor_accessor.hpp: No such file or directory`.

**Root Cause**: The builder agent's system prompt or helper-to-include mapping table contains an incorrect entry for TensorAccessor. The builder explicitly noted this in its execution log Section 7 as Recommendation 1.

**Fix for agents**:
- **ttnn-generic-op-builder**: Remove `tensor_accessor.hpp` from the include mapping table. TensorAccessorArgs is available via `api/dataflow/dataflow_api.h`.
- **Pipeline infrastructure**: Add a static include path validation that checks whether referenced headers exist before writing them to stubs.

### Issue 3: .tdd_state.json Inconsistency Between Committed and On-Disk Versions

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (worked around) |
| Time Cost | Minimal |

**Problem**: The builder's execution log notes: "The `.tdd_state.json` appeared to have only 1 stage when first read (the file on disk had been modified from the committed version with 6 stages). After `git checkout`, the committed 6-stage version was restored." This suggests a race condition or file system inconsistency between the architect's write and the builder's read.

**Root Cause**: The architect wrote `.tdd_state.json` directly instead of using the orchestrator's `add-stage` command (noted in REPORT.md Pain Points). This creates fragile file-based IPC between agents.

**Fix for agents**:
- **ttnn-operation-architect**: Use the orchestrator CLI to register stages rather than writing JSON directly.
- **Pipeline infrastructure**: Implement schema validation on `.tdd_state.json` read, as already proposed in pipeline-improvements.md Issue 3.

### Issue 4: Architect Started Before Analyzers Completed (Overlap)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 1 / Phase 2 overlap |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 |
| Time Cost | None directly, but may have caused the REDUCE_ROW error |

**Problem**: The architect started at 09:07:24, but the analyzers did not complete until 09:09:17 (moreh_sum_h) and 09:09:26 (moreh_sum_w). The architect began working ~2 minutes before the analyzer results were committed. The architect breadcrumbs show `"input_files":["N/A"]` for predecessor, suggesting it may not have read the analysis files.

**Root Cause**: The orchestrator may have launched the architect too early, or the architect began before the analysis files were committed to git.

**Fix for agents**:
- **Orchestrator**: Ensure Phase 2 does not start until all Phase 1 analyzer commits are confirmed.
- **ttnn-operation-architect**: Validate that expected analysis files exist before beginning design.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| passthrough | ~3m (09:29-09:32) | 0 free, 1 hard | PASS | Clean -- writer double-buffer fix was proactive |
| exp_only | ~1.5m (09:33-09:34) | 0 free, 1 hard | PASS | Clean -- single line change |
| softmax_w_stable | ~3m (09:34-09:36) | 0 free, 1 hard | PASS | Clean -- full 4-phase implementation |
| softmax_w_unstable | ~1m (09:37-09:38) | 0 free, 1 hard | PASS | No changes needed (already implemented) |
| softmax_h_stable | ~11m (09:38-09:48) | 0 free, 2 hard | PASS | Numerical failure on non-square shape; REDUCE_ROW vs REDUCE_COL |
| softmax_h_unstable | ~1m (09:48-09:49) | 0 free, 1 hard | PASS | No changes needed |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | softmax_h_stable | ttnn-kernel-writer-tdd | ~11m | 21% | Numerical mismatch on non-square shape required hypothesis cycle and upstream fix to program_descriptor.py | 1 retry | Architect's design incorrectly stated both dims use REDUCE_ROW |
| 2 | Inter-phase gaps | orchestrator | ~7m | 13% | Dead time between agent completion and next agent start | N/A | Agent orchestration overhead |
| 3 | Analysis | ttnn-operation-analyzer | ~10m | 19% | Two analyzers ran in parallel but took 10+ minutes each including DeepWiki queries | 0 | DeepWiki query latency (~1m per query, 4 queries) |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-generic-op-builder | Generated kernel stubs with tensor_accessor.hpp include | Header does not exist; had to be removed | Fix the include mapping table in builder instructions |
| ttnn-kernel-writer-tdd | First attempt at softmax_h_stable (09:38-09:40) ran with REDUCE_ROW | Wrong reduce dimension for dim=-2; test run wasted | Architect should have specified REDUCE_COL for dim=-2 from the start |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: Orchestrator -> ttnn-operation-analyzer

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation paths (moreh_sum_w, moreh_sum_h) |
| Quality | GOOD |
| Issues | None -- correct references selected for W and H reduction patterns |
| Downstream Impact | Analyzers produced useful detailed analyses |
| Suggestion | None needed |

### Handoff 2: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | moreh_sum_w_analysis.md, moreh_sum_h_analysis.md |
| Quality | ADEQUATE (but possibly not read) |
| Issues | Architect started 2 minutes before analyzer commits completed. Architect breadcrumbs show `"input_files":["N/A"]`. The moreh_sum_h analysis likely describes REDUCE_COL for H-dimension reduction, which the architect then incorrectly simplified to "both use REDUCE_ROW". |
| Downstream Impact | The architect may have missed the H-dimension reduction details, leading to the REDUCE_ROW error that cost 1 retry and ~3m debugging in Phase 4. |
| Suggestion | Orchestrator must gate Phase 2 on Phase 1 completion. Architect should explicitly confirm it has read and understood both analysis documents. |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | GOOD |
| Issues | Minor: .tdd_state.json had a transient inconsistency (1 stage vs 6 stages). Builder worked around it via git checkout. |
| Downstream Impact | Minimal -- builder recovered quickly. |
| Suggestion | Use orchestrator CLI for stage registration. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program_descriptor.py, test files, execution log with handoff notes |
| Quality | GOOD |
| Issues | The builder's handoff notes (execution log Section 6) were excellent: listed all CT/RT arg indices, CB configurations, include path warnings, and known limitations. The kernel writer acknowledged reading the design. |
| Downstream Impact | Positive -- kernel writer had clear guidance. The only issue (missing is_dim_h for compute) was an architect error, not a builder error. |
| Suggestion | Builder could additionally validate that all architect-specified CT args are present in program_descriptor.py |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder (instructions) | ttnn-generic-op-builder | Remove tensor_accessor.hpp from include mapping table | HIGH | HIGH |
| ttnn-operation-architect (instructions) | ttnn-kernel-writer-tdd | Architect must specify tile-level reduce direction for EACH supported dim, not assume unified REDUCE_ROW | HIGH | HIGH |
| ttnn-operation-architect (instructions) | ttnn-generic-op-builder | Architect should register stages via orchestrator CLI, not write .tdd_state.json directly | MEDIUM | MEDIUM |
| orchestrator | self-reflection | Gate Phase 2 start on Phase 1 completion (all analyzer commits confirmed) | MEDIUM | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Design validation | Architect stated "both dims use REDUCE_ROW" which was wrong for dim=-2 | Add a mandatory reduce-dimension checklist: for each dim, verify tile-level reduce direction vs data-level meaning | HIGH |
| Build validation | Builder included nonexistent header | Add static validation that all kernel includes resolve to real files before compilation | MEDIUM |
| Phase gating | Architect may have started before analyzers finished | Orchestrator should not launch Phase 2 until all Phase 1 agents have committed | MEDIUM |
| Inter-phase gaps | ~7 minutes (13% of total) spent in gaps between phases | Investigate reducing orchestration overhead between agent launches | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Only 1 numerical failure, resolved in ~3 minutes. This run was very clean. |
| 2 | Too many planning stages (DONE -- merged) | N/A | Architecture already uses merged Architect agent. |
| 3 | .tdd_state.json coupling is fragile | YES | Builder observed stale .tdd_state.json and had to git checkout to recover. |
| 4 | No fast path for simple operations | NO | Softmax is a medium-complexity op (6 CBs, 4 compute phases), appropriate for full pipeline. |
| 6 | Builder runs on Sonnet | POSSIBLY | Builder had 1 compilation error (bad include). Could be model-related or instruction-related. |
| 7 | Discovery keyword matching | NO | Discovery correctly identified moreh_sum_w and moreh_sum_h. |
| 9 | No architect/builder cross-validation | PARTIALLY | The missing is_dim_h for compute wasn't caught until runtime. A static cross-check between design CT args and program_descriptor.py could have caught this. |
| 11 | No incremental re-run capability | NO | Pipeline completed successfully without needing re-run. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect may conflate data-level and tile-level reduce dimensions | The architect stated "both dims use REDUCE_ROW" because the virtual-row abstraction makes both look like row reductions at the dataflow level. But tile-level reduce must match the actual dimension being reduced (REDUCE_COL for H). This is a subtle conceptual error that square-tile tests mask. | HIGH |
| Phase 1/2 overlap: architect may start before analyzers commit | The architect breadcrumbs show start at 09:07:24 with `input_files: ["N/A"]`, while analyzers completed at 09:09:17-09:09:26. The architect may not have read the analysis docs. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Architect Must Specify Per-Dimension Tile-Level Reduce Direction

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add mandatory validation rule: "When designing operations that support multiple reduce dimensions, you MUST specify the tile-level reduce direction for EACH dimension separately. REDUCE_ROW reduces along the W axis within tiles. REDUCE_COL reduces along the H axis within tiles. These are NOT interchangeable. Include a validation table mapping each logical dim to its tile-level ReduceDim."
- **Expected Benefit**: Prevents the REDUCE_ROW vs REDUCE_COL confusion that caused 1 retry and ~3m debugging
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Remove tensor_accessor.hpp from Builder Include Mapping

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent prompt (include mapping table)
- **Change**: Remove the entry `| TensorAccessor | #include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp" |`. Add note: "TensorAccessorArgs is available via `api/dataflow/dataflow_api.h` -- no separate include needed."
- **Expected Benefit**: Eliminates a guaranteed compilation failure on every operation that uses TensorAccessor
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Gate Phase 2 on Phase 1 Completion

- **Type**: pipeline_change
- **Target**: Orchestrator script
- **Change**: Add explicit wait for all Phase 1 analyzer git commits before launching Phase 2 architect. Verify analyzer output files exist and are non-empty.
- **Expected Benefit**: Ensures architect has access to analysis results, potentially preventing design errors
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Static Cross-Check Between Design CT Args and Program Descriptor

- **Type**: new_validation
- **Target**: Pipeline (between Phase 3 and Phase 4)
- **Change**: After builder completes, automatically compare the CT arg lists in op_design.md against the CT arg lists in program_descriptor.py. Flag any args present in design but missing from descriptor (or vice versa).
- **Expected Benefit**: Would have caught the missing `is_dim_h` CT arg for compute before any tests ran
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 5: Reduce Inter-Phase Gap Time

- **Type**: pipeline_change
- **Target**: Orchestrator script
- **Change**: Profile the orchestrator to identify where the ~7 minutes of inter-phase gap time is spent. Possible causes: agent startup overhead, git operations, environment setup. Reduce where possible.
- **Expected Benefit**: Could cut total pipeline time by ~13%
- **Priority**: LOW
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4 | Correctly identified moreh_sum_w and moreh_sum_h as references for both reduction dimensions |
| Analysis quality | 4 | Detailed analyses with CB layouts, work unit definitions, and compute patterns. Unclear if architect actually used them. |
| Design completeness | 3 | Excellent CB layout and helper mappings, but REDUCE_ROW vs REDUCE_COL error for dim=-2 was a significant miss. CB state tables after each phase were outstanding. |
| Build correctness | 4 | All files created correctly. One bad include (systemic issue, not op-specific). Handoff notes were excellent. |
| Kernel implementation | 5 | 6/6 stages passed with only 7/36 hard attempts. Proactive CB sync checking. Forward-looking implementation (stable+unstable in one stage). |
| Inter-agent communication | 3 | Builder-to-kernel-writer handoff was excellent. Analyzer-to-architect handoff may have been missed due to timing overlap. |
| Logging/observability | 4 | Breadcrumbs present for all agents with timestamps. Execution log for builder is detailed. Kernel writer breadcrumbs include hypothesis/fix events. Missing: kernel writer execution log (only breadcrumbs available). |
| Helper usage compliance | 5 | Every compute phase uses the appropriate helper. Reader uses prepare_reduce_scaler. No raw tile register management. No missed helpers. |

### Top 3 Things to Fix

1. **Architect must validate tile-level reduce direction per dimension** -- The "unified REDUCE_ROW" assumption was mathematically wrong for dim=-2 and only passed square-tile tests accidentally. This is the kind of error that wastes time on every multi-dimension reduction operation.

2. **Remove tensor_accessor.hpp from builder's include mapping** -- This is a systemic issue that will cause a guaranteed compilation failure on every operation using TensorAccessor until fixed.

3. **Gate Phase 2 on Phase 1 completion** -- The architect may have started without reading analyzer results, potentially contributing to the REDUCE_ROW error. This is a pipeline sequencing bug.

### What Worked Best

The kernel writer's systematic approach was the strongest aspect of this run. The CB sync checking before each test run (documented in 6 `cb_sync_check` breadcrumbs), the proactive double-buffer fix in Stage 0, and the forward-looking implementation of both stable/unstable paths in Stage 3 collectively resulted in a remarkably efficient TDD phase: 6 stages completed in ~20 minutes with only 1 numerical failure. The 19% budget utilization (7/36 hard attempts) is well below the maximum, leaving substantial margin. This efficiency was enabled by the architect's detailed design (CB state tables, explicit helper call signatures) and the builder's thorough handoff notes.

---

## 10. Helper Usage Audit

### Available Helpers

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `reduce_helpers_compute.hpp` | `reduce<PoolType, ReduceDim, InputPolicy, ReconfigMode>()` | YES -- used for MAX and SUM reductions |
| `reduce_helpers_dataflow.hpp` | `prepare_reduce_scaler<cb_id>()`, `calculate_and_prepare_reduce_scaler<>()` | YES -- used in reader for scaler tile |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()`, `square<>()` with broadcast | YES -- used for sub(input, max) and mul(exp, recip_sum) |
| `copy_tile_helpers.hpp` | `copy_tiles<InputPolicy, ReconfigMode>()` | YES -- used for passthrough and exp-only stages |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT` constant | YES -- used internally by reduce/binary helpers |
| `cb_helpers.hpp` | CB utility functions | NO -- not needed (helpers manage CBs) |
| `tilize_helpers.hpp` | `tilize<>()` | NO -- input is already TILE_LAYOUT |
| `untilize_helpers.hpp` | `untilize<>()` | NO -- output stays in TILE_LAYOUT |
| `l1_helpers.hpp` | L1 address helpers | YES -- used internally by reduce_helpers_dataflow |
| `common_types.hpp` | Shared type definitions | YES -- used internally by all helpers |

### Per-Phase Helper Compliance

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| reader | tile loading | Raw (TensorAccessor + noc_async_read) | Raw (TensorAccessor + noc_async_read) | Raw Justified | No helper exists for general tile loading with strided access patterns |
| reader | scaler prep | `prepare_reduce_scaler<cb_scaler>(1.0f)` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f)` | Correct | Helper used exactly as designed |
| compute | Phase 1 (MAX) | `reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>` | `reduce<MAX, reduce_dim, WaitUpfrontNoPop>` | Correct | Improved over design by parametrizing reduce_dim |
| compute | Phase 2 (SUB+EXP) | `sub<COL, NoWaitNoPop, WaitAndPopPerTile>` + exp_post_op | `sub<bcast_dim, NoWaitNoPop, WaitAndPopPerTile>` + exp_post_op | Correct | Improved over design by parametrizing bcast_dim |
| compute | Phase 3 (SUM+RECIP) | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` + recip_post_op | `reduce<SUM, reduce_dim, WaitUpfrontNoPop>` + recip_post_op | Correct | Improved over design by parametrizing reduce_dim |
| compute | Phase 4 (MUL) | `mul<COL, NoWaitNoPop, WaitAndPopPerTile>` | `mul<bcast_dim, NoWaitNoPop, WaitAndPopPerTile>` | Correct | Improved over design by parametrizing bcast_dim |
| compute | Unstable Phase 1 | `copy_tiles<WaitAndPop>` + exp_post_op | `copy_tiles<WaitAndPop, NONE>` + exp_post_op | Correct | Helper used as designed |
| writer | tile writing | Raw (TensorAccessor + noc_async_write) | Raw (TensorAccessor + noc_async_write) | Raw Justified | No helper exists for general tile writing with strided access patterns |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | 8 (reader: 2, compute: 5 stable + 1 unstable-only, writer: 1) |
| Phases using helpers correctly | 6 |
| Phases with justified raw code | 2 (reader tile loading, writer tile writing) |
| Phases with missed helpers | 0 |
| Phases with misused helpers | 0 |
| **Helper compliance rate** | **100%** |

### Redundant CB Operations Around Helpers

No redundant CB operations detected around helper calls.

The kernel correctly uses manual `cb_pop_front(cb_input, R)` after `sub<NoWaitNoPop>` and `cb_pop_front(cb_exp, R)` after `mul<NoWaitNoPop>`. These are required (not redundant) because the `NoWaitNoPop` policy explicitly delegates pop responsibility to the caller. The `cb_wait_front(cb_scaler, 1)` before the main loop is also not redundant -- it ensures the scaler tile is available before the first reduce call, and the scaler CB is persistent (never popped).

### Missed Helper Opportunities

All available helpers were used correctly. No missed opportunities.
