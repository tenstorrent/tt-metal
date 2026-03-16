# Self-Reflection: softmax

## Metadata
| Field | Value |
|-------|-------|
| Operation | `softmax` |
| Operation Path | `ttnn/ttnn/operations/softmax` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x2), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd, orchestrator |
| Total Git Commits | 12 (softmax-specific, this run: 8757a610 through 2593b642) |
| Total Pipeline Duration | ~53 minutes (08:58 - 09:51 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~3m | COMPLETE | Identified moreh_sum_w and moreh_sum_h as references for W and H reduction patterns |
| 1: Analysis | ttnn-operation-analyzer (x2) | ~10m (parallel) | COMPLETE | Two analyzers ran concurrently: moreh_sum_w (08:58:58-09:09:26) and moreh_sum_h (08:58:55-09:05:14). Produced 538-line and ~539-line analyses with DeepWiki grounding. |
| 2: Design | ttnn-operation-architect | ~4m (09:07-09:11) | COMPLETE | Produced comprehensive op_design.md with 6-CB layout, helper mappings, and 6 TDD stages. |
| 3: Build | ttnn-generic-op-builder | ~8m (09:17-09:25) | COMPLETE | Created 16 files. 1 compilation failure (bad tensor_accessor.hpp include), fixed immediately. 5/5 integration tests passed. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~20m (09:29-09:49) | COMPLETE | All 6 TDD stages passed. 7 hard attempts total (budget 36). Only 1 numerical failure on stage 5 (softmax_h_stable). No device hangs. |
| 5: Report | orchestrator | ~2m (09:49-09:51) | COMPLETE | REPORT.md generated. |
| **Total** | | **~53m** | | Clean end-to-end run. Phase 4 consumed ~38% of total time. |

### Agent Duration Breakdown

Duration method: breadcrumb `start` -> `complete` events as primary source. Git commit timestamps used as cross-validation. Both sources are consistent.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (moreh_sum_h) | 08:58:55 | 09:05:14 | ~6m 19s | 0 | ~6m active (reading code, querying DeepWiki) |
| ttnn-operation-analyzer (moreh_sum_w) | 08:58:58 | 09:09:26 | ~10m 28s | 0 | ~10m active (reading code, querying DeepWiki, commit overhead) |
| ttnn-operation-architect | 09:07:24 | 09:11:38 | ~4m 14s | 0 | ~4m active design |
| ttnn-generic-op-builder | 09:17:13 | 09:25:50 | ~8m 37s | 1 | ~7m active, ~1.5m debugging tensor_accessor.hpp include |
| ttnn-kernel-writer-tdd | 09:29:31 | 09:49:20* | ~19m 49s | 1 (softmax_h_stable) | ~16m productive coding, ~4m debugging dim-H REDUCE_ROW vs REDUCE_COL |

*Note: The kernel writer's breadcrumbs end at 09:43:50 (CB sync check after softmax_h_stable fix). The final stages (softmax_h_stable retest + softmax_h_unstable) are evidenced only by git commits at 09:48:16 and 09:49:20.

### Duration Visualization

```
Phase 0-1 |####################|                                (~13m, 2 analyzers parallel)
Phase 2        |########|                                       (~4m)
Phase 3                  |################|                     (~8m)
Phase 4                                   |######################################| (~20m)
Phase 5                                                                         |###| (~2m)
          0    5    10   15   20   25   30   35   40   45   50 min

Longest phase: Phase 4 (20m) -- 6 TDD stages with one numerical debugging cycle
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~13m | 25% | 2 analyzers in parallel, queried DeepWiki 4+ times each |
| Design (Phase 2) | ~4m | 8% | Clean single-pass design |
| Build (Phase 3) | ~8m | 15% | 1 free retry (include fix) |
| Kernel implementation (Phase 4) | ~20m | 38% | 6 TDD stages |
| -- Productive coding | ~16m | 30% | Writing kernel code that passed |
| -- Debugging/retries | ~4m | 8% | H2 hypothesis -> fix -> retest on softmax_h_stable |
| Reporting (Phase 5) | ~2m | 4% | |
| Inter-phase gaps | ~6m | 10% | Agent startup/shutdown overhead between phases |
| **Total** | **~53m** | **100%** | |

---

## 2. What Went Well

### 1. Exceptional TDD pass rate: 5 of 6 stages passed first try

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: TDD state shows stages 0-3 and 5 all passed on attempt 1 with 0 free retries. Only stage 4 (softmax_h_stable) required a second attempt. Total budget: 7/36 hard attempts (19% utilization). Zero device hangs across all 24 test shapes (4 shapes x 6 stages).
**Why it worked**: The architect's design document was exceptionally detailed -- it specified exact helper function calls with template parameters, CB state tables after each phase, and explicit manual pop requirements. The kernel writer could essentially transcribe the design into code.

### 2. Zero device hangs across all 6 stages

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: All 24 test shapes completed without hangs. No `kill_timeout` or `device_reset` events in breadcrumbs. All test runs finished within typical timeframes (30-90 seconds per stage). The kernel writer performed CB sync checks before every test run (8 `cb_sync_check` breadcrumb entries, all `balanced:true`).
**Why it worked**: The architect's CB state tables after each phase made the push/pop balance explicit. The kernel writer proactively caught the double-buffered cb_out wait pattern issue before the first test run (breadcrumb at 09:31:25).

### 3. Comprehensive and correct helper library usage

**Phase/Agent**: Phase 2 (Design) + Phase 4 (Implementation)
**Evidence**: All 5 compute phases use helpers: `reduce<MAX>`, `reduce<SUM>`, `sub<>`, `mul<>`, `copy_tiles<>`. The reader uses `prepare_reduce_scaler<>`. No raw tile register management anywhere. No raw `tile_regs_acquire/commit/wait/release` calls in the compute kernel -- all DST management delegated to helpers.
**Why it worked**: The architect mapped every compute phase to a specific helper call with precise template parameters in op_design.md Part 2.

### 4. Proactive forward-looking implementation

**Phase/Agent**: Phase 4 / ttnn-kernel-writer-tdd
**Evidence**: During stage 3 (softmax_w_stable), the kernel writer implemented both the stable and unstable paths. Stage 4 (softmax_w_unstable) required zero kernel changes. Similarly, reader/writer strided access for dim=-2 was implemented from stage 0 (passthrough). Stage 6 (softmax_h_unstable) also required zero kernel changes.
**Why it worked**: Good engineering judgment combined with the design doc's clear TDD stage plan showing what each stage would add.

### 5. High-quality reference analyses

**Phase/Agent**: Phase 1 / ttnn-operation-analyzer
**Evidence**: moreh_sum_w analysis (538 lines) covered: work unit definition, CB configuration, compute kernel structure, reduce helper signatures, FP32 accumulation handling, and softmax-specific observations. moreh_sum_h analysis (539 lines) covered: REDUCE_COL dispatch, chunk-based DEST processing, Accumulate::at() mechanism, and mask handling. Both queried DeepWiki 4+ times for hardware-level grounding.
**Why it worked**: The analyzers were thorough and included softmax-specific sections mapping reference patterns to the target operation.

---

## 3. Issues Found

### Issue 1: Design document incorrectly stated both dimensions use REDUCE_ROW

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- softmax_h_stable (Stage 5) |
| Agent | ttnn-operation-architect (root cause) / ttnn-kernel-writer-tdd (discovered) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~4 minutes (09:38:32 first test start -> 09:43:50 fix verified) |

**Problem**: The design document's "Unified Compute Design" section stated: "Both dim=-1 and dim=-2 use REDUCE_ROW in the compute kernel. The difference is only in dataflow." This was architecturally incorrect. For dim=-2, the compute kernel must use REDUCE_COL and ROW broadcast (not REDUCE_ROW and COL broadcast). The kernel writer discovered this when the non-square shape `(1,1,32,256)` failed numerically on softmax_h_stable -- the single-tile `(1,1,32,32)` passed accidentally because for square tiles, REDUCE_ROW and REDUCE_COL produce equivalent results.

The kernel writer formed two hypotheses: H1 (MEDIUM confidence, shape-specific issue) and H2 (HIGH confidence, wrong reduce direction). H2 was correct. The fix required adding `is_dim_h` as compile-time arg index 9 to the compute kernel and parametrizing `reduce_dim`, `bcast_dim`, `reduce_block`, and `binary_block` based on the dimension.

**Root Cause**: The architect conflated the dataflow-level "virtual row" abstraction (where both dim=-1 and dim=-2 work units are treated as "rows") with the tile-level reduction direction. The moreh_sum_h analysis explicitly documents that H-reduction maps to `ReduceDim::REDUCE_COL` at the LLK level (moreh_sum_h_analysis.md line 131), but the architect may not have read or internalized this detail.

Additionally, the architect's compute CT arg table (op_design.md lines 101-108) listed only indices 0-8, omitting `is_dim_h`. The compute kernel had no way to distinguish dimensions.

**Fix for agents**:
- **ttnn-operation-architect**: Add a mandatory validation rule: "For operations supporting multiple reduce dimensions, verify the tile-level reduce direction for each. REDUCE_ROW reduces across W within tiles. REDUCE_COL reduces across H within tiles. If the operation reduces along H (dim=-2), use REDUCE_COL. If along W (dim=-1), use REDUCE_ROW. Never assume 'same compute for both dims' without verification."
- **ttnn-operation-architect**: The CT arg table must include all args needed for ALL TDD stages, not just the initial stages. If dim=-2 requires a dimension flag, it must be in the CT args from the start.

### Issue 2: tensor_accessor.hpp include path does not exist

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build (stub validation) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry (compilation error, easy fix) |
| Time Cost | ~1.5 minutes |

**Problem**: The builder generated kernel stubs with `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` which does not exist. Compilation failed with `fatal error: tensor_accessor.hpp: No such file or directory`. The TensorAccessor type is provided through `api/dataflow/dataflow_api.h`.

**Root Cause**: The builder agent's system prompt contains a helper-to-include mapping table with an incorrect entry for TensorAccessor. The builder explicitly flagged this in its execution log Section 7 (Recommendation 1) and in the upstream_feedback breadcrumb.

**Fix for agents**:
- **ttnn-generic-op-builder**: Remove `tensor_accessor.hpp` from the include mapping table in the system prompt. Add note that TensorAccessorArgs is available from `api/dataflow/dataflow_api.h`.

### Issue 3: Missing kernel writer execution log

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 |
| Time Cost | 0 (observability gap, not a runtime issue) |

**Problem**: The kernel writer produced breadcrumbs (34 entries) but no `ttnn-kernel-writer-tdd_execution_log.md`. The builder produced both breadcrumbs and an execution log. This creates an observability gap -- the breadcrumbs are sufficient for timeline reconstruction but lack the structured recovery tables and handoff notes that execution logs provide.

**Root Cause**: The kernel writer agent either lacks the execution log generation step in its instructions, or it was skipped.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Add a mandatory execution log generation step at session end, matching the builder's format (Sections 1-7: Input Interpretation, Execution Timeline, Recovery Summary, Deviations, Artifacts, Handoff Notes, Instruction Improvement Recommendations).

### Issue 4: Incomplete breadcrumb coverage for late TDD stages

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- softmax_h_stable/softmax_h_unstable |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 |
| Time Cost | 0 (observability gap) |

**Problem**: The kernel writer's breadcrumbs end at line 34 (CB sync check for softmax_h_stable fix). Missing: `test_run:pass` for softmax_h_stable after the fix, `stage_complete` for softmax_h_stable, `stage_start`/`test_run`/`stage_complete` for softmax_h_unstable, and the overall `complete` event. Git commits at 09:48:16 and 09:49:20 confirm these stages passed, but the breadcrumb trail is truncated.

**Root Cause**: The agent likely ran low on context or time and stopped logging breadcrumbs while continuing to execute kernel implementations and tests.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Ensure `stage_complete` and `complete` breadcrumb events are always emitted, even if abbreviated. At minimum, `test_run` with pass/fail per stage is required for analysis.

### Issue 5: Architect may have started before analyzers completed

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 1/Phase 2 overlap |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 |
| Time Cost | 0 (but possibly contributed to Issue 1) |

**Problem**: The architect started at 09:07:24 but the moreh_sum_h analyzer did not commit until 09:09:17 and moreh_sum_w until 09:09:26. The architect breadcrumbs show `"input_files":["N/A"]` for predecessor, suggesting it may not have read the analysis files. The moreh_sum_h analysis explicitly documents that H-reduction maps to REDUCE_COL (line 131), which is precisely the detail the architect got wrong.

**Root Cause**: The orchestrator may have launched Phase 2 before all Phase 1 commits were confirmed.

**Fix for agents**:
- **Orchestrator**: Gate Phase 2 start on confirmed completion of all Phase 1 analyzer commits.
- **ttnn-operation-architect**: Validate that expected analysis files exist and are non-empty before beginning design.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| passthrough | ~3m (09:29:58 - 09:32:25) | 0 free, 1 hard | PASS | Clean -- writer CB wait fix was proactive (before test) |
| exp_only | ~1m (09:33:14 - 09:34:08) | 0 free, 1 hard | PASS | Clean -- single lambda addition |
| softmax_w_stable | ~1.5m (09:34:44 - 09:36:15) | 0 free, 1 hard | PASS | Clean -- full 4-phase compute implemented |
| softmax_w_unstable | ~0.7m (09:37:11 - 09:37:52) | 0 free, 1 hard | PASS | Clean -- zero kernel changes, test-only |
| softmax_h_stable | ~10m (09:38:32 - 09:48:16) | 0 free, 2 hard | PASS | Numerical mismatch: wrong REDUCE_ROW for dim=-2. ~4m diagnosis, ~6m fix + retest |
| softmax_h_unstable | ~1m (09:48:16 - 09:49:20) | 0 free, 1 hard | PASS | Clean -- zero kernel changes, test-only |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | softmax_h_stable | ttnn-kernel-writer-tdd | ~10m | 19% | Numerical mismatch on non-square shape; needed REDUCE_COL not REDUCE_ROW for dim=-2, plus upstream fix to program_descriptor.py | 1 retry | Architect's design said both dims use REDUCE_ROW |
| 2 | Inter-phase gaps | orchestrator | ~6m | 10% | Dead time between agent completion and next agent start | N/A | Agent orchestration overhead |
| 3 | Analysis phase | ttnn-operation-analyzer | ~10m | 19% | Two analyzers ran in parallel with 4+ DeepWiki queries each (~1m per query) | 0 | DeepWiki query latency |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-generic-op-builder | Generated kernel stubs with tensor_accessor.hpp include | Header does not exist; had to remove and recompile | Fix include mapping table in builder system prompt |
| ttnn-kernel-writer-tdd | First softmax_h_stable test run with REDUCE_ROW (09:38-09:40) | Wrong reduce direction for dim=-2 | Architect should specify REDUCE_COL for dim=-2 from the start |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: Orchestrator -> ttnn-operation-analyzer

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation paths (moreh_sum_w, moreh_sum_h) |
| Quality | GOOD |
| Issues | None -- correct references selected for W and H reduction patterns |
| Downstream Impact | Both analyzers produced thorough, useful analyses |
| Suggestion | None needed |

### Handoff 2: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | moreh_sum_w_analysis.md, moreh_sum_h_analysis.md |
| Quality | ADEQUATE (but possibly not read due to timing overlap) |
| Issues | Architect started ~2m before analyzer commits completed. Architect breadcrumbs show `"input_files":["N/A"]`. The moreh_sum_h analysis explicitly documents REDUCE_COL for H-reduction (line 131: "ReduceOpDim::H at the TTNN API level maps to ReduceDim::REDUCE_COL at the LLK/hardware level"), which is precisely the detail the architect got wrong. |
| Downstream Impact | Possibly contributed to the REDUCE_ROW error that cost 1 hard retry and ~4m debugging in Phase 4. |
| Suggestion | Orchestrator must gate Phase 2 on Phase 1 completion. Architect should confirm it has read both analysis documents before starting design. |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | GOOD |
| Issues | Minor: .tdd_state.json had a transient inconsistency. Builder worked around via git checkout. |
| Downstream Impact | Minimal -- builder recovered quickly. |
| Suggestion | Architect should use orchestrator CLI for stage registration to avoid file-based IPC issues. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program_descriptor.py, test files, execution log with handoff notes |
| Quality | GOOD |
| Issues | None. Builder's handoff notes (execution log Section 6) were detailed: all CT/RT arg indices, CB configurations, include path warnings (correctly noted tensor_accessor.hpp issue), known limitations. |
| Downstream Impact | Positive -- kernel writer had clear, actionable guidance. |
| Suggestion | Builder could additionally validate that all architect-specified CT args are present in program_descriptor.py. |

### Handoff 5: ttnn-operation-architect -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 (kernel implementation details) |
| Quality | GOOD overall, with one critical gap |
| Issues | The "Unified Compute Design" section (op_design.md lines 59-68) incorrectly claimed both dimensions use REDUCE_ROW. The compute CT arg table (lines 101-108) omitted is_dim_h. |
| Downstream Impact | 1 hard attempt wasted, ~4 minutes debugging. Kernel writer had to add is_dim_h and parametrize all dimension-dependent operations. |
| Suggestion | When designing multi-dimension operations, trace data through CBs for EACH dimension variant and verify reduce direction matches physical tile layout. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder (system prompt) | ttnn-generic-op-builder | Remove tensor_accessor.hpp from include mapping table; TensorAccessorArgs is in dataflow_api.h | HIGH | HIGH |
| ttnn-operation-architect (instructions) | ttnn-kernel-writer-tdd | Must specify tile-level reduce direction for EACH supported dim. Never assume "same compute for both dims." | HIGH | HIGH |
| ttnn-operation-architect (instructions) | ttnn-generic-op-builder | Should register stages via orchestrator CLI, not write .tdd_state.json directly | MEDIUM | MEDIUM |
| ttnn-kernel-writer-tdd (instructions) | self-reflection | Add mandatory execution log generation at session end | HIGH | MEDIUM |
| orchestrator | self-reflection | Gate Phase 2 start on confirmed Phase 1 completion | MEDIUM | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Design validation | Architect stated "both dims use REDUCE_ROW" which was wrong for dim=-2 | Add mandatory reduce-direction checklist: for each dim, verify tile-level reduce direction vs data-level meaning | HIGH |
| Build validation | Builder included nonexistent header, caught at compilation | Fix include mapping table in builder system prompt | HIGH |
| Phase gating | Architect may have started before analyzers finished | Orchestrator should not launch Phase 2 until all Phase 1 commits confirmed | MEDIUM |
| Logging | Kernel writer produced no execution log and truncated breadcrumbs | Make execution log generation mandatory for all agents | MEDIUM |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Only 1 numerical failure, resolved in ~4 minutes via 2 hypotheses. Not a context burn. |
| 2 | Too many planning stages (DONE -- merged) | N/A | Pipeline already uses merged Architect agent. |
| 3 | .tdd_state.json coupling is fragile | YES | Builder observed stale .tdd_state.json (1 stage vs 6), recovered via git checkout. |
| 4 | No fast path for simple operations | NO | Softmax is medium-complexity (6 CBs, 4 compute phases), appropriate for full pipeline. |
| 6 | Builder runs on Sonnet | POSSIBLY | Builder had 1 compilation error from bad include path. This is a system prompt issue, not model capability. |
| 7 | Discovery keyword matching | NO | Discovery correctly identified moreh_sum_w and moreh_sum_h. |
| 9 | No architect/builder cross-validation | YES | Missing is_dim_h CT arg for compute was not caught until Phase 4 runtime. A static cross-check between design CT args and program_descriptor.py would have caught this. |
| 11 | No incremental re-run capability | NO | Pipeline completed successfully without needing re-run. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect may conflate data-level and tile-level reduce dimensions | The architect stated "both dims use REDUCE_ROW" because the virtual-row abstraction makes both look like row reductions at the dataflow level. But tile-level reduce must match the actual dimension (REDUCE_COL for H). Square-tile tests mask this error. | HIGH |
| Phase 1/2 overlap: architect may start before analyzers commit | Architect breadcrumbs show start at 09:07:24 with `input_files: ["N/A"]`, while analyzers completed at 09:09. The architect may have missed critical details (like REDUCE_COL for H-reduction). | MEDIUM |
| Kernel writer does not generate execution logs | Builder generates both breadcrumbs and execution_log.md, but kernel writer generates only breadcrumbs (and truncated at that). Observability gap for the longest pipeline phase. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Architect must specify per-dimension tile-level reduce direction

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add mandatory validation rule: "When designing operations that support multiple reduce dimensions, you MUST specify the tile-level reduce direction for EACH dimension separately. REDUCE_ROW reduces along the W axis within tiles. REDUCE_COL reduces along the H axis within tiles. These are NOT interchangeable. Include a validation table mapping each logical dim to its tile-level ReduceDim. The compute kernel MUST receive a dimension parameter as a compile-time arg."
- **Expected Benefit**: Prevents REDUCE_ROW vs REDUCE_COL confusion that caused 1 retry and ~4m debugging
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Fix tensor_accessor.hpp include mapping in builder prompt

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent prompt (include mapping table)
- **Change**: Remove entry `| TensorAccessor | #include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp" |`. Add note: "TensorAccessorArgs is available via `api/dataflow/dataflow_api.h` -- no separate include needed."
- **Expected Benefit**: Eliminates guaranteed compilation failure on every op using TensorAccessor
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Add execution log generation to kernel writer

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd agent instructions
- **Change**: Add mandatory session-end step: generate `ttnn-kernel-writer-tdd_execution_log.md` with per-stage attempt breakdown, recovery table, upstream fixes applied, deviations, and handoff notes. Template should match builder's execution log format.
- **Expected Benefit**: Closes observability gap for the longest pipeline phase. Enables complete self-reflection analysis.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Gate Phase 2 on Phase 1 completion

- **Type**: pipeline_change
- **Target**: Orchestrator script
- **Change**: Add explicit wait for all Phase 1 analyzer git commits before launching Phase 2 architect. Verify analyzer output files exist and are non-empty.
- **Expected Benefit**: Ensures architect has access to analysis results, potentially preventing design errors from missed context
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Static cross-check between design CT args and program descriptor

- **Type**: new_validation
- **Target**: Pipeline (between Phase 3 and Phase 4)
- **Change**: After builder completes, compare CT arg lists in op_design.md against CT arg lists in program_descriptor.py. Flag args present in design but missing from descriptor (or vice versa).
- **Expected Benefit**: Would have caught missing is_dim_h CT arg for compute before any tests ran
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4 | Correctly identified moreh_sum_w and moreh_sum_h as references for both reduction dimensions |
| Analysis quality | 5 | Exceptional 500+ line analyses with DeepWiki grounding, complete helper API docs, softmax-specific mapping observations |
| Design completeness | 4 | Comprehensive CB layout, helper mappings, and TDD plan. Deducted 1 point for incorrect "unified REDUCE_ROW" claim and missing is_dim_h CT arg |
| Build correctness | 4 | All 16 files correct. One free retry for known bad include path. Excellent handoff notes. |
| Kernel implementation | 5 | 5/6 stages first-pass. Fast diagnosis (2 hypotheses) of the one failure. No hangs. Full helper library usage. Forward-looking implementation. |
| Inter-agent communication | 3 | Builder-to-kernel-writer handoff excellent. Analyzer-to-architect handoff possibly missed due to timing overlap. Missing is_dim_h flowed uncaught from design to implementation. |
| Logging/observability | 3 | Builder logs comprehensive (breadcrumbs + execution log). Kernel writer breadcrumbs good but truncated (no execution log, incomplete late-stage coverage). No analyzer execution logs. |
| Helper usage compliance | 5 | Every compute phase uses the appropriate helper. Reader uses prepare_reduce_scaler. No raw tile register management. No missed helpers. See Section 10. |

### Top 3 Things to Fix

1. **Architect must validate tile-level reduce direction per dimension** -- The "unified REDUCE_ROW" assumption was mathematically wrong for dim=-2 and only passed on square-tile tests accidentally. This class of error will recur for every multi-dimension reduction operation.

2. **Fix tensor_accessor.hpp include mapping in builder system prompt** -- Systemic issue that causes a guaranteed compilation failure on every operation using TensorAccessor until fixed.

3. **Gate Phase 2 on Phase 1 completion and require execution logs from kernel writer** -- The architect may have started without reading analyzer results (contributing to the REDUCE_ROW error), and the kernel writer's missing execution log left a significant observability gap.

### What Worked Best

The architect's design document quality was the strongest aspect of this run. The CB state tables after each compute phase, the explicit helper call specifications with precise template parameters, the per-phase CB lifetime annotations, and the progressive TDD stage plan enabled the kernel writer to implement 5 of 6 stages on the first attempt with zero device hangs. The compute kernel code (`softmax_compute.cpp`) is a near-direct translation of the design document's pseudocode into helper library calls, which is exactly the pipeline's intended operating mode. Combined with the kernel writer's systematic CB sync checking and proactive forward-looking implementation, this resulted in only 19% budget utilization (7/36 hard attempts) -- demonstrating that detailed, correct designs dramatically reduce implementation friction.

---

## 10. Helper Usage Audit

### Available Helpers

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `reduce_helpers_compute.hpp` | `reduce<PoolType, ReduceDim, InputPolicy, ReconfigMode>()` | YES -- used for MAX and SUM reductions |
| `reduce_helpers_dataflow.hpp` | `prepare_reduce_scaler<cb_id>()`, `calculate_and_prepare_reduce_scaler<>()` | YES -- used in reader for scaler tile (1.0f) |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()`, `square<>()` with broadcast | YES -- used for sub(input, max) and mul(exp, 1/sum) |
| `copy_tile_helpers.hpp` | `copy_tiles<InputPolicy, ReconfigMode>()` with optional post-op | YES -- used for passthrough and exp-only stages |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT` constant | YES -- used internally by reduce and binary helpers |
| `cb_helpers.hpp` | CB utility functions | NO -- not needed; helpers manage CBs internally |
| `tilize_helpers.hpp` | `tilize<>()` | NO -- input already in TILE_LAYOUT |
| `untilize_helpers.hpp` | `untilize<>()` | NO -- output stays in TILE_LAYOUT |
| `l1_helpers.hpp` | L1 address helpers | YES -- used internally by reduce_helpers_dataflow |
| `common_types.hpp` | Shared type definitions (policies, enums) | YES -- used internally by all helpers |

### Per-Phase Helper Compliance

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| reader | Tile loading | Raw (TensorAccessor + noc_async_read) | Raw (TensorAccessor + noc_async_read) | Raw Justified | No helper exists for general tile loading with strided access patterns |
| reader | Scaler prep | `prepare_reduce_scaler<cb_scaler>(1.0f)` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f)` | Correct | Helper used exactly as designed |
| compute | Phase 1: MAX reduction | `reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>` | `reduce<MAX, reduce_dim, WaitUpfrontNoPop, NONE>` | Correct | Improved over design by parametrizing reduce_dim for both dimensions |
| compute | Phase 2: SUB + EXP | `sub<COL, NoWaitNoPop, WaitAndPopPerTile, PerTile>` + exp_post_op | `sub<bcast_dim, NoWaitNoPop, WaitAndPopPerTile, PerTile, INPUT_AND_OUTPUT>` + exp_post_op | Correct | Improved over design by parametrizing bcast_dim |
| compute | Phase 3: SUM + RECIP | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` + recip_post_op | `reduce<SUM, reduce_dim, WaitUpfrontNoPop, INPUT_AND_OUTPUT>` + recip_post_op | Correct | Improved over design by parametrizing reduce_dim |
| compute | Phase 4: MUL | `mul<COL, NoWaitNoPop, WaitAndPopPerTile, PerTile>` | `mul<bcast_dim, NoWaitNoPop, WaitAndPopPerTile, PerTile, INPUT_AND_OUTPUT>` | Correct | Improved over design by parametrizing bcast_dim |
| compute | Unstable Phase 1: COPY + EXP | `copy_tiles<WaitAndPop, NONE>` + exp_post_op | `copy_tiles<WaitAndPop, NONE>` + exp_post_op | Correct | Helper used exactly as designed |
| writer | Tile writing | Raw (TensorAccessor + noc_async_write) | Raw (TensorAccessor + noc_async_write) | Raw Justified | No helper exists for general tile writing with strided access patterns |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | 8 (reader: 2, compute: 5 stable + 1 unstable-only, writer: 1) |
| Phases using helpers correctly | 6 |
| Phases with justified raw code | 2 (reader tile loading, writer tile writing -- no helpers exist) |
| Phases with missed helpers | 0 |
| Phases with misused helpers | 0 |
| **Helper compliance rate** | **100%** |

### Redundant CB Operations Around Helpers

No redundant CB operations detected around helper calls.

The kernel has two `cb_pop_front` calls adjacent to helper calls:
1. `cb_pop_front(cb_input, R)` at softmax_compute.cpp:73 -- after `sub<NoWaitNoPop, ...>`. Required because `NoWaitNoPop` on input A explicitly delegates pop to the caller. Not redundant.
2. `cb_pop_front(cb_exp, R)` at softmax_compute.cpp:97 -- after `mul<NoWaitNoPop, ...>`. Same pattern. Required. Not redundant.

The `cb_wait_front(cb_scaler, 1)` at line 52 is also not redundant -- it ensures the persistent scaler tile is available before the first reduce call in the loop.

### Missed Helper Opportunities

All available helpers were used correctly. No missed opportunities.
