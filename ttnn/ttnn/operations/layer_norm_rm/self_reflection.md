# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | 3x ttnn-operation-analyzer, 1x ttnn-operation-architect, 1x ttnn-generic-op-builder, 1x ttnn-kernel-writer-tdd, 1x orchestrator |
| Total Git Commits | 12 (9128348cab through 671dd1a25b) |
| Total Pipeline Duration | ~62 minutes (12:02 to 12:56 UTC from git timestamps) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~8 min (before 11:54) | COMPLETE | Identified tilize, untilize, softmax as references; assigned role-based focus directives |
| 1: Analysis | 3x ttnn-operation-analyzer (parallel) | ~13 min (11:54 - 12:07) | COMPLETE | Three analyzers ran in parallel: tilize (8 min), untilize (8 min), softmax (6 min). Produced 87KB of analysis |
| 2: Design | ttnn-operation-architect | ~9 min (12:08 - 12:17) | COMPLETE | Produced op_design.md (19KB), 4 TDD stages, 13 CBs, 10 compute phases |
| 3: Build | ttnn-generic-op-builder | ~10 min (12:19 - 12:28) | COMPLETE | Created Python infra, 14 CBs, stub kernels, 23 tests collected. Had 2 compilation errors (both FREE), fixed in 2 retries |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~26 min (12:28 - 12:54) | COMPLETE | All 4 stages passed. 1 hard attempt consumed on stage 4 (affine_transform) |
| 5: Report | orchestrator | ~3 min (12:54 - 12:56) | COMPLETE | Produced REPORT.md |

### Agent Duration Breakdown

Duration calculation method: primary source is breadcrumb `"event":"start"` and `"event":"complete"` timestamps. Git commit timestamps used as cross-check. Breadcrumbs and commits are consistent within ~1 minute.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 2026-03-10T11:54:13 | 2026-03-10T12:03:02 | ~9m | 0 | ~9m active (reading, researching, writing) |
| ttnn-operation-analyzer (untilize) | 2026-03-10T11:54:42 | 2026-03-10T12:03:47 | ~9m | 0 | ~9m active |
| ttnn-operation-analyzer (softmax) | 2026-03-10T12:01:32 | 2026-03-10T12:06:57 | ~5.5m | 0 | ~5.5m active |
| ttnn-operation-architect | 2026-03-10T12:08:48 | 2026-03-10T12:16:53 | ~8m | 0 | ~8m active (all design, no debugging) |
| ttnn-generic-op-builder | 2026-03-10T12:19:32 | 2026-03-10T12:27:28 | ~8m | 2 free (compile errors) | ~5m active, ~3m fixing compile paths |
| ttnn-kernel-writer-tdd | 2026-03-10T12:28:56 | 2026-03-10T12:53:50 | ~25m | 1 hard (stage 4) | ~20m coding, ~5m debugging stage 4 tilize volume issue |

### Duration Visualization

```
Phase 0  |████|                                                (~8m)
Phase 1  |██████████████|                                      (~13m) 3 analyzers in parallel
Phase 2       |██████████|                                     (~9m)
Phase 3            |██████████|                                (~10m)
Phase 4                 |████████████████████████████████████|  (~26m) 4 TDD stages
Phase 5                                                    |██| (~3m)
         0    5    10   15   20   25   30   35   40   45   50  55  60 min

Longest phase: Phase 4 (26m) -- kernel implementation across 4 TDD stages, with one retry on stage 4 affine_transform
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~21 min | 34% | 3 analyzers, 87KB output |
| Design (Phase 2) | ~9 min | 15% | |
| Build (Phase 3) | ~10 min | 16% | 2 free retries (compile path issues) |
| Kernel implementation (Phase 4) | ~26 min | 42% | 4 TDD stages |
| -- Productive coding | ~21 min | 34% | Stages 1-3 first-pass, stage 4 implementation |
| -- Debugging/retries | ~5 min | 8% | Stage 4 tilize volume error hypothesis + fix |
| Reporting (Phase 5) | ~3 min | 5% | |
| **Total** | **~62 min** | **100%** | |

Note: Phase 0 duration overlaps with the orchestrator setup before analyzer breadcrumbs begin. The ~8 min estimate is based on the gap between the branch creation and first analyzer breadcrumb.

---

## 2. What Went Well

### 1. First-Pass Success on 3 of 4 TDD Stages

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: Stages 1 (data_pipeline), 2 (reduce_mean_sub), and 3 (variance_normalize) all passed on the first attempt with zero retries. The .tdd_state.json shows `attempts: 0` for all three. Git timestamps confirm rapid progression: data_pipeline passed at 12:34, reduce_mean_sub at 12:36 (2 min later), variance_normalize at 12:39 (3 min later).
**Why it worked**: The architect's op_design.md was exceptionally detailed, providing per-phase CB state tables, explicit wait/pop annotations, and precise helper call signatures with policy templates. The kernel writer could translate the design almost directly into code. The helper library abstraction (tilize, untilize, reduce, binary_op) worked as documented.

### 2. CB Layout Design -- Zero CB-Related Bugs Across All Stages

**Phase/Agent**: Phase 2 (ttnn-operation-architect)
**Evidence**: 13 CBs were designed with explicit page counts, data formats, lifetimes, and push/pop annotations. The .breadcrumbs.md file contains a detailed CB sync verification audit for the final stage (affine_transform) showing all 13 CBs balanced. No CB-related errors appear anywhere in the breadcrumbs or failure history.
**Why it worked**: The architect validated CB decisions against helper library requirements inline during design. The binary op broadcast verification table in op_design.md (Phase/Op/CB_A/CB_B/Broadcast mapping) prevented incorrect broadcast dimensions. The explicit "CB state after Phase N" tables at each phase boundary made reasoning about data lifetime tractable.

### 3. Parallel Analyzer Execution Worked Efficiently

**Phase/Agent**: Phase 1 (3x ttnn-operation-analyzer)
**Evidence**: Three analyzers ran concurrently. The tilize and untilize analyzers started within 30 seconds of each other (11:54:13 and 11:54:42). All three completed within a ~13 minute window. The softmax analyzer started slightly later (12:01:32) but was the fastest (~5.5 min). The total analysis phase was ~13 min wall clock versus ~24 min if sequential.
**Why it worked**: Role-based focus directives (input_stage, output_stage, compute_core) scoped each analyzer's work, preventing redundant coverage.

### 4. Clean Kernel Code Quality

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: Final kernel files are compact and well-structured: reader (103 lines), compute (148 lines), writer (44 lines). The compute kernel follows the 10-phase structure exactly as designed. Helper calls use correct policy templates throughout. No debugging artifacts, no commented-out code, no dead code paths.
**Why it worked**: The incremental TDD approach let each stage build on a working foundation. The kernel writer committed to cleaning up discussion comments before finalizing (commit 4e453b5285: "Cleaned up ~90 lines of design discussion comments in compute kernel").

### 5. Kernel Writer Autonomously Fixed Upstream Issues

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: Two upstream fixes were made to the program descriptor during implementation: (1) Fixed bf16_tile_size from `buffer_page_size()` to `ttnn.tile_size(ttnn.bfloat16)` during stage 1; (2) Added gamma/beta TensorAccessorArgs during stage 4. Both fixes were correctly scoped and did not introduce regressions.
**Why it worked**: The kernel writer had sufficient context about the program descriptor's role and was empowered to make targeted upstream fixes rather than working around bugs.

---

## 3. Issues Found

### Issue 1: Architect Design Did Not Account for ttnn.tilize() Volume Constraint on Gamma/Beta

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- affine_transform |
| Agent | ttnn-operation-architect (root cause), ttnn-kernel-writer-tdd (discovered) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~5 minutes |

**Problem**: The architect's design (op_design.md, "Critical Notes" item 4) proposed two alternatives for gamma/beta handling: (a) reader writes RM sticks, compute tilizes in-kernel; or (b) "host converts to TILE before kernel launch (simpler, recommended for initial impl)." The kernel writer chose option (b) and called `ttnn.tilize()` on the gamma/beta tensors (shape 1,1,1,W). This failed with `TT_FATAL: Input tensor physical volume (32) must be divisible by TILE_HW (1024)`. The gamma/beta tensor has H=1, which is below the 32-row minimum required for tilize.

The kernel writer diagnosed this via `.breadcrumbs.md` hypothesis: "ttnn.tilize() cannot tilize a (1,1,1,W) tensor because H=1 < 32. Tilize requires H to be a multiple of 32." The fix was to `ttnn.repeat(gamma, [1,1,32,1])` before tilizing -- expanding the single row to 32 rows.

**Root Cause**: The architect recommended host-side tilize without verifying that `ttnn.tilize()` has a minimum volume constraint. The tilize analysis (agent_logs/tilize_analysis.md) covered multi-core interleaved tilize of full tensors, not edge cases with sub-tile tensors. The architect's "recommended for initial impl" annotation provided false confidence.

**Fix for agents**:
- **ttnn-operation-architect**: When recommending host-side tilize for parameter tensors (gamma, beta, scalers), verify that the tensor shape meets `ttnn.tilize()` requirements (H >= 32, W >= 32, volume >= 1024). If a parameter has H=1, the design must explicitly specify the repeat/pad step.
- **ttnn-operation-analyzer (tilize)**: Include minimum shape requirements for ttnn.tilize() in the analysis output, specifically the TILE_HW=1024 volume constraint.

### Issue 2: Builder Used Incorrect Include Paths in Stub Kernels

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 2 free retries |
| Time Cost | ~3 minutes |

**Problem**: The builder's first test run failed with `tensor_accessor.hpp not found`. Second attempt failed with `compute_kernel_api.h not found`. The builder's execution log (Section 4) documents both: "Include path in system prompt is wrong for kernel-side" and "Template uses wrong path."

**Root Cause**: The builder's system prompt or template files contain stale include paths. The correct paths are `api/dataflow/dataflow_api.h` (reader/writer) and `api/compute/common.h` (compute), but the builder initially tried `tensor_accessor.hpp` and `compute_kernel_api.h`. The builder recovered by empirical verification.

**Fix for agents**:
- **ttnn-generic-op-builder**: Update stub kernel templates to use verified include paths: `api/dataflow/dataflow_api.h` and `api/compute/common.h`. The builder's handoff notes already recommend this -- the fix should be applied to the template files.
- **Pipeline infrastructure**: Add a compile-only validation step in the builder that tests stub compilation before committing, so include path errors are caught within the builder's session and don't appear as "retries."

### Issue 3: Builder Used buffer_page_size() Instead of tile_size() for CB Page Sizes

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- data_pipeline (discovered by kernel writer) |
| Agent | ttnn-generic-op-builder (root cause), ttnn-kernel-writer-tdd (fixed) |
| Retries Consumed | 0 (kernel writer fixed proactively before testing) |
| Time Cost | ~2 minutes of kernel writer time |

**Problem**: The program descriptor's bf16_tile_size was computed using `input_tensor.buffer_page_size()`, which returns the RM stick size (W * 2 bytes) for row-major tensors, not the tile size (2048 bytes). This would have caused every tile-based CB to have incorrect page sizes.

The kernel writer's breadcrumb at 12:32:58 documents: "replaced input_tensor.buffer_page_size() with ttnn.tile_size(ttnn.bfloat16) for bf16_tile_size... buffer_page_size() returns stick_size for RM tensors, not tile_size."

**Root Cause**: The builder conflated page size (layout-dependent) with tile size (fixed at 2048 for bf16). For RM tensors, `buffer_page_size()` equals `stick_size`, not `tile_size`. This is a common confusion when operations mix RM input/output with tile-based internal computation.

**Fix for agents**:
- **ttnn-generic-op-builder**: When the operation design specifies tile-sized CBs but RM input/output, always use `ttnn.tile_size(dtype)` for tile CB page sizes. Add a validation rule: if `input_tensor.layout == ROW_MAJOR_LAYOUT`, do not use `input_tensor.buffer_page_size()` for tile-based CB page sizes. This is now documented in the program descriptor's comment (line 69) but should be a builder-level invariant check.
- **Pipeline infrastructure**: This matches known issue #9 (no validation between architect output and builder output). A static cross-check could verify CB page sizes against the architect's specification.

### Issue 4: Softmax Analyzer Started Late Relative to Tilize/Untilize

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 1 -- Analysis |
| Agent | orchestrator (scheduling) |
| Retries Consumed | 0 |
| Time Cost | ~7 min of serialization waste |

**Problem**: The tilize and untilize analyzers started at 11:54:13 and 11:54:42 respectively. The softmax analyzer did not start until 12:01:32 -- a 7-minute gap. Since all three are supposed to run in parallel, the softmax analyzer was delayed, though it was actually the fastest once started (~5.5 min).

**Root Cause**: Likely the orchestrator launched softmax analysis after the tilize and untilize analyzers were already partly done, possibly due to sequential agent spawning or a dependency on discovering which reference to use for the compute_core role.

**Fix for agents**:
- **Orchestrator**: Ensure all parallel analyzers are spawned simultaneously. If the compute_core reference requires an additional discovery step (e.g., finding softmax in tt-train after the main TTNN softmax was removed), pre-resolve the reference path before entering the analysis phase.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~6 min (12:28 - 12:34) | 0 free, 0 hard | PASS | Clean -- includes reader/compute/writer implementation + 1 upstream fix (bf16_tile_size) |
| reduce_mean_sub | ~2 min (12:35 - 12:36) | 0 free, 0 hard | PASS | Clean -- compute-only changes, very fast |
| variance_normalize | ~3 min (12:37 - 12:39) | 0 free, 0 hard | PASS | Clean -- added 4 phases, all first-pass |
| affine_transform | ~15 min (12:39 - 12:54) | 0 free, 1 hard | PASS | Stage 4 required reader + compute + entry point + program descriptor changes; 1 failure on tilize volume constraint |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Affine transform implementation | ttnn-kernel-writer-tdd | ~15 min | 24% | Largest TDD stage: reader gamma/beta reads, compute phases 8-9, entry point tilize logic, program descriptor TensorAccessorArgs | 1 hard | Complex multi-file change + tilize volume constraint discovery |
| 2 | Analysis output volume | 3x ttnn-operation-analyzer | ~13 min | 21% | 87KB total analysis for a single operation | 0 | Analysis depth exceeds what the architect actually needs (architect read each in seconds) |
| 3 | Builder compile path fixes | ttnn-generic-op-builder | ~3 min | 5% | Two free retries fixing include paths | 2 free | Stale include paths in templates |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-operation-analyzer | 87KB of analysis produced | The architect consumed key findings in seconds (breadcrumbs show ~2 sec per analysis read). Most analysis detail was never referenced during design. | Reduce analysis scope: the architect needs CB sizing patterns, helper call signatures, and data flow structure -- not full line-by-line code walkthroughs |
| ttnn-generic-op-builder | First two test runs with incorrect includes | Stale template paths required empirical discovery | Fix template include paths permanently |
| ttnn-kernel-writer-tdd | Initial gamma/beta tilize attempt (first attempt, stage 4) | ttnn.tilize() fails on sub-tile tensors | Architect should specify the repeat+tilize pattern for sub-tile parameters |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: orchestrator -> ttnn-operation-analyzer

| Field | Value |
|-------|-------|
| Artifact Passed | Role-based focus directives (input_stage, output_stage, compute_core) |
| Quality | GOOD |
| Issues | Softmax analyzer started ~7 min late; softmax reference from tt-train rather than main TTNN (removed in recent commits) |
| Downstream Impact | Minor -- analysis phase was slightly longer than necessary |
| Suggestion | Pre-resolve all reference file paths before spawning analyzers |

### Handoff 2: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md (20KB), untilize_analysis.md (24KB), softmax_analysis.md (43KB) |
| Quality | ADEQUATE |
| Issues | Analysis output is voluminous (87KB total). The architect read each file's key findings in ~2 seconds per breadcrumb event. The tilize analysis missed the `ttnn.tilize()` minimum volume constraint (TILE_HW=1024), which caused a downstream failure in stage 4. |
| Downstream Impact | The architect's gamma/beta handling recommendation was incomplete because the tilize analysis did not cover edge cases for sub-tile tensors. |
| Suggestion | (1) Reduce analysis output to structured key-findings format (5-10KB max per analysis). (2) Include API constraints/edge cases section in every analysis (minimum shapes, alignment requirements, unsupported modes). |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md (19KB) + .tdd_state.json (5.6KB) |
| Quality | GOOD |
| Issues | The design specified `tile_size(DataFormat::Float16_b)` for page sizes (correctly), but the builder still used `buffer_page_size()` in the program descriptor. This suggests the builder did not fully internalize the RM-vs-tile page size distinction from the design. |
| Downstream Impact | Kernel writer had to fix the program descriptor (bf16_tile_size). Low cost but unnecessary. |
| Suggestion | Add an explicit "Page Size Source" column to the CB table in op_design.md: "ttnn.tile_size(bfloat16)" vs "input_tensor.buffer_page_size()". Make the builder validate its CB page sizes against the architect's specification. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files, execution log with handoff notes |
| Quality | GOOD |
| Issues | The builder's handoff notes correctly warned that "binary_op_helpers.hpp exists in source but NOT in build install dir." The kernel writer had no issues with this (helpers resolved correctly at compile time). The builder also correctly documented verified include paths. |
| Downstream Impact | Minimal -- the kernel writer was well-prepared |
| Suggestion | The builder's handoff notes format is effective. Maintain this practice. |

### Handoff 5: ttnn-operation-architect -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 (kernel implementation details) |
| Quality | GOOD |
| Issues | The architect's gamma/beta handling section (Critical Notes item 4) presented two alternatives without fully resolving which one to use: "host converts to TILE before kernel launch (simpler, recommended for initial impl)" -- but this recommendation was not validated against ttnn.tilize() constraints. The kernel writer chose this path and hit the volume constraint. |
| Downstream Impact | 1 hard attempt consumed, ~5 minutes of debugging |
| Suggestion | When the architect recommends a specific approach, that approach should be validated end-to-end, including host-side API constraints. Alternatively, mark unvalidated alternatives clearly as "NOT TESTED -- verify before using." |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder (self) | "System prompt include paths for kernel helpers were incorrect for this codebase version. Used empirically verified paths." Fix template paths. | H | M |
| ttnn-generic-op-builder | ttnn-kernel-writer-tdd | "replaced input_tensor.buffer_page_size() with ttnn.tile_size(ttnn.bfloat16)" -- builder should use ttnn.tile_size() for tile CBs when input is RM | H | H |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Gamma/beta tilize requires repeat to H=32 first. Architect should specify this step when recommending host-side tilize for sub-tile parameters. | H | M |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Analysis | 87KB of analysis output; architect reads key findings in seconds | Compress analysis to structured format: CB patterns, helper signatures, constraints/edge cases. Target 5-10KB per analysis. | M |
| Build | bf16_tile_size bug passes silently from builder to kernel writer | Add static cross-check: verify CB page sizes match architect's specification before kernel phase | M |
| Design | Architect recommends approaches without full API constraint validation | Architect should validate recommended APIs (e.g., ttnn.tilize minimum volume) before committing to a design approach | H |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | No numerical debugging needed -- all correctness issues were structural (tilize volume, include paths, page sizes). This is notable: the operation completed without any numerical mismatch issues. |
| 2 | Too many planning stages | NO (DONE) | Merged architect worked well. Single op_design.md was sufficient. |
| 3 | .tdd_state.json fragility | NO | State file worked correctly throughout. All 4 stages tracked properly. |
| 4 | No fast path for simple operations | NO | Layer norm is a complex operation (10 compute phases, 13 CBs). Fast path would not apply. |
| 6 | Builder runs on Sonnet | PARTIALLY | Builder had 2 free retries on include paths and the bf16_tile_size bug. These are exactly the kinds of detail-sensitivity issues flagged for Sonnet. The bf16_tile_size bug in particular (using buffer_page_size() instead of tile_size()) is a semantic error that a stronger model might catch from the design doc. |
| 7 | Discovery keyword matching | UNCLEAR | Cannot assess from available evidence -- no discovery phase breadcrumbs. |
| 9 | No architect/builder cross-validation | YES | The builder's bf16_tile_size bug would have been caught by a static cross-check between the architect's CB table (specifying `tile_size(DataFormat::Float16_b)`) and the builder's program descriptor (using `buffer_page_size()`). |
| 11 | No incremental re-run | NO | Pipeline completed fully; no manual intervention needed. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect does not validate host-side API constraints for sub-tile tensors | When the architect recommends ttnn.tilize() for parameter tensors with H<32, the recommendation is silently wrong. The architect should validate that tensors meet minimum shape requirements for recommended APIs. | MEDIUM |
| Analyzer output volume disproportionate to architect consumption | 87KB of analysis for 3 references, consumed by architect in ~6 seconds total (3 references x ~2 sec each). Analysis provides depth that is never used. Structured key-findings format would be more efficient. | LOW |
| Builder CB page size confusion for RM-input tile-compute operations | When an operation has RM input/output but tile-based internal compute, the builder confuses buffer_page_size() (which returns stick_size for RM tensors) with tile_size(). This is a recurring pattern for hybrid RM/tile operations. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Validate API Constraints in Architect Design for Parameter Tensors

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add a validation step: when the architect recommends host-side API calls (ttnn.tilize, ttnn.pad, ttnn.repeat) for parameter tensors, the design must specify the full pipeline including any required shape adjustments. For ttnn.tilize specifically, note that H and W must be >= 32 and volume must be >= 1024. When parameters have H=1 (common for gamma/beta/bias), the design must include the repeat/pad step.
- **Expected Benefit**: Prevents hard attempts in the kernel writer caused by unvalidated API recommendations. Would have saved 1 hard attempt and ~5 minutes in this run.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Fix Builder Template Include Paths

- **Type**: tool_improvement
- **Target**: `.claude/references/generic_op_template/` kernel stub templates
- **Change**: Update stub kernel templates to use verified include paths: `#include "api/dataflow/dataflow_api.h"` for reader/writer, `#include "api/compute/common.h"` for compute. Remove references to `tensor_accessor.hpp` and `compute_kernel_api.h`.
- **Expected Benefit**: Eliminates 2 free retries per pipeline run. Would have saved ~3 minutes in this run.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Add Builder Invariant Check for tile_size vs buffer_page_size

- **Type**: new_validation
- **Target**: ttnn-generic-op-builder agent prompt
- **Change**: Add explicit instruction: "When creating CB page sizes for tile-based CBs in an operation with ROW_MAJOR input, ALWAYS use `ttnn.tile_size(dtype)`, NEVER use `input_tensor.buffer_page_size()`. The latter returns the RM stick size, not the tile size."
- **Expected Benefit**: Prevents the buffer_page_size/tile_size confusion for all hybrid RM/tile operations. Would have saved the kernel writer ~2 minutes and one upstream fix.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 4: Compress Analyzer Output to Structured Key-Findings Format

- **Type**: instruction_change
- **Target**: ttnn-operation-analyzer agent prompt
- **Change**: Cap analysis output at ~10KB per reference. Use structured format with sections: (1) Data Flow Pattern (2) CB Layout (3) Helper Call Signatures (4) API Constraints/Edge Cases (5) Key Code Patterns. Eliminate line-by-line code commentary.
- **Expected Benefit**: Reduces architect context consumption, speeds up design phase. The current 87KB is consumed in ~6 seconds -- most content is never referenced.
- **Priority**: LOW
- **Effort**: MEDIUM

### Recommendation 5: Add Static CB Cross-Check Between Architect and Builder

- **Type**: pipeline_change
- **Target**: Pipeline orchestration (between Phase 3 and Phase 4)
- **Change**: After the builder produces the program descriptor, automatically extract CB configurations and compare against the architect's CB table in op_design.md. Flag any mismatches in page size, page count, or data format before the kernel writer starts.
- **Expected Benefit**: Catches builder bugs (like buffer_page_size vs tile_size) before they reach the kernel writer. Implements known issue #9.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4 | Correct references identified (tilize, untilize, softmax). Softmax from tt-train was a valid adaptation to removed main TTNN softmax. |
| Analysis quality | 3 | Thorough but over-verbose (87KB). Missed tilize API constraint that caused downstream failure. |
| Design completeness | 4 | Excellent CB layout, phase-by-phase state tracking, helper validation. One gap: unvalidated host-side API recommendation for gamma/beta tilize. |
| Build correctness | 3 | Infrastructure correct, but had 2 include path errors (templates) and 1 semantic error (buffer_page_size vs tile_size). All recoverable but unnecessary. |
| Kernel implementation | 5 | Outstanding: 3/4 stages first-pass, clean code, autonomous upstream fixes, rapid stage progression. |
| Inter-agent communication | 4 | Design doc was detailed enough for first-pass kernel implementation. Builder handoff notes were helpful. One gap in architect->kernel-writer for gamma/beta handling. |
| Logging/observability | 4 | Breadcrumbs present for all agents with timestamps, action types, and detailed context. Builder execution log was well-structured. One gap: kernel writer breadcrumbs file was relatively sparse (24 lines for 25 min of work); the .breadcrumbs.md file (84 lines) was more detailed. Standardizing on a single breadcrumb format would improve analysis. |

### Top 3 Things to Fix

1. **Architect must validate host-side API constraints** -- the ttnn.tilize() volume constraint on gamma/beta caused the only hard attempt in this run. This is a class of bug that will recur whenever sub-tile parameter tensors need tilizing.
2. **Builder must use ttnn.tile_size() for tile CBs in RM operations** -- this buffer_page_size/tile_size confusion is a semantic trap for hybrid RM/tile operations. Add it as an explicit instruction and/or static validation.
3. **Fix builder template include paths** -- two free retries per run is a low-cost but unnecessary friction that should be eliminated from templates.

### What Worked Best

The architect's design quality was the strongest aspect of this pipeline run. The op_design.md produced by the `ttnn-operation-architect` enabled 3 out of 4 TDD stages to pass on the first attempt with zero retries -- a 75% first-pass rate. The per-phase CB state tables, explicit NoWaitNoPop annotations, and binary op broadcast verification table gave the kernel writer enough precision to translate the design directly to working code. The kernel writer's total productive coding time for stages 1-3 was only ~11 minutes (12:28 to 12:39), averaging under 4 minutes per stage including test execution. This demonstrates that when the design is right, kernel implementation can be extremely efficient.
