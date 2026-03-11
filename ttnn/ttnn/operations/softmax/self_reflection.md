# Self-Reflection: softmax

## Metadata
| Field | Value |
|-------|-------|
| Operation | `softmax` |
| Operation Path | `ttnn/ttnn/operations/softmax` |
| Pipeline Phases Executed | Phase 0 (Discovery), Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels), Phase 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x2), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 11 (2 analyzer, 2 architect, 2 builder, 4 kernel-writer, 1 report) |
| Total Pipeline Duration | ~79 minutes (17:05 to 18:24 UTC) |
| Overall Result | SUCCESS -- all 5/5 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~5m (est.) | PASS | Selected reduce_w and reduce_h as reference operations. Good choices for the two reduction dimensions. |
| 1: Analysis | ttnn-operation-analyzer (x2) | ~13m | PASS | Two analyzers ran in parallel. reduce_h analysis completed first (17:15), reduce_w analysis completed at 17:17. |
| 2: Design | ttnn-operation-architect | ~8m | PASS | Produced comprehensive op_design.md with 4-phase compute strategy for both dim=-1 and dim=-2. One self-correction during design (SCALAR -> COL broadcast). |
| 3: Build | ttnn-generic-op-builder | ~15m | PASS | Created 12 files. Hit one kernel compilation error (tensor_accessor.hpp include path). Also needed build_metal.sh run. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~33m | PASS | 5 stages completed. 3/5 first-attempt pass. 1 stage had 2 failures (1 compilation + 1 numerical). Context compacted once between stages 4 and 5. |
| 5: Report | orchestrator | ~2m | PASS | REPORT.md generated. |

### Agent Duration Breakdown

Timing derived from breadcrumb `"event":"start"` and `"event":"complete"` timestamps. Git commit timestamps used as secondary validation.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (reduce_h) | 17:05:02 | 17:15:34 | 10m 32s | 0 | ~10m active (all productive reading + writing) |
| ttnn-operation-analyzer (reduce_w) | 17:05:07 | 17:17:40 | 12m 33s | 0 | ~12m active (all productive reading + writing) |
| ttnn-operation-architect | 17:20:01 | 17:28:05 | 8m 4s | 0 | ~7m design, ~1m git (pre-commit hook re-stage) |
| ttnn-generic-op-builder | 17:31:38 | 17:46:23 | 14m 45s | 1 | ~5m coding, ~4m build_metal.sh, ~3m debugging include path, ~2m testing |
| ttnn-kernel-writer-tdd | 17:48:08 | 18:21:11 | 33m 3s | 2 | ~25m productive coding, ~8m debugging (stage 3 numerical) |

**Duration calculation method**: Primary source is breadcrumb timestamps. All agents had both `start` and `complete` events. Git commits cross-validated and consistent (within 1-2 minutes of breadcrumb timestamps, accounting for commit overhead).

### Duration Visualization

```
Phase 0  |###|                                                        (~5m est.)
Phase 1  |##########|                                                 (~13m) 2 analyzers in parallel
Phase 2            |######|                                           (~8m)
Phase 3                   |###########|                               (~15m) includes build_metal.sh
Phase 4                                |########################|     (~33m) 5 TDD stages
Phase 5                                                         |#|   (~2m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80 min

Longest phase: Phase 4 (33m) -- kernel implementation across 5 stages, including 1 context compaction
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~18m | 23% | 2 analyzers ran in parallel; wall time limited by the slower analyzer |
| Design (Phase 2) | ~8m | 10% | Clean execution, one pre-commit hook retry |
| Build (Phase 3) | ~15m | 19% | Includes ~4m for build_metal.sh first-time build |
| Kernel implementation (Phase 4) | ~33m | 42% | 5 TDD stages |
| -- Productive coding | ~25m | 32% | Writing kernel code that passed |
| -- Debugging/retries | ~8m | 10% | Stage 3 compilation error + numerical mismatch |
| Reporting (Phase 5) | ~2m | 3% | Clean |
| Overhead (gaps between phases) | ~3m | 4% | Agent startup, orchestrator handoffs |
| **Total** | **~79m** | **100%** | |

---

## 2. What Went Well

### 1. TDD Stage Design Enabled Efficient Debugging

**Phase/Agent**: Phase 2 (architect) and Phase 4 (kernel-writer)
**Evidence**: The 5-stage progression (data_pipeline -> exp -> unstable_softmax -> stable_softmax -> height_dim) allowed the numerical mismatch in stage 3 to be isolated to the reduce+multiply phase. Stages 1-2 had already validated data movement and exp correctness, so the kernel writer could focus on the reduce helper parameters without questioning the pipeline.
**Why it worked**: The architect designed stages that cleanly separate concerns. Each stage adds exactly one layer of complexity, making it clear where bugs originate.

### 2. Reference Analysis Quality Was High

**Phase/Agent**: Phase 1 (analyzers)
**Evidence**: Both analysis documents were comprehensive (472 lines for reduce_w, 556 lines for reduce_h). The architect explicitly cited them in breadcrumbs: `"key_findings":"REDUCE_ROW pattern, reduce helper lib, WaitAndPopPerTile policy"` and `"key_findings":"REDUCE_COL pattern, chunked column processing, DEST_AUTO_LIMIT chunking"`. The kernel writer never had to re-read the original reference C++ code -- the analysis documents were sufficient.
**Why it worked**: Running two focused analyzers in parallel, each with a single reference file, produced well-scoped, detailed output. No cross-contamination between the two analysis paths.

### 3. Zero Device Hangs

**Phase/Agent**: Phase 3-4
**Evidence**: All 7 test runs across the pipeline (1 builder integration + 5 TDD stages + 1 failed compilation that did not reach device) completed without hangs. This is notable for an operation with complex multi-phase compute kernels that manage 6 circular buffers.
**Why it worked**: The CB push/pop balance was tracked explicitly at each stage (breadcrumb `cb_sync_check` events), and the architect's design included detailed CB state tables after each phase.

### 4. dim=-2 (Height Softmax) Passed First Attempt

**Phase/Agent**: Phase 4, stage softmax_stable_h
**Evidence**: Stage 5 involved writing entirely new kernels (reader_h.cpp, compute_h.cpp, writer_h.cpp) plus modifying the program descriptor. All 4 test shapes passed on the first attempt. This is the most complex stage, requiring chunked column reads, REDUCE_COL, ROW broadcast, and a custom writer that maps column-interleaved output back to row-major DRAM.
**Why it worked**: The architect's design doc was detailed enough for the dim=-2 path (including ReduceInputMemoryLayout::with_row_stride, BinaryInputBlockShape::of(Ht, current_chunk), and ROW broadcast). The kernel writer also correctly identified that a separate writer_h.cpp was needed (the design originally specified a shared writer, but the kernel writer recognized the output tile ordering difference and created a dedicated writer).

### 5. Low Retry Budget Consumption

**Phase/Agent**: Phase 4
**Evidence**: Only 2 out of 30 total budget attempts were consumed. Of those, 1 was a free retry (compilation error) and 1 was a hard attempt (numerical mismatch). The hard attempt was resolved in a single hypothesis-fix cycle with HIGH confidence.
**Why it worked**: The design doc gave the kernel writer a clear starting point. Most stages required only translating the design's pseudocode into actual C++ using the helper library APIs.

---

## 3. Issues Found

### Issue 1: Architect Specified SCALAR Broadcast, Should Have Been COL

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- softmax_unstable_w (stage 3) |
| Agent | ttnn-operation-architect (root cause), ttnn-kernel-writer-tdd (discovered) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~8 minutes (18:00:06 test fail -> 18:07:47 test pass) |

**Problem**: The op_design.md specified `BroadcastDim::SCALAR` for the Phase 4 multiply operation in the dim=-1 compute kernel. The design's reasoning stated: "Since we process one row at a time (Ht=1), REDUCE_ROW output is a single tile. Broadcast is SCALAR (1x1 tile applied to 1xWt)." However, REDUCE_ROW output is not a 1x1 scalar -- it is a tile with valid data in column 0, with each of the 32 rows containing the reduction result for that row within the tile. SCALAR broadcast would replicate a single value (row 0, col 0) across all positions, but the correct broadcast is COL, which replicates each row's col-0 value across the row's columns.

The numerical mismatch manifested as max diff 0.29. The kernel writer's hypothesis (H2) correctly identified the issue: "Output diverges per-row: first row matches (same reciprocal applied) but later rows use wrong normalization factor." Confidence was HIGH.

**Root Cause**: The architect confused "single tile output" with "scalar output." A REDUCE_ROW operation produces a tile where each row independently holds the reduction result for that row. The valid region is Col0, not (0,0). The architect's own "Binary Op Broadcast Verification" table in op_design.md labels the valid region as "Col0 (REDUCE_ROW max)" but then concludes "SCALAR" broadcast, which is contradictory.

The architect did make a self-correction during design (recorded in the execution log under "Architecture Revisions (Pass 2 corrections)"): "dim=-1 broadcast type: COL (for reduce_row output) -> SCALAR". This "correction" was actually wrong -- the original COL was correct, and the architect second-guessed it.

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "If the valid region of a CB is Col0 (column vector), the broadcast must be COL, not SCALAR. SCALAR is only correct when the valid region is a single element at (0,0)." This should be a checklist item in the Binary Op Broadcast Verification section.
- **ttnn-operation-architect**: When processing "one row at a time" (Ht=1 per iteration), REDUCE_ROW still produces a Col0-shaped result within a 32x32 tile. The confusion arises from conflating "1 tile output" with "scalar." Instructions should explicitly state: "The reduction output tile always has 32 rows of independent results; Ht=1 per iteration does NOT make the output a scalar."

### Issue 2: Test Reference Functions Had Wrong Variable Names

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2/3 (architect/builder generated tests), discovered in Phase 4 stage 1 |
| Agent | ttnn-operation-architect (via tdd_orchestrator) |
| Retries Consumed | 0 (fixed by kernel writer as part of stage 1 implementation) |
| Time Cost | ~2 minutes (kernel writer fixed all 5 tests before running stage 1) |

**Problem**: All 5 TDD stage test files had broken `pytorch_reference` functions. The breadcrumb from the kernel writer states: `"critical_issue":"All 5 test files have broken pytorch_reference: missing return, using input instead of input_tensor"`. The builder's execution log also flagged this: `"Stage test reference bodies have syntax issues: input instead of return input_tensor in data_pipeline_w"`.

**Root Cause**: The tdd_orchestrator template used `input` as the variable name in the reference body, but the generated test function parameter is named `input_tensor`. The `reference_body` field in `.tdd_state.json` contains raw expressions like `"input"` (for data_pipeline_w) and `"torch.nn.functional.softmax(input, dim=-1)"` which do not match the test function's parameter name.

**Fix for agents**:
- **tdd_orchestrator/template**: The reference body template should use `input_tensor` as the variable name, or the generator should perform a substitution from `input` to `input_tensor`. A lint step could catch undefined variable references in generated test files.
- **ttnn-operation-architect**: When specifying `reference_body` in TDD stages, always use the variable name `input_tensor` (which matches the generated test function signature), not `input`.

### Issue 3: Builder Hit Invalid tensor_accessor.hpp Include Path

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 (kernel compilation error on first test run) |
| Time Cost | ~3 minutes (17:39:31 build complete -> 17:42:43 fix applied) |

**Problem**: The builder included `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` in all dataflow kernel stubs. This path does not exist in the kernel compilation environment. TensorAccessor is auto-included by `dataflow_api.h`.

**Root Cause**: The builder's instruction set contains a helper-to-include mapping table that lists this incorrect path. The builder's execution log (Section 7) specifically calls this out as an instruction improvement recommendation with HIGH confidence.

**Fix for agents**:
- **ttnn-generic-op-builder instructions**: Remove `tensor_accessor.hpp` from the helper-to-include mapping table. Add a note that TensorAccessor is auto-included via `dataflow_api.h` and needs no explicit include.

### Issue 4: Builder Needed to Run build_metal.sh Before Testing

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (expected step, not a failure) |
| Time Cost | ~4 minutes (build_metal.sh execution) |

**Problem**: The builder attempted to run tests before the C++ project was built, encountering `ModuleNotFoundError: No module named 'ttnn._ttnn'`. It then ran `./build_metal.sh` which took approximately 4 minutes.

**Root Cause**: In a fresh worktree clone, the native C++ module has not been compiled. The builder should check for this before attempting tests, or the pipeline orchestrator should ensure the build is done before Phase 3 begins.

**Fix for agents**:
- **Pipeline orchestrator**: Run `build_metal.sh` (or verify it has been run) before launching the builder. This avoids wasting time on a predictable failure.
- **ttnn-generic-op-builder**: Add an early check for `ttnn._ttnn` importability before attempting tests.

### Issue 5: Design Specified Shared Writer for dim=-2, but Kernel Writer Created Separate Writer

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- softmax_stable_h (stage 5) |
| Agent | ttnn-operation-architect (design), ttnn-kernel-writer-tdd (implementation) |
| Retries Consumed | 0 |
| Time Cost | ~3 minutes (kernel writer created writer_h.cpp and updated program descriptor) |

**Problem**: The op_design.md states under "Writer Kernel (shared: writer.cpp)": "Generic tile writer... Same kernel for both dim=-1 and dim=-2." It also notes: "output tiles are produced in chunked column order... the writer maps back to correct output tile positions... the writer can use the same sequential start_id-based indexing as the reader." However, the kernel writer correctly identified that sequential start_id indexing does NOT work for chunked column order -- the output tiles need to be written to non-sequential DRAM positions. The kernel writer created `softmax_writer_h.cpp` and updated the program descriptor (breadcrumb: `upstream_fix: Changed dim=-2 writer from softmax_writer.cpp to softmax_writer_h.cpp`).

**Root Cause**: The architect's design made an incorrect simplification. Sequential tile ID indexing only works when output tiles are produced in row-major order. For dim=-2, tiles emerge in chunked column order, requiring column-to-row-major address mapping in the writer. The architect's note about "output tile IDs = input tile IDs" is technically true (softmax preserves shape), but the ORDER in which tiles are produced by the compute kernel does not match sequential tile ID order.

**Fix for agents**:
- **ttnn-operation-architect**: When designing operations where the compute kernel processes tiles in non-row-major order (e.g., chunked column order for height reduction), explicitly flag that the writer must reverse the reordering. A shared writer using sequential IDs is only valid when the compute kernel produces tiles in row-major order.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline_w | ~4m (17:49-17:52) | 0 free, 0 hard | PASS | Clean. Also fixed 5 test files (upstream bug). |
| exp_w | ~4m (17:52-17:56) | 0 free, 0 hard | PASS | Clean. Minimal delta from stage 1. |
| softmax_unstable_w | ~12m (17:56-18:08) | 1 free, 1 hard | PASS | Compilation error (NoAccumulation scope) + numerical mismatch (SCALAR vs COL broadcast). |
| softmax_stable_w | ~2m (18:08-18:10) | 0 free, 0 hard | PASS | Clean. Reused COL broadcast fix from stage 3. |
| softmax_stable_h | ~8m (18:13-18:21) | 0 free, 0 hard | PASS | Most complex stage. 3 new kernel files + descriptor changes. First attempt pass. |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | softmax_unstable_w debugging | kernel-writer | ~8m | 10% | Numerical mismatch from wrong broadcast type | 2 | Architect design error (SCALAR vs COL) |
| 2 | build_metal.sh | builder | ~4m | 5% | First-time C++ build in fresh worktree | 0 | Expected in fresh environment |
| 3 | Context compaction | kernel-writer | ~3m | 4% | Context compaction pause between stages 4 and 5 | 0 | Long session (33m) hit context window |
| 4 | tensor_accessor include fix | builder | ~3m | 4% | Invalid include path in kernel stubs | 1 | Incorrect instruction mapping table |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| architect | Broadcast verification table computed, then overridden with wrong "correction" | The Pass 2 revision changed COL to SCALAR incorrectly | Add validation rule: Col0 valid region -> COL broadcast, never SCALAR |
| builder | First test attempt without build | Predictable failure in fresh worktree | Check for native module availability or pre-build in orchestrator |
| kernel-writer | First softmax_unstable_w implementation used SCALAR (per design) | Design was wrong, required investigation | Better design validation at architect phase |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: Analyzers -> Architect

| Field | Value |
|-------|-------|
| Artifact Passed | `reduce_w_analysis.md`, `reduce_h_analysis.md` |
| Quality | GOOD |
| Issues | None identified. Both documents were comprehensive and well-structured. |
| Downstream Impact | Positive. Architect cited specific findings from both analyses in design decisions. |
| Suggestion | No changes needed. The parallel analyzer pattern worked well here. |

### Handoff 2: Architect -> Builder

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md`, `.tdd_state.json` |
| Quality | ADEQUATE |
| Issues | (1) Test reference bodies used wrong variable name (`input` vs `input_tensor`). (2) The SCALAR broadcast error, while not directly affecting the builder, propagated to the kernel writer via the design doc. |
| Downstream Impact | Builder flagged the test reference issue in upstream_feedback. Kernel writer had to fix 5 test files before stage 1. Not a significant time cost (~2m) but adds friction. |
| Suggestion | The tdd_orchestrator should validate that `reference_body` expressions use the correct variable names from the generated test function signature. |

### Handoff 3: Builder -> Kernel Writer

| Field | Value |
|-------|-------|
| Artifact Passed | Python infrastructure (softmax.py, softmax_program_descriptor.py), 5 kernel stubs, test files |
| Quality | GOOD |
| Issues | (1) Kernel stubs were empty (expected). (2) Builder's handoff notes correctly warned about tensor_accessor include removal and broken test references. (3) Builder's note about `kernel_lib` include paths being valid was helpful. |
| Downstream Impact | Kernel writer was well-informed about the infrastructure. No surprises. |
| Suggestion | The builder's handoff notes were effective here. Continue requiring explicit handoff documentation. |

### Handoff 4: Architect Design -> Kernel Writer Implementation

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md` Part 2 (Kernel Implementation) |
| Quality | ADEQUATE |
| Issues | (1) SCALAR vs COL broadcast error cost 1 hard attempt. (2) Shared writer assumption for dim=-2 was incorrect. Kernel writer had to create writer_h.cpp. (3) The design used `NoAccumulation{}` without namespace qualification, which the kernel writer initially copied verbatim causing a compilation error. |
| Downstream Impact | ~11 minutes of extra work in Phase 4 (8m broadcast debugging + 3m writer_h creation). |
| Suggestion | (a) Add explicit broadcast validation rules to the architect's checklist. (b) All code snippets in op_design.md should use fully-qualified namespace identifiers to prevent compilation errors. (c) When the compute kernel produces tiles in non-row-major order, explicitly call out the need for a dedicated writer. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder | Remove tensor_accessor.hpp from helper-to-include mapping; TensorAccessor is auto-included via dataflow_api.h | HIGH | MEDIUM |
| tdd_orchestrator template | ttnn-generic-op-builder | Fix reference_body variable name: use `input_tensor` not `input` | MEDIUM | LOW |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Add broadcast validation rule: Col0 valid region must use COL broadcast, not SCALAR | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Use fully-qualified namespace identifiers in all code snippets (e.g., `compute_kernel_lib::NoAccumulation{}` not bare `NoAccumulation{}`) | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Flag non-row-major output ordering explicitly when recommending shared writer | MEDIUM | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Build | Builder wasted ~4m on build_metal.sh in fresh worktree | Pre-build in orchestrator before launching builder, or verify build status as first step | LOW |
| Design validation | Architect self-corrected a correct COL broadcast to incorrect SCALAR | Add automated cross-validation between CB valid regions and broadcast types in the design verification step | HIGH |
| Logging | No execution log from kernel writer (only breadcrumbs) | Ensure kernel writer produces an execution_log.md with recovery table and handoff notes for self-reflection | MEDIUM |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | PARTIALLY | Stage 3 had a numerical mismatch but it was resolved quickly (~8m, 1 hypothesis with HIGH confidence). The kernel writer did not enter an extended debugging spiral. The COL vs SCALAR issue had a clear diagnostic signature ("first row matches, later rows diverge"). |
| 2 | Long leash (planner/designer gap) | DONE (not applicable) | Pipeline correctly used merged Architect agent. |
| 3 | `.tdd_state.json` fragility | NO | No issues with TDD state format or stage registration. |
| 4 | No fast path for simple operations | NO | Softmax is a complex operation (multi-phase, two dimension paths). Full pipeline was appropriate. |
| 6 | Builder runs on Sonnet | PARTIALLY | Builder hit include path error and needed build step, but completed successfully. The issue was in instruction quality, not model capability. |
| 7 | Discovery keyword matching | NO | Reference selection was appropriate (reduce_w, reduce_h). |
| 9 | No architect/builder cross-validation | YES | The broadcast type error in the architect's design was not caught before Phase 4. If a cross-validator compared the design's broadcast verification table (which listed "Col0" valid region) against the specified broadcast type (SCALAR), it could have caught the inconsistency before any kernel code was written. |
| 11 | No incremental re-run | NO | Pipeline completed successfully without needing re-runs. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect self-correction reverses correct decisions | The architect's Pass 2 revision changed a correct COL broadcast to incorrect SCALAR. The self-review step needs guardrails to prevent "corrections" that introduce errors. | HIGH |
| Design code snippets lack namespace qualification | Unqualified identifiers in op_design.md (e.g., `NoAccumulation{}`) cause predictable compilation errors when the kernel writer copies them verbatim. | MEDIUM |
| Non-row-major output ordering requires explicit writer flagging | When compute kernels produce tiles in non-sequential order, the design should explicitly flag that a shared generic writer is insufficient. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Add Broadcast Type Validation Rule to Architect Instructions

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions (broadcast verification section)
- **Change**: Add mandatory validation rule: "If the valid region of a reduce output is Col0 (column vector), the broadcast type MUST be COL. If the valid region is Row0 (row vector), the broadcast type MUST be ROW. SCALAR is only correct when the valid region is a single element at position (0,0). A single output tile does NOT imply SCALAR -- the tile still has 32 independent rows and columns."
- **Expected Benefit**: Prevents the exact error that caused 1 hard attempt and ~8 minutes of debugging in this run. This is a systematic confusion that will recur in any operation using reduce + broadcast.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Require Fully-Qualified Identifiers in Design Code Snippets

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions (kernel implementation section)
- **Change**: Add rule: "All identifiers in code snippets must use fully-qualified namespaces (e.g., `compute_kernel_lib::NoAccumulation{}`, not `NoAccumulation{}`). The kernel writer will copy these snippets, and unqualified names cause compilation errors."
- **Expected Benefit**: Prevents free-retry compilation errors from namespace issues. Saves ~2 minutes per occurrence.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Fix TensorAccessor Include Path in Builder Instructions

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent instructions (helper-to-include mapping table)
- **Change**: Remove the entry for `tensor_accessor.hpp`. Add note: "TensorAccessor is automatically available via `#include "api/dataflow/dataflow_api.h"`. No additional include is needed."
- **Expected Benefit**: Prevents 1 compilation error per run. Saves ~3 minutes.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Fix TDD Test Reference Variable Names

- **Type**: tool_improvement
- **Target**: `tdd_orchestrator.py` (test file generation template)
- **Change**: Replace `input` with `input_tensor` in all `reference_body` expressions, or add a post-generation validation that checks all variable references in the generated `pytorch_reference()` function resolve to function parameters.
- **Expected Benefit**: Eliminates a recurring upstream bug that the kernel writer must fix before starting TDD.
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 5: Add Non-Row-Major Output Warning to Architect Instructions

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions (writer kernel design section)
- **Change**: Add rule: "If the compute kernel processes tiles in non-row-major order (e.g., chunked column order for height reduction, transposed order), the writer kernel MUST handle address remapping. Do NOT recommend a shared generic writer for such cases -- design a dedicated writer or explicitly document the tile-ID-to-DRAM-address mapping the writer must implement."
- **Expected Benefit**: Prevents the kernel writer from needing to diverge from the design and create ad-hoc writer kernels.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 6: Pre-Build in Fresh Worktrees

- **Type**: pipeline_change
- **Target**: Pipeline orchestrator / worktree setup script
- **Change**: After creating a worktree, run `build_metal.sh` before launching any agents. This ensures the `ttnn._ttnn` native module is available when the builder tests stubs.
- **Expected Benefit**: Saves ~4 minutes per fresh-worktree run by removing a predictable failure from the builder's critical path.
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4 | Correct reference selection (reduce_w, reduce_h). Both were directly relevant to the two softmax dimensions. |
| Analysis quality | 5 | Two comprehensive, well-structured analyses (472 and 556 lines). Architect used both effectively. No re-reads of original C++ needed. |
| Design completeness | 3 | Comprehensive architecture, but the SCALAR vs COL broadcast error was a significant design flaw. The shared writer assumption for dim=-2 was also incorrect. Self-correction during design introduced the broadcast bug rather than fixing it. |
| Build correctness | 4 | All infrastructure worked correctly after fixing the include path. Program descriptor for both dim=-1 and dim=-2 was correct. CB sizes, work distribution, and kernel dispatch all worked first try. |
| Kernel implementation | 4 | 5/5 stages passed. Only 2 attempts consumed out of 30 budget. Stage 5 (most complex) passed first try. Context compaction between stages 4-5 was handled cleanly. |
| Inter-agent communication | 3 | Two issues propagated across handoffs: (1) broadcast type error from architect to kernel writer, (2) test reference variable names from orchestrator to builder to kernel writer. Neither caused pipeline failure, but both added friction. |
| Logging/observability | 4 | Breadcrumbs were present for all agents with timestamps. Execution logs were present for architect and builder. Kernel writer had breadcrumbs but no execution log. All breadcrumbs had sufficient detail for this analysis (including hypothesis IDs, CB sync checks, upstream fix records). |

### Top 3 Things to Fix

1. **Add broadcast type validation rules** to the architect's instructions. The SCALAR vs COL confusion is a systematic issue that will recur in any operation using reduce followed by broadcast. A simple rule (Col0 -> COL, Row0 -> ROW, (0,0) -> SCALAR) would have prevented the only hard failure in this run.

2. **Require fully-qualified namespace identifiers** in all design document code snippets. This prevents compilation errors from being inherited by the kernel writer and saves a free retry per occurrence.

3. **Add architect/builder cross-validation** (known issue #9). The broadcast error was detectable from the design document alone -- the architect's own table listed "Col0 valid region" but specified "SCALAR broadcast." An automated consistency check between valid regions and broadcast types would catch this class of error before Phase 4.

### What Worked Best

The **5-stage TDD progression** was the single strongest aspect of this pipeline run. By building incrementally from data passthrough to exp to unstable softmax to stable softmax to height-dimension softmax, the pipeline isolated the only numerical bug to exactly the stage where reduce and broadcast operations were first introduced (stage 3). Stages 1-2 had already validated the data pipeline and exp correctness, so the kernel writer could immediately focus on the reduce/broadcast parameters. Stages 4-5 then built on the proven stage 3 fix with zero additional failures. This progressive decomposition turned a potentially complex multi-phase softmax implementation into a manageable series of incremental validations, completing the full operation in just 33 minutes of kernel writing time.
