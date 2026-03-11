# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 12 (this run, 2026-03-11) |
| Total Pipeline Duration | ~71 minutes (17:06 - 18:17 UTC) |
| Overall Result | SUCCESS - All 6 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m (pre-17:06) | Completed | Selected tilize, untilize, batch_norm as references; batch_norm chosen over softmax (no standalone factory) |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~12m (17:06 - 17:18) | Completed | 3 parallel analyzers; DeepWiki unavailable for batch_norm analyzer, fell back to local docs |
| 2: Design | ttnn-operation-architect | ~7m (17:20 - 17:27) | Completed | 3-pass architecture designed; 6 TDD stages; 12 CBs; helper library fully validated |
| 3: Build | ttnn-generic-op-builder | ~11m (17:30 - 17:41) | Completed | 1 compilation error (tensor_accessor include path); fixed on retry |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~33m (17:43 - 18:16) | Completed | 6/6 stages passed; 1 hard + 1 free retry total |
| 5: Report | orchestrator | ~2m (18:16 - 18:18) | Completed | REPORT.md generated |

### Agent Duration Breakdown

Duration calculation method: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps as primary source. Git commit timestamps used as cross-check. Breadcrumbs had complete start/complete pairs for all agents.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 17:06:38 | 17:16:06 | ~9m 28s | 0 | ~9m active (research + writing) |
| ttnn-operation-analyzer (untilize) | 17:06:42 | 17:13:59 | ~7m 17s | 0 | ~7m active |
| ttnn-operation-analyzer (batch_norm) | 17:06:50 | 17:18:50 | ~12m 00s | 0 | ~10m active, ~2m DeepWiki failure workaround |
| ttnn-operation-architect | 17:20:12 | 17:27:42 | ~7m 30s | 0 | ~7m productive (reference reads, helper analysis, design writing) |
| ttnn-generic-op-builder | 17:30:14 | 17:41:18 | ~11m 04s | 1 | ~8m productive, ~3m debugging include path |
| ttnn-kernel-writer-tdd | 17:43:04 | 18:15:40 | ~32m 36s | 2 (1 hard, 1 free) | ~28m productive, ~5m debugging |

### Duration Visualization

```
Phase 0  |#|                                                     (~1m)
Phase 1  |############|                                          (~12m) 3 analyzers in parallel
Phase 2        |#######|                                         (~7m)
Phase 3              |###########|                               (~11m)
Phase 4                           |#################################| (~33m)
Phase 5                                                           |##| (~2m)
         0    5    10   15   20   25   30   35   40   45   50 min

Longest phase: Phase 4 (33m) -- 6 TDD stages with compilation and test execution
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~13m | 18% | 3 analyzers ran in parallel; batch_norm took longest |
| Design (Phase 2) | ~7m | 10% | Single architect agent |
| Build (Phase 3) | ~11m | 15% | 1 include-path failure + fix |
| Kernel implementation (Phase 4) | ~33m | 46% | 6 TDD stages |
| -- Productive coding | ~28m | 39% | Writing kernel code that passed |
| -- Debugging/retries | ~5m | 7% | Namespace collision fix, Shape indexing fix |
| Reporting (Phase 5) | ~2m | 3% | |
| Inter-phase gaps | ~5m | 7% | Agent spin-up, orchestrator overhead |
| **Total** | **~71m** | **100%** | |

---

## 2. What Went Well

### 1. Exceptional TDD efficiency: 1 hard + 1 free retry across 6 stages (2.8% budget used)

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows only 2 failures total out of 6 stages. Stages 3-6 (subtract_mean, variance, normalize, affine) all passed on first attempt with zero retries. The kernel writer used only 1 of its 36 available hard attempts (2.8% of budget).
**Why it worked**: The architect's design was detailed enough that the kernel writer could translate it directly to code. The design document included exact helper call signatures with template parameters, CB flow states after each phase, and explicit `cb_wait_front`/`cb_pop_front` lifecycle management. The 13-phase compute kernel implementation matched the design exactly with no deviations.

### 2. Design document quality: All 13 compute phases implemented verbatim from the design

**Phase/Agent**: Phase 2, ttnn-operation-architect
**Evidence**: Comparing `op_design.md` Part 2 (lines 196-415) against `compute_layer_norm_rm.cpp` (lines 77-205): every phase, every helper call, every CB index, every template parameter, every BinaryInputPolicy, and every broadcast dimension matches the design exactly. The `c_pre_untilize` compile-time routing logic (line 39 of compute kernel) matches the design's "Affine routing" note (design line 425). The architect even pre-specified the `cb_wait_front(c_3, 1)` before the main loop (design line 337), which the kernel writer implemented at line 67.
**Why it worked**: The architect validated all helper calls against the actual helper library headers (breadcrumb entries show 5 separate `helper_analysis` events). The design included CB state tables after each phase, making push/pop balance verification trivial for the kernel writer.

### 3. Zero device hangs across the entire pipeline

**Phase/Agent**: Phase 4, all test runs
**Evidence**: All 6 TDD stages ran without any device hang. No timeout interruptions in any breadcrumb. All test runs completed with clean pytest output. REPORT.md explicitly confirms: "No device hangs encountered."
**Why it worked**: The 3-pass architecture with explicit push/pop balance (the kernel writer checked balance at every stage via `cb_sync_check` breadcrumbs) prevented any deadlock conditions. The design's careful treatment of persistent CBs (epsilon waited once before loop, gamma/beta waited once after tilize) avoided the common "double wait" and "premature pop" hang patterns.

### 4. Stages 3-6 all passed on first attempt

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: Breadcrumbs show stages subtract_mean (17:56-18:00), variance (18:01-18:05), normalize (18:05-18:09), and affine (18:10-18:15) each passed on first test run with zero retries. The kernel writer implemented each stage incrementally, adding only the delta from the previous stage as prescribed by the TDD plan.
**Why it worked**: The TDD stage plan was well-structured with clear deltas between stages. Each stage's scope was narrow enough that the kernel writer could implement it correctly in a single pass. The upstream fixes applied in earlier stages (output shape switching, writer CT args) were already in place by the time later stages needed them.

### 5. Clean reference selection: batch_norm over softmax

**Phase/Agent**: Phase 0, orchestrator
**Evidence**: REPORT.md notes: "Selected `batch_norm` over softmax (softmax has no standalone program factory in TTNN)." Git history from prior runs (2026-03-03, 2026-03-05, 2026-03-06) shows those earlier attempts used softmax as the compute_core reference, and the git log shows many more commits per run (indicating more churn). This run's cleaner batch_norm reference yielded a much more efficient Phase 4.
**Why it worked**: Batch_norm directly implements mean/variance/normalization with gamma/beta affine -- exactly the computational pattern needed for layer_norm. The analyzer's 600-line batch_norm analysis provided complete CB routing patterns, affine parameter handling, and epsilon management that the architect could adapt directly.

---

## 3. Issues Found

### Issue 1: Fully qualified namespace requirement for tilize/untilize config enums

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- data_pipeline (Stage 1) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 free retry |
| Time Cost | ~2 minutes |

**Problem**: The compute kernel initially used `using namespace` directives for both `tilize_config` and `untilize_config` namespaces. Both namespaces define identically-named enums (`InitUninitMode::InitAndUninit`, `WaitMode::WaitBlock`, `ReconfigureRegisterDatatypeMode`). The compiler emitted: `error: parse error in template argument list` at line 36 where `InitUninitMode::InitAndUninit` was ambiguous between tilize and untilize variants (breadcrumb `H1` at 17:47:15, confidence HIGH).

**Root Cause**: The kernel writer generated `using namespace compute_kernel_lib::tilize_config;` and `using namespace compute_kernel_lib::untilize_config;` in the same translation unit. The tilize/untilize helper headers both define `InitUninitMode`, `WaitMode`, and `ReconfigureRegisterDatatypeMode` enum classes with identical value names. When both namespaces are imported, the compiler cannot disambiguate.

**Fix for agents**:
- **ttnn-operation-architect**: Add a note to the design doc's "Critical Notes" section: "When using both tilize and untilize helpers in the same kernel, always use fully qualified names (`compute_kernel_lib::tilize_config::InitAndUninit` and `compute_kernel_lib::untilize_config::InitAndUninit`). Do NOT use `using namespace` for either config namespace."
- **ttnn-kernel-writer-tdd**: Add to the kernel writer's instructions: "NEVER use `using namespace` for tilize_config or untilize_config when both helpers are used in the same kernel."

### Issue 2: ttnn.Shape does not support Python slice indexing

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- reduce_mean (Stage 2) |
| Agent | ttnn-kernel-writer-tdd (upstream fix to entry point) |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~3 minutes |

**Problem**: The entry point `layer_norm_rm.py` used `shape[:-1]` to construct the reduced output shape. `ttnn.Shape` does not support Python slice syntax, raising `TypeError: __getitem__(): incompatible function arguments` at runtime (breadcrumb at 17:52:29). The kernel writer had to apply an upstream fix to convert the shape to a Python list before slicing.

**Root Cause**: The builder generated the output shape computation using `input_tensor.shape[:-1]`, which works for Python tuples/lists but not for `ttnn.Shape` objects. The builder was not aware of this API limitation. The kernel writer's upstream fix (breadcrumb at 17:52:47) was: "Convert shape to list before slicing to avoid ttnn.Shape slice incompatibility."

**Fix for agents**:
- **ttnn-generic-op-builder**: Add to the builder's instructions: "Always use `list(tensor.shape)` before performing Python slice operations. `ttnn.Shape` does not support `[:-1]`, `[1:]`, or similar slice syntax. Convert to list first."
- **ttnn-operation-architect**: When specifying `output_shape_expr` in TDD stages, always wrap in `list()` conversion, e.g., `list(shape[:-1]) + [32]` and add a comment noting this is required for ttnn.Shape compatibility.

### Issue 3: Tensor accessor include path mismatch in builder's helper-to-include mapping

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 -- Infrastructure |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 compilation error (fixed in same session) |
| Time Cost | ~3 minutes |

**Problem**: The builder generated kernel stubs with `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"`, but the actual device-side header is at `api/tensor/tensor_accessor.h`. This caused a compilation error: `tensor_accessor.hpp: No such file or directory` (breadcrumb at 17:34:31).

**Root Cause**: The builder's helper-to-include mapping table (in its instructions) has an incorrect path for TensorAccessor. The builder explicitly logged this as upstream feedback (execution_log.md, Section 7, Recommendation 1): "Mapping table says `TensorAccessor -> #include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"`, should be `TensorAccessor -> #include "api/tensor/tensor_accessor.h"`."

**Fix for agents**:
- **Pipeline maintainer**: Update the builder's prompt helper-to-include mapping: change `TensorAccessor -> "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` to `TensorAccessor -> "api/tensor/tensor_accessor.h"`. This is a one-line instruction fix that would eliminate a guaranteed first-attempt failure for every operation that uses TensorAccessor.

### Issue 4: Builder-generated stage test reference functions had bare expressions (no return statements)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Infrastructure |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (builder fixed them before handoff) |
| Time Cost | ~2 minutes of builder time |

**Problem**: The auto-generated stage test files had `pytorch_reference` functions with bare expressions like `x` or `x.mean(dim=-1, keepdim=True)` instead of `return x` or `return x.mean(dim=-1, keepdim=True)`. Additionally, they used the variable name `x` instead of the parameter name `input_tensor`. The builder fixed these before committing (execution_log.md, Section 4).

**Root Cause**: The architect's `.tdd_state.json` `reference_body` fields contain mathematical expressions (e.g., `"x"`, `"x.mean(dim=-1, keepdim=True)"`) intended as human-readable references, not as Python return statements. The code that generates stage tests from these bodies naively injects them without adding `return` or mapping `x` to the actual parameter name.

**Fix for agents**:
- **ttnn-operation-architect**: Change `reference_body` values to be valid Python expressions using `input_tensor` as the variable name (e.g., `"input_tensor.mean(dim=-1, keepdim=True)"` not `"x.mean(dim=-1, keepdim=True)"`). Or add a note that these are reference-only and the builder must add `return` and map variable names.
- **Orchestrator/test generator**: When generating stage tests from `reference_body`, automatically prepend `return` and replace `x` with `input_tensor`.

### Issue 5: Output shape toggling between reduced and full across stages

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- reduce_mean, subtract_mean, variance, normalize |
| Agent | ttnn-kernel-writer-tdd (upstream fixes to entry point) |
| Retries Consumed | 0 (incorporated into upstream fixes, no test failures from this) |
| Time Cost | ~1 minute per stage transition (4 transitions) |

**Problem**: The entry point's output shape had to be toggled between reduced shape (`list(shape[:-1]) + [32]`) and full shape (`list(shape)`) at different TDD stages. Stages 2 (reduce_mean) and 4 (variance) produce reduced output, while stages 1, 3, 5, and 6 produce full-shape output. The kernel writer applied upstream fixes to `layer_norm_rm.py` at stages 2, 3, 4, and 5 to toggle the output shape (breadcrumbs at 17:49:48, 17:57:16, 18:04:09, 18:07:48).

**Root Cause**: The TDD pipeline's staged testing approach requires the entry point to produce different output shapes at different stages. The builder cannot know in advance which stages will need reduced output. The current approach requires the kernel writer to manually patch the entry point at each shape-changing stage boundary.

**Fix for agents**:
- **Orchestrator**: Consider adding a `stage_output_shape` field to `.tdd_state.json` that the entry point can read (e.g., via environment variable or a stage-aware parameter) to automatically select the correct output shape. Alternatively, have the TDD test override the output shape construction in the entry point rather than requiring kernel writer to patch it.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~5m (17:43-17:48) | 1 free, 0 hard | PASS | Namespace collision: 2m to diagnose + fix |
| reduce_mean | ~7m (17:49-17:55) | 0 free, 1 hard | PASS | Shape indexing TypeError: 3m to diagnose + upstream fix |
| subtract_mean | ~4m (17:56-18:00) | 0 free, 0 hard | PASS | Clean -- first-attempt pass |
| variance | ~4m (18:01-18:05) | 0 free, 0 hard | PASS | Clean -- first-attempt pass |
| normalize | ~4m (18:05-18:09) | 0 free, 0 hard | PASS | Clean -- first-attempt pass |
| affine | ~5m (18:10-18:15) | 0 free, 0 hard | PASS | Clean -- upstream fix for gamma/beta TA args was straightforward |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | reduce_mean upstream fix | ttnn-kernel-writer-tdd | ~3m | 4% | Shape slice indexing + writer CT args fix | 1 hard | ttnn.Shape API limitation not documented in builder instructions |
| 2 | data_pipeline namespace fix | ttnn-kernel-writer-tdd | ~2m | 3% | Fully qualified namespace for tilize/untilize enums | 1 free | Both namespaces have identical enum names |
| 3 | Builder include path fix | ttnn-generic-op-builder | ~3m | 4% | tensor_accessor.hpp vs tensor_accessor.h | 1 compile error | Wrong include path in helper-to-include mapping |
| 4 | batch_norm analysis | ttnn-operation-analyzer | ~12m | 17% | Large analysis (599 lines) with DeepWiki failure | 0 | DeepWiki unavailable; large reference codebase |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | Scaler computation in reader kernel using `float scaler_val = 1.0f / static_cast<float>(W)` (reader line 45) | The `prepare_reduce_scaler` helper already accepts a float and handles packing internally, but the reader also receives `scaler_packed` as a runtime arg (line 36) which is never used. The packed value is computed in the program descriptor but unused because the reader computes the scaler independently from `stick_size`. | Remove `scaler_value` from reader runtime args entirely since the reader computes it from the compile-time `stick_size`. Alternatively, have the reader use the runtime-provided packed value and remove the float computation. |
| ttnn-generic-op-builder | Generated both runtime `scaler_packed`/`eps_packed` AND compile-time `stick_size` that can derive the same value | Redundant parameters -- the scaler can be derived from `stick_size` (a compile-time arg), making the runtime arg unnecessary. | The architect should specify clearly whether scaler/epsilon are compile-time-derivable or must be runtime args. In this case, scaler is `1.0/W` where `W = stick_size/2`, all compile-time. Only epsilon truly needs a runtime arg (it's a user parameter). |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: orchestrator -> ttnn-operation-analyzer(s)

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation paths + role assignments |
| Quality | GOOD |
| Issues | DeepWiki was unavailable for the batch_norm analyzer; it fell back to local docs successfully |
| Downstream Impact | Minimal -- the batch_norm analysis was still comprehensive (599 lines). The tilize/untilize analyses were well-focused. |
| Suggestion | None required. Consider pre-caching DeepWiki responses for common references to avoid the ~2m fallback delay. |

### Handoff 2: ttnn-operation-analyzer(s) -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| Quality | GOOD |
| Issues | The batch_norm analysis at 599 lines was very large. However, the architect efficiently extracted the relevant patterns (architect breadcrumbs show 3 `reference_read` events in 3 seconds at 17:20:19-17:20:22). |
| Downstream Impact | Positive -- the architect had enough detail to design the 3-pass architecture, CB layout, and helper call sequences with full confidence. |
| Suggestion | The REPORT.md notes: "The batch_norm analyzer output was ~33KB - role-based focus directives helped but could be further narrowed for compute_core role." This is reasonable feedback -- a tighter scope for compute_core analysis would reduce token consumption without losing essential information. |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | ADEQUATE |
| Issues | (1) The `reference_body` values used `x` instead of `input_tensor` and lacked `return` statements, requiring builder fixes. (2) The helper-to-include mapping (in builder instructions, not from architect) had a wrong path for TensorAccessor. |
| Downstream Impact | Builder spent ~2m fixing stage test reference functions. Spent ~3m on include path error. |
| Suggestion | The architect should produce `reference_body` values that are valid Python return expressions using `input_tensor` as the variable name. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Python infrastructure (entry point, program descriptor, stub kernels), stage tests |
| Quality | ADEQUATE |
| Issues | The builder's handoff notes (execution_log.md Section 6) warned about potential binary_op_helpers.hpp path issue, which turned out not to materialize. However, the builder did not flag that `ttnn.Shape` does not support slice indexing, leading to a hard retry in Stage 2. The builder also left `num_tile_rows` as a compile-time arg initially, but this was already flagged and fixed by the kernel writer in Stage 1. |
| Downstream Impact | 4 out of 6 stages required upstream fixes to the program descriptor or entry point. While each fix was small (1-3 lines), this pattern of kernel-writer-fixing-builder-output consumed ~15% of Phase 4 time. |
| Suggestion | Add a validation step between build and TDD: run each stage test in "dry-run" mode (just allocate tensors, build program, but don't actually execute) to catch shape and argument errors before the kernel writer starts. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder | Fix tensor_accessor include path in helper-to-include mapping: `"api/tensor/tensor_accessor.h"` not `"ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` | HIGH | HIGH |
| ttnn-generic-op-builder | ttnn-kernel-writer-tdd | Always use `list(tensor.shape)` before Python slice operations on ttnn.Shape | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-generic-op-builder | Use `input_tensor` (not `x`) in `reference_body` and include `return` keyword | MEDIUM | MEDIUM |
| ttnn-kernel-writer-tdd | ttnn-kernel-writer-tdd | Never use `using namespace` for tilize_config or untilize_config when both are in the same kernel | HIGH | LOW |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Build | 4/6 TDD stages required upstream fixes to program descriptor or entry point | Add a build validation step: after builder creates stubs, run a "dry compile" of all kernels and a "dry execute" with stub kernels to catch argument/shape errors before kernel writer starts | MEDIUM |
| TDD | Output shape toggles between reduced/full at different stages | Add `stage_output_shape` to `.tdd_state.json` so the entry point can auto-select the correct shape per stage | LOW |
| Analysis | batch_norm analysis was 599 lines (~33KB) | Further narrow role-based focus directives for `compute_core` role to reduce token consumption | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | This run had zero numerical debugging -- all stages that passed compilation also passed numerically. The 3-pass architecture with explicit helper calls avoided the usual numerical pitfalls. |
| 2 | Too many planning stages before touching kernel code | N/A (DONE) | This run used the merged Architect agent. Worked well -- 7m design phase produced a correct and complete design. |
| 3 | `.tdd_state.json` coupling between architect and builder is fragile | NO | No schema issues. Builder successfully parsed all 6 stages. |
| 4 | No fast path for simple operations | N/A | layer_norm_rm is a complex operation (3-pass, 13 phases, 12 CBs) -- full pipeline was appropriate. |
| 6 | Builder runs on Sonnet while everything else uses Opus | PARTIALLY | The builder had 1 include-path error and needed to fix reference function bodies. These are detail-sensitive issues that could indicate model capability limitations, though both were resolved quickly. |
| 7 | Discovery phase uses keyword matching | NO | References were correctly selected. |
| 9 | No validation between architect output and builder output | PARTIALLY | The kernel writer had to apply upstream fixes at 4/6 stages due to mismatches between the architect's design and the builder's output. A static cross-check would have caught the `num_tile_rows` (CT vs RT) and writer CT args (input vs output dimensions) issues. |
| 11 | No incremental re-run capability | NO | Pipeline ran to completion without needing re-runs. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| ttnn.Shape does not support Python slice indexing | `ttnn.Shape` objects do not support `[:-1]` or similar slice syntax. This causes `TypeError` at runtime. All agents that construct output shapes from input shapes must use `list(tensor.shape)` first. | MEDIUM |
| Redundant scaler/epsilon runtime args when derivable from compile-time args | The reader kernel receives `scaler_packed` and `eps_packed` as runtime args but computes the scaler independently from the compile-time `stick_size`. The program descriptor computes and packs these values redundantly. The architect should clearly specify whether parameters are CT-derivable or require RT args. | LOW |
| Tilize/untilize config namespace collision | When both `tilize_helpers.hpp` and `untilize_helpers.hpp` are included in the same compute kernel, their config enum names collide. `using namespace` for either will cause compilation errors. Must use fully qualified names. | LOW |

---

## 8. Actionable Recommendations

### Recommendation 1: Fix tensor_accessor include path in builder instructions

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent prompt (helper-to-include mapping table)
- **Change**: Replace `TensorAccessor -> #include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` with `TensorAccessor -> #include "api/tensor/tensor_accessor.h"`
- **Expected Benefit**: Eliminates a guaranteed first-attempt compilation failure for every operation using TensorAccessor (estimated 3 minutes saved per run)
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add ttnn.Shape slice restriction to builder/architect instructions

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder and ttnn-operation-architect agent prompts
- **Change**: Add instruction: "ttnn.Shape does not support Python slice indexing. Always use `list(tensor.shape)` before performing `[:-1]`, `[1:]`, or similar operations."
- **Expected Benefit**: Prevents a hard retry in stages that produce reduced-shape output (estimated 3 minutes + 1 hard attempt saved)
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Architect should produce valid Python reference_body expressions

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Instruct architect to write `reference_body` values as valid Python expressions using `input_tensor` as the variable name, not `x`. For example: `"input_tensor.mean(dim=-1, keepdim=True)"` instead of `"x.mean(dim=-1, keepdim=True)"`.
- **Expected Benefit**: Builder no longer needs to fix reference function bodies (saves ~2 minutes of builder time per run)
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Add namespace collision warning for tilize+untilize kernels

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd agent prompt
- **Change**: Add instruction: "When a compute kernel uses both tilize and untilize helpers, always use fully qualified template parameters: `compute_kernel_lib::tilize_config::InitAndUninit`, `compute_kernel_lib::untilize_config::InitAndUninit`. Never use `using namespace` for either config namespace."
- **Expected Benefit**: Eliminates a common first-stage compilation error (saves ~2 minutes + 1 free retry)
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 5: Add static cross-validation between architect design and builder output

- **Type**: new_validation
- **Target**: Orchestrator (between Phase 3 and Phase 4)
- **Change**: After builder completes, run a validation script that compares: (a) CB indices and page counts in `op_design.md` vs `program_descriptor.py`, (b) kernel compile-time arg indices and types, (c) whether `num_tile_rows` is CT vs RT in design vs builder output.
- **Expected Benefit**: Catches upstream fix situations before kernel writer starts. Would have caught the `num_tile_rows` CT/RT mismatch and writer CT args mismatch in this run.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5/5 | Correctly selected batch_norm over softmax; all three references were highly relevant |
| Analysis quality | 4/5 | Comprehensive and well-focused; batch_norm analysis was slightly oversized at 599 lines but usable |
| Design completeness | 5/5 | All 13 compute phases specified with exact helper calls; CB state tables at each phase; zero design drift |
| Build correctness | 3/5 | Correct CB layout and work distribution, but: wrong include path, missing `return` in reference bodies, `num_tile_rows` as CT instead of RT |
| Kernel implementation | 5/5 | 2.8% budget usage; zero numerical issues; zero hangs; clean incremental implementation |
| Inter-agent communication | 4/5 | Good overall; architect-to-builder handoff had minor reference_body issues; builder-to-writer handoff required 4/6 upstream fixes |
| Logging/observability | 4/5 | All agents produced breadcrumbs with timestamps; builder produced execution log with handoff notes; kernel writer had per-stage cb_sync_check entries. Missing: no execution log from kernel writer (only breadcrumbs). |

### Top 3 Things to Fix

1. **Fix tensor_accessor include path in builder instructions** -- This is a zero-effort fix that eliminates a guaranteed compilation failure on every operation using TensorAccessor. Every single pipeline run that uses RM tensors hits this.
2. **Add ttnn.Shape slice restriction to agent instructions** -- Prevents a hard retry that costs 3 minutes and consumes budget. A one-line instruction addition.
3. **Add static cross-validation between architect and builder output** -- Would have caught 4 out of 6 upstream fixes needed in Phase 4 before the kernel writer even started.

### What Worked Best

The architect's design document was the single strongest aspect of this pipeline run. The `op_design.md` Part 2 was so detailed and correct that the kernel writer was able to implement all 13 compute phases verbatim, resulting in zero numerical issues, zero hangs, and only 2 retries (both compilation-level, fixed in under 3 minutes each). The design included CB state tables after each phase, exact helper call signatures with template parameters, explicit push/pop lifecycle management, and a clear TDD stage plan with narrow per-stage deltas. This level of design precision converted what is normally the hardest part of the pipeline (kernel implementation with numerical debugging) into straightforward code translation.
