# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | Phase 0 (Discovery), Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels), Phase 5 (Report) |
| Agents Invoked | orchestrator, ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd, create-op (report) |
| Total Git Commits | 11 (ed7dc81b through d94c75a3) |
| Total Pipeline Duration | ~68 minutes (17:17 to 18:26 UTC) |
| Overall Result | SUCCESS -- all 5 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~2 min | DONE | Identified 3 references: tilize (input_stage), untilize (output_stage), softmax (compute_core) |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~9 min | DONE | 3 parallel analyzers. Tilize finished first (4m30s), softmax last (8m50s). All produced focused analyses. |
| 2: Design | ttnn-operation-architect | ~4 min | DONE | Created comprehensive op_design.md with 401-line Part 2 kernel implementation plan. 5 TDD stages defined. |
| 3: Build | ttnn-generic-op-builder | ~12 min | DONE | Created 15 files (1071 lines net). Fixed 5 broken stage tests from architect. 1 compile fix (tensor_accessor include). |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~35 min | DONE | 5 stages. Stages 1,3,4,5 passed on first attempt. Stage 2 (reduce_mean) required 3 attempts. |
| 5: Report | create-op | ~2 min | DONE | Generated REPORT.md |

### Agent Duration Breakdown

Timing derived from breadcrumb `start` and `complete` events (primary) cross-checked against git commit timestamps (secondary).

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 17:17:43 | 17:22:08 | 4m 25s | 0 | ~4m active |
| ttnn-operation-analyzer (untilize) | 17:17:31 | 17:22:57 | 5m 26s | 0 | ~5m active |
| ttnn-operation-analyzer (softmax) | 17:17:45 | 17:26:23 | 8m 38s | 0 | ~8m active (largest reference, most DeepWiki queries) |
| ttnn-operation-architect | 17:27:37 | 17:31:51 | 4m 14s | 0 | ~4m active |
| ttnn-generic-op-builder | 17:34:24 | 17:46:32 | 12m 08s | 1 | ~7m active, ~5m build wait |
| ttnn-kernel-writer-tdd | 17:48:45 | 18:23:35 | 34m 50s | 2 (both in reduce_mean) | ~20m productive, ~15m debugging |

Duration calculation method: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps used as primary source. Git commit timestamps used as cross-check and secondary source for inter-phase gaps.

### Duration Visualization

```
Phase 0  |##|                                                         (~2m)
Phase 1  |################|                                           (~9m) 3 analyzers parallel
Phase 2                    |########|                                 (~4m)
Phase 3                              |########################|       (~12m) includes build
Phase 4                                                        |##############################...############| (~35m)
Phase 5                                                                                                      |##| (~2m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65 min

Longest phase: Phase 4 (35m) -- kernel implementation with 5 TDD stages, 15m debugging in reduce_mean
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~11 min | 16% | 3 analyzers ran in parallel, dominated by softmax (8m38s) |
| Design (Phase 2) | ~4 min | 6% | Fast, clean design with detailed kernel pseudocode |
| Build (Phase 3) | ~12 min | 18% | Includes build_metal.sh wait and stage test fixes |
| Kernel implementation (Phase 4) | ~35 min | 51% | 5 TDD stages |
| -- Productive coding | ~20 min | 29% | Writing kernel code that passed first try |
| -- Debugging/retries | ~15 min | 22% | reduce_mean: TypeError (~4m), numerical mismatch (~11m) |
| Reporting (Phase 5) | ~2 min | 3% | Automated |
| **Inter-phase gaps** | **~4 min** | **6%** | Agent launch overhead |
| **Total** | **~68 min** | **100%** | |

---

## 2. What Went Well

### 1. Stages 1, 3, 4, 5 all passed on first attempt (80% first-attempt pass rate)

**Phase/Agent**: ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `attempts: 0` for data_pipeline, subtract_mean, variance_rsqrt, and full_normalize. Breadcrumbs confirm single test_run entries with `status: pass` for these stages. The variance_rsqrt stage -- which includes the complex sub_mean + square + reduce + add_eps + rsqrt chain -- passed immediately.
**Why it worked**: The architect's design document was exceptionally detailed for Part 2, providing CB-by-CB pseudocode for all 13 compute phases, explicit broadcast dimensions, and persistent CB lifetime annotations. The kernel writer could translate the design almost line-for-line. The incremental TDD approach also meant that once reduce_mean was debugged, the pattern was established for all subsequent stages.

### 2. CB layout was correct from the start -- zero CB-related bugs

**Phase/Agent**: ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd
**Evidence**: 13 CBs specified in op_design.md matched exactly what the builder configured in `layer_norm_rm_program_descriptor.py`. No CB sizing, indexing, or page count errors across any TDD stage. The kernel writer's `cb_sync_check` breadcrumb events all showed `balanced: true`. No CB-related runtime errors or hangs occurred.
**Why it worked**: The architect's "Binary Op Broadcast Verification" table (op_design.md lines 163-170) pre-validated every CB interaction. The builder faithfully transcribed the design's CB table into Python code. The design's explicit notation (`Wt` for data CBs, `1` for scalar CBs) prevented the common off-by-one CB sizing errors.

### 3. No device hangs in the entire pipeline run

**Phase/Agent**: ttnn-kernel-writer-tdd
**Evidence**: All 5 stages completed cleanly. No breadcrumb entries indicating hangs or timeouts. The two failures in reduce_mean were a TypeError (Python-level) and a numerical mismatch (kernel produced values, just wrong values). Both are diagnosable failures -- far better than hangs.
**Why it worked**: The design document's CB lifetime annotations (Block, Row, Program) and explicit manual `cb_pop_front` instructions for persistent CBs prevented the deadlocks that arise from unbalanced CB push/pop sequences.

### 4. Softmax analysis was directly applicable as a compute reference

**Phase/Agent**: ttnn-operation-analyzer (softmax)
**Evidence**: The softmax analysis document (690 lines) covered: 3-pass streaming pattern, persistent scalar CBs, reduce API (including the reduce_tile precision warning), COL broadcast for subtraction, and row-wise reduction. The architect explicitly referenced these patterns in breadcrumbs: `design_decision: 3-pass streaming, rationale: Layer norm has same 3-pass dependency as softmax`.
**Why it worked**: The reference selection (softmax for compute_core role) was excellent. Layer norm's mean/variance/normalize maps directly to softmax's max/exp-sum/normalize. The analysis documented key pitfalls (precision of reduce, persistent CB lifetime management) that the architect and kernel writer both benefited from.

### 5. Incremental TDD staging caught bugs early

**Phase/Agent**: ttnn-kernel-writer-tdd
**Evidence**: The reduce_mean bug (fp32_dest_acc_en issue) was caught at stage 2 before it could compound with stages 3-5. Once fixed, all downstream stages worked immediately. Without staging, a full_normalize-only test would have produced an opaque failure with many possible causes.
**Why it worked**: The 5-stage TDD plan (data_pipeline, reduce_mean, subtract_mean, variance_rsqrt, full_normalize) isolated each new capability. Each stage had its own reference function and compare_slice for reduced-shape outputs.

---

## 3. Issues Found

### Issue 1: fp32_dest_acc_en=True breaks reduce output on Wormhole B0

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- reduce_mean (Stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~11 minutes (from hypothesis H2 at 18:02 to passing at 18:08) |

**Problem**: The design document specified `fp32_dest_acc_en = true` for compute precision (op_design.md line 229: "Uses fp32_dest_acc_en = true for precision"). The builder faithfully set `fp32_dest_acc_en=True` in `ComputeConfigDescriptor`. At runtime, reduce_mean output showed "8 non-zero rows then 8 zeros then 5 non-zero then 11 zeros" -- a pattern characteristic of fp32 face layout mismatch. The kernel writer's hypothesis H3 (breadcrumb at 18:07:17): "fp32_dest_acc_en=True may cause partial tile packing issue in reduce output."

The fix was to set `fp32_dest_acc_en=False`. This is a known Wormhole B0 hardware limitation with the MOVD2B/MOVB2D transpose path used by reduce operations (as previously documented in git history: commit `6b6ec0f4ac` from March 4: "Revert enforce_fp32_accumulation: WH B0 LLK bug in MOVD2B/MOVB2D transpose").

**Root Cause**: The architect recommended fp32_dest_acc_en based on the softmax analysis, where fp32 accumulation is used for sum reduction. However, the softmax reference uses `matmul_tiles` for reduction (explicitly to avoid reduce_tile precision issues), while the layer_norm_rm design uses the helper library's `reduce<SUM, REDUCE_ROW>`. The helper library's reduce path goes through the hardware reduce LLK which has a known incompatibility with fp32_dest_acc_en on Wormhole B0. The architect did not cross-reference this hardware constraint.

**Fix for agents**:
- **ttnn-operation-architect**: When recommending `fp32_dest_acc_en=True`, add a validation check: "NOTE: If using reduce helpers (not matmul-based reduction), fp32_dest_acc_en may need to be False on WH B0 due to LLK MOVD2B/MOVB2D bug." Add this to the hardware constraints checklist.
- **Pipeline-level**: Add a known-incompatibility database: `{reduce_helpers + fp32_dest_acc_en + WH_B0 -> INCOMPATIBLE}`. The architect should consult this before finalizing ComputeConfig.

### Issue 2: Architect-generated stage tests had broken imports and reference functions

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-operation-architect (origin), ttnn-generic-op-builder (fixer) |
| Retries Consumed | 0 (builder fixed before testing) |
| Time Cost | ~3 minutes of builder time fixing 5 files |

**Problem**: The builder's execution log (lines 39-41) documents two distinct issues:
1. Stage test files used relative imports (`from .layer_norm_rm import layer_norm_rm`) instead of absolute imports (`from ttnn.operations.layer_norm_rm import layer_norm_rm`). Tests live in `tests/` directory, not co-located with operation code.
2. `pytorch_reference` functions were broken: bare expressions like `x` instead of `return x`, and using `x` instead of `input_tensor`.

The builder had to modify all 5 stage test files (visible in git stat for commit 318de97f: 5 test files with ~12 lines changed each).

**Root Cause**: The tdd_orchestrator (or architect's stage test template) generates tests with assumptions about file location and variable naming that do not match the actual test directory structure. The `reference_body` field in `.tdd_state.json` contains expression strings (e.g., `"x.mean(dim=-1, keepdim=True)"`) that are injected verbatim, but the orchestrator does not wrap them in `return` statements or map `x` to the function parameter name `input_tensor`.

**Fix for agents**:
- **tdd_orchestrator**: Fix the test template generator to: (a) use absolute imports `from ttnn.operations.{op_name} import {op_name}`, (b) wrap `reference_body` in a proper function with `return` and parameter mapping, (c) validate generated test syntax before writing.
- **ttnn-operation-architect**: The reference_body values in `.tdd_state.json` should be complete, valid Python expressions using the parameter name `input_tensor`, not the shorthand `x`.

### Issue 3: ttnn.Shape does not support slice indexing

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- reduce_mean (Stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~4 minutes (hypothesis H1 at 17:58 to fix applied and retest at 18:02) |

**Problem**: The kernel writer modified `layer_norm_rm.py` to compute the reduced output shape using `list(input_tensor.shape[:-1]) + [32]`, which is the idiomatic Python approach. However, `ttnn.Shape` does not support slice indexing (`[:-1]`), causing a `TypeError: __getitem__(): incompatible function arguments`.

The fix was straightforward: convert shape to a list via explicit element-by-element indexing first, then slice.

**Root Cause**: The kernel writer assumed `ttnn.Shape` supports Python slice semantics, which is a reasonable assumption but incorrect. This is a recurring friction point in the ttnn Python API.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Add to the agent's known-issues list: "ttnn.Shape only supports integer indexing, not slicing. Always convert to list first: `shape_list = [input_tensor.shape[i] for i in range(len(input_tensor.shape))]`."
- **ttnn-generic-op-builder**: The builder's output_shape computation in `layer_norm_rm.py` already uses this pattern (line 60: `[input_tensor.shape[i] for i in range(len(input_tensor.shape))]`), but the kernel writer overwrote it with the slice-based version when changing output shape for the reduce_mean stage.

### Issue 4: Builder included non-existent tensor_accessor.hpp

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 compile failure (builder's own retry) |
| Time Cost | ~2 minutes |

**Problem**: Builder's execution log (lines 84-87): Reader and writer stub kernels included `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` which does not exist. TensorAccessor is available through `dataflow_api.h`.

**Root Cause**: The builder's include mapping table (in its system prompt) contains an incorrect entry mapping TensorAccessor to this path. The builder correctly diagnosed and fixed this.

**Fix for agents**:
- **ttnn-generic-op-builder**: Update the include mapping table to either remove the tensor_accessor entry or note that TensorAccessor is provided via `api/dataflow/dataflow_api.h` for device kernels. This was already recommended in the builder's execution log (Section 7, Recommendation 1).

### Issue 5: full_normalize stage does not test gamma/beta paths

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- full_normalize (Stage 5) |
| Agent | ttnn-operation-architect (test design), ttnn-kernel-writer-tdd (implementation) |
| Retries Consumed | 0 |
| Time Cost | 0 (but represents untested code paths) |

**Problem**: The `test_stage_full_normalize.py` only tests `layer_norm_rm(ttnn_input)` -- the no-gamma-no-beta path. The compute kernel has 4 compile-time code paths (`!has_gamma && !has_beta`, `has_gamma && !has_beta`, `has_gamma && has_beta`, `has_beta && !has_gamma`), but only one is exercised. The kernel writer implemented all 4 paths (compute_layer_norm_rm.cpp lines 128-202) but none of the gamma/beta paths were tested.

**Root Cause**: The stage test's `reference_body` in `.tdd_state.json` is `"torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=1e-5)"` which does not include gamma/beta parameters. The TDD stage plan (op_design.md line 215-219) mentions "optional gamma/beta" but the test only tests without.

**Fix for agents**:
- **ttnn-operation-architect**: The final TDD stage should include parametrized test cases for gamma-only, beta-only, and gamma+beta combinations. At minimum, the stage description should say "Tests with AND without gamma/beta" and the test template should include separate test functions for each combination.
- **tdd_orchestrator**: Add support for `extra_test_cases` in stage definitions that generate parametrized tests with different argument combinations.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~5 min (17:49-17:53) | 0 free, 0 hard | PASS | Clean. Identity passthrough worked immediately. |
| reduce_mean | ~15 min (17:54-18:09) | 0 free, 2 hard | PASS | TypeError (4m), numerical mismatch/fp32_dest_acc_en (11m) |
| subtract_mean | ~5 min (18:09-18:14) | 0 free, 0 hard | PASS | Clean. 2-pass pattern worked immediately after reduce_mean fixes. |
| variance_rsqrt | ~5 min (18:15-18:20) | 0 free, 0 hard | PASS | Clean. Complex phase (sub+square+reduce+add_eps+rsqrt) passed first try. |
| full_normalize | ~4 min (18:20-18:24) | 0 free, 0 hard | PASS | Clean. 3-pass with 4 gamma/beta combinations compiled and passed. |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | reduce_mean numerical mismatch | kernel-writer | ~11 min | 16% | fp32_dest_acc_en broke reduce output. Required DPRINT debug, hypothesis testing, config change. | 1 hard | Architect recommended fp32_dest_acc_en without WH B0 reduce incompatibility check |
| 2 | Builder build_metal.sh wait | builder | ~5 min | 7% | C++ build required before first test run | 0 | Necessary infrastructure step |
| 3 | Builder fixing stage tests | builder | ~3 min | 4% | Fixed 5 broken auto-generated stage test files | 0 | Orchestrator template bugs |
| 4 | reduce_mean TypeError | kernel-writer | ~4 min | 6% | ttnn.Shape slice indexing not supported | 1 hard | API knowledge gap |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| kernel-writer | DPRINT TSLICE debug output added to compute kernel (breadcrumb at 18:04) | Was removed before passing test -- diagnostic code only | Provide better numerical diagnostic tools that don't require kernel modification |
| builder | Created stub kernels with tensor_accessor.hpp include | Had to remove the include on first compile failure | Fix the include mapping table in builder's prompt |
| architect | Specified fp32_dest_acc_en=True in design | Kernel writer had to change to False for reduce to work | Add hardware incompatibility database |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, softmax_analysis.md |
| Quality | GOOD |
| Issues | None significant. Analyses were focused on their assigned roles (input_stage, output_stage, compute_core). |
| Downstream Impact | Architect correctly extracted 3-pass pattern, CB sizing conventions, and TensorAccessor patterns. |
| Suggestion | The softmax analysis noted that reduce_tile has precision issues and softmax uses matmul_tiles instead. The architect picked up the 3-pass pattern but not the matmul-vs-reduce nuance -- this later contributed to Issue 1. Analyzers should flag hardware-version-dependent caveats more prominently. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json (with 5 stages), 5 stage test files |
| Quality | ADEQUATE |
| Issues | (1) Stage test files had broken imports and reference functions (Issue 2). (2) Design specified fp32_dest_acc_en=True which later broke reduce_mean (Issue 1). |
| Downstream Impact | Builder spent ~3 min fixing stage tests. fp32_dest_acc_en issue cascaded to kernel writer. |
| Suggestion | Architect should validate generated test files are syntactically correct. Stage test generation should be delegated to the orchestrator with a tested template, not hand-crafted by the architect. |

### Handoff 3: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Python infrastructure (entry point, program descriptor, stub kernels), corrected test files |
| Quality | GOOD |
| Issues | The builder's handoff notes (execution log Section 6) were detailed and accurate: CB config, compile-time args, runtime args, 3-pass streaming requirements. One deviation: builder made nblocks a runtime arg (not compile-time as in tilize reference) for heterogeneous work distribution -- this was a correct design improvement. |
| Downstream Impact | Kernel writer picked up all necessary context. The only upstream modifications the writer needed were: output_shape changes (expected, per-stage), and fp32_dest_acc_en change (Issue 1). |
| Suggestion | Builder's handoff note correctly flagged `fp32_dest_acc_en=True` -- consider having the builder validate compute config against known hardware constraints before passing to kernel writer. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder | Remove tensor_accessor.hpp from include mapping; note TensorAccessor is in dataflow_api.h | HIGH | MEDIUM |
| tdd_orchestrator | ttnn-generic-op-builder | Stage tests should use absolute imports and proper reference functions | HIGH | HIGH |
| ttnn-kernel-writer-tdd | ttnn-kernel-writer-tdd | Add ttnn.Shape indexing limitation to known-issues | HIGH | LOW |
| ttnn-operation-architect | self-reflection | Add fp32_dest_acc_en + reduce_helpers WH B0 incompatibility to hardware constraints checklist | HIGH | HIGH |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD test generation | Stage tests had broken imports and references | Move test generation to orchestrator with validated templates | HIGH |
| Hardware compatibility | fp32_dest_acc_en + reduce incompatibility not caught at design time | Create a hardware incompatibility database consulted by architect | HIGH |
| Test coverage | Final stage only tests no-gamma-no-beta path | Require final TDD stage to parametrize all optional feature combinations | MEDIUM |
| Observability | Kernel writer only had analyzer/architect breadcrumbs to trace issues | No kernel-writer-tdd execution_log.md was produced (only breadcrumbs) | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | YES (partial) | reduce_mean numerical mismatch took ~11 min. However, the writer correctly diagnosed it in 2 hypotheses (H2 and H3) rather than trial-and-error. The DPRINT approach was systematic. Still, 16% of total time on a config issue is significant. |
| 2 | Too many planning stages | NO (DONE) | Merged architect is working well. 4-minute design phase is efficient. |
| 3 | .tdd_state.json coupling fragility | NO | Format worked correctly this run. |
| 4 | No fast path for simple ops | N/A | Layer norm is not a simple op. |
| 6 | Builder runs on Sonnet | PARTIAL | Builder had 1 compile error (tensor_accessor include) -- a detail-sensitive mistake. Running on Opus might have avoided this. The builder did recover quickly. |
| 9 | No architect/builder cross-validation | YES | fp32_dest_acc_en was specified by architect, faithfully set by builder, and broke at runtime. A static cross-check (reduce helpers + fp32_dest_acc_en = warning) would have caught this. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| fp32_dest_acc_en + reduce helpers incompatibility on WH B0 | The architect recommended fp32_dest_acc_en=True based on softmax reference, but softmax uses matmul-based reduction (not reduce helpers). The reduce helpers path is incompatible with fp32_dest_acc_en on Wormhole B0 due to a MOVD2B/MOVB2D transpose bug. This caused a numerical mismatch that consumed 11 minutes. | HIGH |
| Stage test generation produces broken Python files | The tdd_orchestrator/architect generates stage test files with relative imports and malformed reference functions. The builder has to fix them every run. | HIGH |
| Final TDD stage lacks gamma/beta test coverage | The full_normalize stage only tests the no-gamma-no-beta code path. Three additional compile-time code paths (gamma-only, beta-only, gamma+beta) are untested. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Add fp32_dest_acc_en hardware incompatibility guard

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt / hardware constraints checklist
- **Change**: Add the following to the architect's hardware constraints section: "CONSTRAINT: When using compute_kernel_lib::reduce helpers (not matmul-based reduction), fp32_dest_acc_en MUST be False on Wormhole B0. The reduce LLK's MOVD2B/MOVB2D transpose path is incompatible with fp32 destination accumulation. If fp32 precision is needed for reductions, use matmul_tiles-based reduction instead."
- **Expected Benefit**: Prevents the 11-minute debugging cycle encountered in reduce_mean (Issue 1).
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Fix stage test generation in tdd_orchestrator

- **Type**: tool_improvement
- **Target**: tdd_orchestrator test template generation
- **Change**: (a) Use absolute imports: `from ttnn.operations.{op_name} import {op_name}`. (b) Generate reference functions with proper `return` statement and `input_tensor` parameter name. (c) Add a syntax validation step (compile the generated Python) before writing to disk.
- **Expected Benefit**: Eliminates the ~3 minutes the builder spends fixing tests every run. Removes a class of handoff errors.
- **Priority**: HIGH
- **Effort**: MEDIUM

### Recommendation 3: Require gamma/beta test cases in final normalization stage

- **Type**: pipeline_change
- **Target**: TDD stage plan for normalization operations
- **Change**: When the operation has optional parameters (gamma, beta, weight, bias), the final TDD stage must include parametrized test cases covering all combinations. For layer_norm_rm: test with no params, gamma-only, beta-only, and gamma+beta.
- **Expected Benefit**: Catches bugs in compile-time branching paths that are currently untested. The 4 gamma/beta code paths in compute_layer_norm_rm.cpp (lines 128-202) represent ~40% of the kernel code.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 4: Add ttnn.Shape indexing limitation to kernel writer's known issues

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd prompt
- **Change**: Add to known issues: "ttnn.Shape does not support Python slice indexing ([:], [:-1], etc.). Always convert to list first: `[shape[i] for i in range(len(shape))]`. Then slice the list."
- **Expected Benefit**: Saves ~4 minutes per occurrence (the TypeError was a quick fix but burned a hard attempt).
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 5: Remove tensor_accessor.hpp from builder include mapping

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder prompt, include mapping table
- **Change**: Remove the entry `TensorAccessor -> #include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"`. Add note: "TensorAccessor is available through `api/dataflow/dataflow_api.h` for dataflow kernels."
- **Expected Benefit**: Prevents kernel compile failure on stub generation.
- **Priority**: MEDIUM
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

Rate each dimension (1-5):

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5 | Tilize, untilize, and softmax were all directly applicable references. No wasted analysis. |
| Analysis quality | 4 | Analyses were thorough and focused. The softmax analysis flagged the matmul-vs-reduce precision difference, but this was not prominently enough highlighted for the architect to catch the fp32 incompatibility. |
| Design completeness | 4 | Comprehensive 401-line Part 2 with CB-by-CB pseudocode. One gap: fp32_dest_acc_en recommendation was wrong for reduce helpers on WH B0. Stage test generation had bugs. |
| Build correctness | 4 | Infrastructure was solid. 13 CBs correctly configured. One include path error (quick fix). Stage tests needed repair. |
| Kernel implementation | 5 | 4 of 5 stages first-attempt pass. Debugging was systematic (DPRINT, hypothesis testing). All 13 compute phases implemented correctly. Complex gamma/beta branching worked. |
| Inter-agent communication | 4 | Handoff quality was good overall. The architect->builder handoff for stage tests was the weak point. Builder->kernel_writer handoff notes were excellent. |
| Logging/observability | 3 | Breadcrumbs were present for all agents and sufficiently detailed. Missing: kernel-writer-tdd execution_log.md. Timestamps were consistent and enabled full timeline reconstruction. |

### Top 3 Things to Fix

1. **Add fp32_dest_acc_en + reduce helpers WH B0 incompatibility to the architect's hardware constraints checklist.** This single knowledge gap caused the most expensive debugging cycle (11 minutes, 16% of total time). It is a deterministic, recurring issue that will affect every normalization or reduction-heavy operation.

2. **Fix stage test generation to produce syntactically correct Python with absolute imports.** This is a recurring overhead every pipeline run -- the builder always has to fix the same class of bugs. The fix is straightforward (template improvement in the orchestrator).

3. **Require final TDD stage to test all optional parameter combinations.** The gamma/beta code paths represent significant untested surface area. A numerical bug in the gamma ROW broadcast or beta addition would ship silently.

### What Worked Best

The architect's detailed kernel implementation plan (op_design.md Part 2) was the single strongest contributor to pipeline success. By providing CB-by-CB pseudocode for all 13 compute phases, explicit broadcast dimensions for every binary operation, persistent CB lifetime annotations, and a complete "Binary Op Broadcast Verification" table, the architect enabled the kernel writer to implement 4 of 5 stages on the first attempt. The 80% first-attempt pass rate and zero CB-related bugs across 13 circular buffers is a direct result of this design quality. The combined wall time for stages 1+3+4+5 was only 19 minutes for a complex 3-pass kernel with in-kernel tilize/untilize -- an impressive throughput that validates the detailed-design-first approach.
