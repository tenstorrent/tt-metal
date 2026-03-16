# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | orchestrator, 3x ttnn-operation-analyzer, ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 10 on this run (67d6d57 through 535441a) |
| Total Pipeline Duration | ~53 minutes (08:59 - 09:52 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1 min | PASS | Identified tilize, untilize, batch_norm as references |
| 1: Analysis | 3x ttnn-operation-analyzer | ~11 min (08:59-09:10) | PASS | Three parallel analyzers produced tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| 2: Design | ttnn-operation-architect | ~9 min (09:07-09:16) | PASS | Comprehensive op_design.md with 9-phase compute pipeline, 12 CBs, 4 TDD stages |
| 3: Build | ttnn-generic-op-builder | ~8 min (09:19-09:27) | PASS | Created all infrastructure files; 1 free retry for tensor_accessor include path |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~22 min (09:27-09:49) | PASS | All 4 stages passed; 3 upstream fixes during data_pipeline, stages 2-4 first-attempt |
| 5: Report | orchestrator | ~2 min (09:50-09:52) | PASS | REPORT.md generated |
| **Total** | | **~53 min** | | Earliest commit 08:59, latest 09:52 |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (untilize) | 08:59:02 | 09:09:54 | ~11m | 0 | ~11m active (read, research, write) |
| ttnn-operation-analyzer (tilize) | 08:59:05 | 09:09:40 | ~11m | 0 | ~11m active |
| ttnn-operation-analyzer (batch_norm) | 08:59:30 | 09:09:55 | ~10m | 0 | ~10m active |
| ttnn-operation-architect | 09:07:33 | 09:16:11 | ~9m | 0 | ~9m active, 0m debugging |
| ttnn-generic-op-builder | 09:18:55 | 09:26:03 | ~7m | 1 free | ~5m productive, ~2m fixing include path |
| ttnn-kernel-writer-tdd | 09:30:38 | 09:49:25 | ~19m | 0 hard, 2 free | ~15m productive, ~4m upstream fixes |

**Duration calculation method**: Derived from breadcrumb `"event":"start"` and `"event":"complete"` timestamps. All agents had both events recorded. Cross-verified against git commit timestamps which are consistent.

### Duration Visualization

```
Phase 0  |#|                                                  (~1m)
Phase 1  |############|                                       (~11m) 3 analyzers in parallel
Phase 2       |##########|                                    (~9m)
Phase 3              |########|                               (~7m)
Phase 4                      |#####################|          (~19m)
Phase 5                                             |##|      (~2m)
         0    5    10   15   20   25   30   35   40   45   50 min

Longest phase: Phase 4 (~19m) -- TDD kernel implementation across 4 stages
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~12 min | 23% | 3 parallel analyzers + discovery |
| Design (Phase 2) | ~9 min | 17% | Single architect |
| Build (Phase 3) | ~7 min | 13% | 1 free retry on include path |
| Kernel implementation (Phase 4) | ~19 min | 36% | 4 TDD stages |
| -- Productive coding | ~15 min | 28% | Writing kernel code that passed |
| -- Debugging/retries | ~4 min | 8% | Upstream fixes (conftest, strict-aliasing, placeholder) |
| Reporting (Phase 5) | ~2 min | 4% | |
| Overlap/gaps | ~4 min | 7% | Phases 1-2 overlapped by ~3 min |
| **Total** | **~53 min** | **100%** | |

---

## 2. What Went Well

### 1. All 4 TDD stages passed with zero hard attempts

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `"attempts": 0, "free_retries": 0` for all 4 stages. The breadcrumbs confirm: `data_pipeline` passed after 2 free retries (upstream infrastructure fixes only), while `centering`, `normalize`, and `affine` each passed on the very first test run.
**Why it worked**: The architect's design document was exceptionally detailed -- every compute phase had explicit helper function calls with correct policy parameters, CB persistence annotations, and CB state tables. The kernel writer could translate the design almost line-by-line into working code.

### 2. Comprehensive helper library usage -- 100% compliance

**Phase/Agent**: Phase 2 (architect) and Phase 4 (kernel writer)
**Evidence**: The compute kernel uses 7 distinct helper functions (tilize, untilize, reduce x2, sub, mul, add, square) with zero raw `tile_regs_acquire`/`cb_wait_front` operations. No helper was missed, misused, or wrapped with redundant CB operations. See Section 10 for full audit.
**Why it worked**: The architect explicitly mapped every compute phase to a specific helper call in the design document. The kernel writer followed these mappings faithfully.

### 3. CB layout was correct from the start -- zero CB-related bugs

**Phase/Agent**: Phase 2 (architect) through Phase 4 (kernel writer)
**Evidence**: 15 circular buffers (12 always-allocated + 3 conditional for gamma/beta) were defined in the design, implemented identically in the program descriptor, and consumed by the kernel writer without any CB sizing, indexing, or synchronization bugs across all 4 TDD stages.
**Why it worked**: The architect provided a detailed CB table with purposes, producers, consumers, page counts, and lifetimes. The binary op broadcast verification table and CB state tables after key phases made the data flow unambiguous.

### 4. Clean git history -- one commit per phase, minimal noise

**Phase/Agent**: All phases
**Evidence**: The run's commits are: 67d6d57 (analyzer untilize), 379574c (analyzer batch_norm), aa12b88 (architect design), 524dd7d (architect breadcrumbs), b1aecb8 (builder stubs), 8ce0965 (builder execution log), 9a62d98 (TDD data_pipeline), 301fa7b (TDD centering), 7f2ff1e (TDD normalize), d2921b5 (TDD affine), 535441a (report). Each commit corresponds to a meaningful unit of work with no fixup or revert commits.
**Why it worked**: Each agent committed cleanly at completion. No mid-stage panic commits or desperate debugging checkpoints.

### 5. Architect's rsqrt post-op pattern worked perfectly

**Phase/Agent**: Phase 2 (architect design), Phase 4 (kernel writer)
**Evidence**: The design specified `add<SCALAR>` with a lambda post-op `[](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); }` to combine add(var, eps) and rsqrt into a single DEST pass. This compiled and produced correct results on the first attempt.
**Why it worked**: The binary_op helper's post-op callback interface was well-documented in the helper library, and the architect correctly identified it as the right pattern for fused add+rsqrt.

---

## 3. Issues Found

### Issue 1: TensorAccessor include path wrong in system prompt

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry |
| Time Cost | ~2 minutes |

**Problem**: The builder's kernel stubs used `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` which does not exist for device-side kernel compilation. The correct path is `#include "api/tensor/tensor_accessor.h"`. This caused a compilation failure on the first test run (breadcrumb at 09:22:44).

**Root Cause**: The builder's system prompt (or helper-to-include mapping table) provides a host-side include path that differs from the device-side include path. The builder execution log (Section 7) explicitly calls this out: "The system prompt says `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` but device kernels need `api/tensor/tensor_accessor.h`."

**Fix for agents**:
- **ttnn-generic-op-builder instructions**: Update the helper-to-include mapping to use `api/tensor/tensor_accessor.h` for all kernel stubs.

### Issue 2: TensorAccessorArgs placeholder size mismatch (1 vs 2 elements)

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- data_pipeline |
| Agent | ttnn-kernel-writer-tdd (fixed builder's output) |
| Retries Consumed | Part of data_pipeline's free retries |
| Time Cost | ~1 minute |

**Problem**: When gamma or beta is absent, the program descriptor originally used `[0]` as a placeholder for the TensorAccessorArgs compile-time args. However, interleaved TensorAccessorArgs produces 2 compile-time args (args_config + aligned_page_size). Using a 1-element placeholder caused the next chained TensorAccessorArgs declaration in the reader kernel to read from the wrong offset, corrupting subsequent args.

**Root Cause**: Undocumented behavior -- the number of compile-time args that TensorAccessorArgs produces depends on the memory layout (interleaved = 2 args). The builder had no way to know this without reading the TensorAccessorArgs implementation.

**Fix for agents**:
- **ttnn-operation-architect**: When specifying TensorAccessorArgs in kernel arg tables, explicitly note the placeholder size for absent optional tensors (e.g., "placeholder: `[0, 0]` for interleaved").
- **ttnn-generic-op-builder instructions**: Document that interleaved TensorAccessorArgs uses 2 CT args, so absent tensor placeholders need `[0, 0]`.

### Issue 3: Scaler/epsilon encoding confusion (packed bf16 vs raw float32 bits)

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- data_pipeline |
| Agent | ttnn-kernel-writer-tdd (fixed builder's output) |
| Retries Consumed | Part of data_pipeline's free retries |
| Time Cost | ~2 minutes |

**Problem**: The reader kernel originally received scaler and epsilon as packed bf16 values and used `reinterpret_cast` to convert them to float for `prepare_reduce_scaler()`. This caused a strict-aliasing compilation error. The fix changed the encoding to raw float32 bit patterns with union-based decoding.

**Root Cause**: The design document specified `scaler_value` as "1/W as packed bf16 (bf16<<16 | bf16)" in the runtime args table (op_design.md line 121). The `prepare_reduce_scaler` helper actually expects a plain float. The packed bf16 format was inherited from an earlier design iteration and was incorrect for this helper.

**Fix for agents**:
- **ttnn-operation-architect**: When specifying scaler runtime args, state the format expected by the actual helper function. For `prepare_reduce_scaler`, that's "raw float passed as uint32_t bits via union".
- **Pipeline documentation**: Clarify that `prepare_reduce_scaler<cb_id>(float)` takes a plain float, not packed bf16.

### Issue 4: conftest.py device fixture uses positional arg

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- data_pipeline |
| Agent | ttnn-kernel-writer-tdd (fixed builder's output) |
| Retries Consumed | 0 (discovered alongside Issue 3, fixed together) |
| Time Cost | ~30 seconds |

**Problem**: The builder-generated conftest.py called `ttnn.open_device(0)` instead of `ttnn.open_device(device_id=0)`. The TTNN API requires keyword argument `device_id`.

**Root Cause**: Builder template/instructions did not specify the keyword argument requirement.

**Fix for agents**:
- **ttnn-generic-op-builder**: Update conftest template to use `ttnn.open_device(device_id=0)`.

### Issue 5: Builder handoff note contradicts final implementation

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build handoff |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 |
| Time Cost | 0 (caught by kernel writer, no confusion) |

**Problem**: The builder's execution log handoff notes (Section 6) state: "Scaler is packed as bf16: (bf16 << 16) | bf16 via float_to_packed_bf16()" and "Optional tensor placeholder: reader CT args use [0] placeholder when gamma/beta absent." Both of these were already fixed by the time the builder committed, yet the handoff notes describe the pre-fix state.

**Root Cause**: The execution log was likely written before the fixes were applied, and not updated afterward. This is a logging hygiene issue.

**Fix for agents**:
- **ttnn-generic-op-builder**: Execution log handoff notes should describe the FINAL state of the code, not the initial state. The agent should review and update handoff notes after applying fixes.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~9 min (09:30-09:39) | 2 free, 0 hard | PASS | Upstream infrastructure fixes (conftest, strict-aliasing, placeholder) |
| centering | ~3 min (09:40-09:42) | 0 free, 0 hard | PASS | Clean -- first attempt pass |
| normalize | ~2 min (09:43-09:44) | 0 free, 0 hard | PASS | Clean -- first attempt pass |
| affine | ~5 min (09:45-09:50) | 0 free, 0 hard | PASS | Clean -- slightly longer due to conditional affine branching code |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | data_pipeline upstream fixes | kernel-writer-tdd | ~4 min | 8% | Fixing conftest, strict-aliasing, placeholder, scaler encoding | 2 free | Builder output had known recurring issues |
| 2 | Analysis research phase | 3x analyzer | ~5 min | 9% | DeepWiki queries and codebase reads before writing analysis | 0 | Normal research time, not a waste |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-generic-op-builder | Created stubs with wrong include path | Had to re-run after fixing | Fix include path in builder instructions |
| ttnn-generic-op-builder | Wrote `float_to_packed_bf16()` utility | Replaced by `float_to_uint32_bits()` during TDD | Design doc should specify correct encoding from the start |

Overall efficiency is high. Only ~8% of total time was spent on debugging/retries, and all of it was on known recurring infrastructure issues (not novel bugs).

---

## 5. Inter-Agent Communication Issues

### Handoff 1: orchestrator -> ttnn-operation-analyzer(s)

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation paths (tilize, untilize, batch_norm) |
| Quality | GOOD |
| Issues | None -- all three references were appropriate |
| Downstream Impact | Analyzers produced useful, focused analysis documents |
| Suggestion | None needed |

### Handoff 2: ttnn-operation-analyzer(s) -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| Quality | GOOD |
| Issues | The architect noted "No single reference analysis; designed from requirements + helper library" (op_design.md line 8), suggesting the architect may not have heavily relied on the analysis documents |
| Downstream Impact | Minimal negative impact -- the architect produced an excellent design regardless |
| Suggestion | Architect should explicitly cite which analysis findings it incorporated, so the pipeline can measure analysis ROI |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | GOOD |
| Issues | Design doc scaler format (packed bf16) was wrong; builder followed it faithfully, producing code that needed to be fixed later |
| Downstream Impact | 1 free retry in TDD stage 1, plus the dead `float_to_packed_bf16()` function still in the codebase |
| Suggestion | Architect should validate scaler format against the actual `prepare_reduce_scaler` helper signature |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files, execution log |
| Quality | ADEQUATE |
| Issues | (1) Execution log handoff notes described pre-fix state. (2) Builder used wrong include path. (3) Placeholder size was wrong. The kernel writer had to fix 3 upstream issues. |
| Downstream Impact | ~4 minutes of upstream fixing in the first TDD stage |
| Suggestion | Builder should run a smoke test that validates kernel compilation before handoff |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder (self) | Fix TensorAccessor include path from `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` to `api/tensor/tensor_accessor.h` | HIGH | HIGH |
| ttnn-generic-op-builder | ttnn-kernel-writer-tdd | TensorAccessorArgs placeholder for absent interleaved tensors must be `[0, 0]` not `[0]` | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Scaler runtime args should specify "raw float32 bits" not "packed bf16" for `prepare_reduce_scaler` | HIGH | MEDIUM |
| ttnn-generic-op-builder | ttnn-kernel-writer-tdd | conftest.py must use `ttnn.open_device(device_id=0)` keyword arg | HIGH | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Build | Builder creates stubs with wrong include path every run | Add include-path validation to builder agent or fix system prompt mapping table | HIGH |
| Build | Builder smoke test only validates Python-side, not kernel compilation | Add a compile-only test before handoff to catch include/syntax errors | MEDIUM |
| Design | Scaler format in design doc was wrong | Add a "format cross-check" step where architect verifies runtime arg formats against helper function signatures | MEDIUM |
| Logging | Builder execution log handoff notes describe pre-fix state | Require execution log to be written AFTER all fixes are applied | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | No numerical debugging was needed -- all stages passed immediately. This is a strong positive signal for the design quality. |
| 2 | Too many planning stages (long leash) | N/A (DONE) | Pipeline already uses merged Architect agent. Confirmed working well in this run. |
| 3 | `.tdd_state.json` coupling is fragile | NO | TDD state file was consumed correctly. No format issues. |
| 4 | No fast path for simple operations | NO | Layer norm is not a simple operation -- full pipeline was appropriate. |
| 6 | Builder runs on Sonnet | YES (partially) | The builder hit the TensorAccessor include path issue, which is a known recurring error consistent with detail-sensitivity challenges. |
| 7 | Discovery keyword matching | NO | References were correctly identified. |
| 9 | No architect/builder cross-validation | YES | Scaler format mismatch between architect's design and actual helper requirements was not caught until TDD runtime. A cross-validation step would have caught this. |
| 11 | No incremental re-run capability | NO | Pipeline completed fully; no need for resume. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Builder execution log handoff notes lag behind fixes | The builder's execution log Section 6 (handoff notes) described the pre-fix state of scaler encoding and placeholder sizes, which could mislead the kernel writer if relied upon | LOW |
| Architect analysis citation gap | Architect stated "No single reference analysis" in design doc despite receiving 3 analysis documents. If analysis is not being used, it represents wasted pipeline time (~11 min of 3 parallel analyzers) | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Fix TensorAccessor include path in builder instructions

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder system prompt / helper-to-include mapping
- **Change**: Replace `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` with `api/tensor/tensor_accessor.h` in all kernel stub templates
- **Expected Benefit**: Eliminates 1 free retry per pipeline run (~2 min savings)
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Document TensorAccessorArgs placeholder sizes

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder and ttnn-operation-architect instructions
- **Change**: Add explicit documentation: "Interleaved TensorAccessorArgs produces 2 compile-time args. Use `[0, 0]` as placeholder for absent optional tensors."
- **Expected Benefit**: Eliminates confusing compile-time arg offset bugs
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Validate scaler format against helper signatures

- **Type**: new_validation
- **Target**: ttnn-operation-architect
- **Change**: When specifying scaler runtime args, architect must verify the format against the helper's actual parameter type (e.g., `prepare_reduce_scaler<cb_id>(float)` takes plain float, not packed bf16)
- **Expected Benefit**: Prevents scaler encoding mismatches that require kernel-writer fixes
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Fix conftest.py template

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder conftest generation
- **Change**: Use `ttnn.open_device(device_id=0)` in conftest template
- **Expected Benefit**: Eliminates a recurring trivial fix
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Add kernel compile-only smoke test to builder

- **Type**: new_validation
- **Target**: ttnn-generic-op-builder
- **Change**: After creating stub kernels, run a compile-only test (or a minimal kernel load) to validate include paths and syntax before handoff
- **Expected Benefit**: Catches include path errors before kernel writer starts, cleaner handoff
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 6: Require architect to cite analysis documents

- **Type**: instruction_change
- **Target**: ttnn-operation-architect
- **Change**: Architect must explicitly list which findings from each analysis document it incorporated into the design (e.g., "From tilize_analysis.md: adopted stick-to-tile batching pattern for reader")
- **Expected Benefit**: Makes analysis ROI measurable; identifies when analysis is not useful so it can be skipped
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Correct references identified; batch_norm was a good compute-core reference |
| Analysis quality | 3/5 | Analysis docs were produced but architect may not have relied on them heavily |
| Design completeness | 5/5 | Exceptional -- every phase had helper calls with correct policies, CB state tables, and broadcast verification |
| Build correctness | 3/5 | Three infrastructure issues required kernel-writer fixes (include path, placeholder size, scaler encoding) |
| Kernel implementation | 5/5 | All 4 TDD stages passed with zero hard attempts; code is clean and well-structured |
| Inter-agent communication | 4/5 | Design-to-implementation handoff was excellent; build-to-TDD handoff had 3 minor issues |
| Logging/observability | 4/5 | Breadcrumbs had timestamps and covered key events; builder execution log was detailed but handoff notes lagged fixes |
| Helper usage compliance | 5/5 | 100% helper compliance -- every applicable phase uses the correct helper with no redundant CB operations |

### Top 3 Things to Fix

1. **Fix TensorAccessor include path in builder instructions** -- This is a recurring issue across multiple pipeline runs. A 30-second instruction fix eliminates a retry every time.
2. **Document TensorAccessorArgs placeholder sizes per memory layout** -- The interleaved=2 args behavior is undocumented and causes confusing offset bugs.
3. **Validate scaler format in architect design against helper signatures** -- The packed bf16 vs raw float32 confusion appeared in the design doc and cascaded to the builder and kernel writer.

### What Worked Best

The architect's design document was the single strongest aspect of this pipeline run. Its detail level -- with explicit helper calls including policy template parameters, CB state tables between phases, binary op broadcast verification, and the rsqrt post-op lambda pattern -- enabled the kernel writer to translate the design almost mechanically into correct code. This is evidenced by zero hard attempts across all 4 TDD stages, which is the best possible outcome for a non-trivial 9-phase compute kernel. This run demonstrates that when the design is complete and correct, the downstream phases become near-deterministic.

---

## 10. Helper Usage Audit

### Available Helpers

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `tilize_helpers.hpp` | `tilize<Wt, icb, ocb>(num_blocks)` | YES -- input is RM, needs tilize for compute |
| `untilize_helpers.hpp` | `untilize<Wt, icb, ocb>(num_blocks)` | YES -- output is RM, needs untilize after compute |
| `reduce_helpers_compute.hpp` | `reduce<PoolType, ReduceDim, Policy>(icb, scaler_cb, ocb, shape)` | YES -- mean and variance require row reduction |
| `reduce_helpers_dataflow.hpp` | `prepare_reduce_scaler<cb_id>(float)` | YES -- generates scaler tile for reduce ops |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()`, `square<>()` | YES -- centering (sub), normalize (mul), affine (mul, add), variance (square) |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT`, `get_dest_limit()` | YES (indirect) -- used by other helpers for DEST capacity |
| `copy_tile_helpers.hpp` | `copy_tiles()` | NO -- no tile copy needed in this operation |
| `cb_helpers.hpp` | `get_full_tile_size_impl()` | NO -- tile sizes are known from CB format |
| `l1_helpers.hpp` | `addr_to_l1_ptr()` | YES (indirect) -- used by `prepare_reduce_scaler` |
| `common_types.hpp` | `BroadcastDim`, `BinaryInputPolicy`, `BinaryInputBlockShape` | YES -- types used by binary_op_helpers |

### Per-Phase Helper Compliance

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| reader | Scaler generation | `prepare_reduce_scaler<cb_scaler>()` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_conv.f)` | Correct | Helper used correctly |
| reader | Epsilon generation | `prepare_reduce_scaler<cb_eps>()` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_conv.f)` | Correct | Helper used correctly |
| reader | Input stick reads | TensorAccessor raw reads | `noc_async_read_page` via TensorAccessor | Correct (Raw Justified) | No helper for RM stick reads; TensorAccessor is the API |
| reader | Gamma/beta replication | Raw 32-row copy | Raw `noc_async_read_page` in loop | Correct (Raw Justified) | No helper for stick replication |
| compute | Phase 1: Tilize | `tilize<Wt, cb_in, cb_tilized>(1)` | `compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1)` | Correct | Exact match to design |
| compute | Gamma tilize | `tilize<Wt, cb_gamma_rm, cb_gamma>(1)` | `compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma>(1)` | Correct | Exact match |
| compute | Beta tilize | `tilize<Wt, cb_beta_rm, cb_beta>(1)` | `compute_kernel_lib::tilize<Wt, cb_beta_rm, cb_beta>(1)` | Correct | Exact match |
| compute | Phase 2: Reduce mean | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` | Correct | Exact match, including policy |
| compute | Phase 3: Subtract mean | `sub<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` | `compute_kernel_lib::sub<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` | Correct | Exact match |
| compute | Phase 4: Square | `square<WaitUpfrontNoPop>` | `compute_kernel_lib::square<WaitUpfrontNoPop>` | Correct | Exact match |
| compute | Phase 5: Reduce var | `reduce<SUM, REDUCE_ROW>` | `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` | Correct | Default policy |
| compute | Phase 6: Add eps + rsqrt | `add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>` + rsqrt post-op | `compute_kernel_lib::add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>` with lambda | Correct | Including rsqrt post-op |
| compute | Phase 7: Mul inv_std | `mul<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` | `compute_kernel_lib::mul<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` | Correct | Exact match |
| compute | Phase 8a: Mul gamma | `mul<NONE, WaitAndPopPerTile, WaitUpfrontNoPop>` | `compute_kernel_lib::mul<NONE, WaitAndPopPerTile, WaitUpfrontNoPop>` | Correct | Exact match |
| compute | Phase 8b: Add beta | `add<NONE, WaitAndPopPerTile, WaitUpfrontNoPop>` | `compute_kernel_lib::add<NONE, WaitAndPopPerTile, WaitUpfrontNoPop>` | Correct | Exact match |
| compute | Phase 9: Untilize | `untilize<Wt, cb_final, cb_out>(1)` | `compute_kernel_lib::untilize<Wt, cb_X, cb_out>(1)` | Correct | CB varies by affine config via constexpr-if |
| writer | Output stick writes | TensorAccessor raw writes | `noc_async_write_page` via TensorAccessor | Correct (Raw Justified) | No helper for RM stick writes |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | 17 |
| Phases using helpers correctly | 14 (Correct) |
| Phases with justified raw code | 3 (Correct -- reader stick I/O, gamma/beta replication) |
| Phases with missed helpers | 0 |
| Phases with misused helpers | 0 |
| **Helper compliance rate** | **100%** |

### Redundant CB Operations Around Helpers

No redundant CB operations detected around helper calls. The compute kernel contains zero raw `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`, `tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, or `tile_regs_release` calls. All CB synchronization is handled internally by the helpers. The writer kernel uses `cb_wait_front` and `cb_pop_front` on `cb_out`, which is correct since it is the consumer endpoint (not wrapped around a helper).

### Missed Helper Opportunities

All available helpers were used correctly. No missed opportunities.
