# Self-Reflection: softmax

## Metadata
| Field | Value |
|-------|-------|
| Operation | `softmax` |
| Operation Path | `ttnn/ttnn/operations/softmax` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd (x2 launches) |
| Total Git Commits | 11 (this pipeline run) |
| Total Pipeline Duration | ~102 minutes (09:23 -- 11:05 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~2m (inferred) | Completed | Identified 3 references: tt-train softmax, reduce_w, reduce_h |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~31m | Completed | 3 analyses in parallel; reduce_h took longest (~31m). Total ~86KB of analysis output. |
| 2: Design | ttnn-operation-architect | ~5m | Completed | Hybrid design leveraging all 3 references. Streaming 3-pass approach. 4 TDD stages defined. |
| 3: Build | ttnn-generic-op-builder | ~11m | Completed | Stubs + infrastructure. 1 build + 2 test runs (tensor_accessor.hpp include fix). |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~49m | Completed | 4 stages passed. softmax_dim_w required ~20m of debugging (hangs + CB sizing deadlock). Agent relaunched once during this stage. |
| 5: Report | orchestrator | ~2m | Completed | REPORT.md generated. |
| **Total** | | **~102m** | | Earliest Phase 1 start: 09:23 UTC. Latest Phase 5 end: 11:05 UTC. |

### Agent Duration Breakdown

Duration calculated from breadcrumb `"event":"start"` and `"event":"complete"` timestamps.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (reduce_h) | 09:23:05 | 09:53:55 | ~31m | 0 | ~31m active (file reads + DeepWiki + writing) |
| ttnn-operation-analyzer (tt-train) | 09:23:05 | 09:40:28 | ~17m | 0 | ~17m active |
| ttnn-operation-analyzer (reduce_w) | 09:23:10 | 09:37:57 | ~15m | 0 | ~15m active |
| ttnn-operation-architect | 09:52:44 | 09:57:29 | ~5m | 0 | ~5m active (read 3 analyses + helper audit + design) |
| ttnn-generic-op-builder | 09:59:41 | 10:10:57 | ~11m | 1 | ~8m active, ~3m fixing tensor_accessor.hpp compile error |
| ttnn-kernel-writer-tdd (launch 1) | 10:13:06 | ~10:28:00 | ~15m | 3 (2 compile, 2 hangs) | ~5m productive, ~10m debugging hangs in softmax_dim_w |
| ttnn-kernel-writer-tdd (launch 2) | 10:30:39 | 11:02:19 | ~32m | 1 (1 hang) | ~22m productive, ~10m debugging CB deadlock |

**Duration calculation method**: Breadcrumb start/complete events used as primary source. Git commit timestamps used as cross-check. For launch 1 of kernel-writer, the `complete` event is absent (agent was relaunched); end time estimated from last breadcrumb before relaunch (H3 hypothesis at 10:27:53).

### Duration Visualization

```
Phase 0  |#|                                                   (~2m)
Phase 1  |################|                                    (~31m) 3 analyzers in parallel
Phase 2                   |###|                                (~5m)
Phase 3                       |######|                         (~11m)
Phase 4                              |#########################| (~49m) -- 2 kernel-writer launches
Phase 5                                                       |#| (~2m)
         0    10   20   30   40   50   60   70   80   90  100 min

Longest phase: Phase 4 (49m) -- softmax_dim_w stage consumed ~20m debugging 3 hangs
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~33m | 32% | 3 analyzers ran in parallel; wall time dominated by reduce_h (~31m) |
| Design (Phase 2) | ~5m | 5% | Fast -- architect had strong analysis input |
| Build (Phase 3) | ~11m | 11% | Includes one build_metal.sh + tensor_accessor fix |
| Kernel implementation (Phase 4) | ~49m | 48% | 4 TDD stages, 2 kernel-writer launches |
| -- Productive coding | ~27m | 26% | Stages 1, 2, 4 clean + writing Stage 3 code |
| -- Debugging/retries | ~22m | 22% | Stage 3 (softmax_dim_w): 2 compile errors, 3 hangs |
| Reporting (Phase 5) | ~2m | 2% | |
| **Total** | **~102m** | **100%** | |

---

## 2. What Went Well

### 1. Helper library usage was comprehensive and correct

**Phase/Agent**: Phase 2 (Architect) + Phase 4 (Kernel Writer)
**Evidence**: The final compute kernel (`compute_softmax.cpp`) uses `reduce<MAX>`, `reduce<SUM>`, `sub<COL/ROW>`, `mul<COL/ROW>`, and `copy_tiles` -- all from the helper library. The reader uses `prepare_reduce_scaler` from `reduce_helpers_dataflow.hpp`. Zero raw tile_regs_acquire/commit/wait/release calls in the final code. 6 out of 6 compute phases use helpers (see Section 10).
**Why it worked**: The architect explicitly analyzed the helper library (breadcrumb at 09:53:15-09:53:20) and mapped each softmax phase to a specific helper call in the design document. The kernel writer followed these mappings faithfully.

### 2. Three of four TDD stages passed on first attempt

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: `data_pipeline` (1 attempt), `exp_passthrough` (1 attempt), `softmax_dim_h` (1 attempt). Only `softmax_dim_w` required debugging. The `.tdd_state.json` shows `attempts: 0` for all stages (the counter was not incremented by the kernel writer -- however the breadcrumbs confirm single-attempt success for 3 of 4 stages).
**Why it worked**: The incremental TDD stage plan was well-designed. Stages 1 and 2 isolated the data pipeline and exp functionality, building confidence before the complex 3-pass softmax in Stage 3. Stage 4 then leveraged all patterns from Stage 3 and passed cleanly.

### 3. Architect produced high-quality, implementable design in 5 minutes

**Phase/Agent**: Phase 2 (ttnn-operation-architect)
**Evidence**: 5-minute wall time. Design document covers: CB layout, kernel argument tables, binary op broadcast verification, TDD stage plan with exact helper call signatures and CB state transitions per phase, critical notes on multi-pass synchronization and persistent CB patterns. The kernel writer followed the design almost exactly -- the only deviation was CB_EXP sizing (2 pages in design, changed to Wt/Ht pages at runtime).
**Why it worked**: The architect had 3 high-quality analysis documents (~86KB) covering the exact patterns needed. The analysis-to-design pipeline worked as intended.

### 4. Builder handoff notes were detailed and accurate

**Phase/Agent**: Phase 3 (ttnn-generic-op-builder)
**Evidence**: Execution log Section 6 (Handoff Notes) includes: CB indices, work distribution details, compile-time vs runtime arg mapping, special considerations for prepare_reduce_scaler, TensorAccessorArgs start indices, and known limitations. The kernel writer did not produce any upstream_feedback complaints about the builder's output.
**Why it worked**: The builder correctly interpreted the architect's design and translated it into a working ProgramDescriptor with proper split_work_to_cores, dual-group compute kernel descriptors, and dynamic CB sizing.

### 5. dim=-2 leveraged all learnings from dim=-1

**Phase/Agent**: Phase 4, Stage softmax_dim_h
**Evidence**: Stage 4 passed on first attempt with all 4 shapes, taking ~21m wall time but with zero debugging cycles. The compute kernel's dim=-2 path mirrors the dim=-1 path exactly, substituting REDUCE_COL for REDUCE_ROW and BroadcastDim::ROW for BroadcastDim::COL. The CB_EXP fix from Stage 3 (using Ht pages for dim=-2) was already applied.
**Why it worked**: The TDD stage ordering was correct -- implementing dim=-1 first established all the patterns, and dim=-2 was purely mechanical substitution.

---

## 3. Issues Found

### Issue 1: CB_EXP (c_25) sizing caused tile_regs deadlock on multi-tile shapes

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w |
| Agent | ttnn-operation-architect (design) + ttnn-kernel-writer-tdd (debugging) |
| Retries Consumed | 1 hard attempt (hang on (1,1,64,128)) |
| Time Cost | ~8m (from first hang at 10:33:40 to fix at 10:38:27) |

**Problem**: The architect designed CB_EXP (c_25) with 2 pages (double-buffered). At runtime with Wt=4, the sub helper (Phase 2a) produces 4 tiles into c_25, but can only fit 2 before blocking on `cb_reserve_back`. Meanwhile, `tile_regs_wait` forces the unpack thread to synchronize with the pack thread, preventing the reduce helper (Phase 2b) from starting to consume c_25 tiles until Phase 2a completes. This creates a circular dependency: pack waits for CB space, but unpack (which would consume via the next phase) waits for pack to release tile_regs. The result is a deadlock that only manifests when Wt > 2 (or Ht > 2 for dim=-2).

The first shape (1,1,32,32) with Wt=1 passed because 1 tile fits in 2 pages. The second shape (1,1,64,128) with Wt=4 hung.

**Root Cause**: The architect did not account for the TRISC tile_regs synchronization constraint when sizing c_25. The design says "2 pages (double-buffered)" but the sub and reduce helpers that produce and consume c_25 run in the same TRISC execution context. The sub helper must complete ALL Wt tiles before reduce can start consuming them, so c_25 must hold the full row/column.

This is a design-time issue, not a kernel-writing issue. The information about TRISC synchronization is available in the helper library documentation (reduce_helpers_compute.hpp mentions tile_regs_acquire/commit/wait/release), but the implication for CB sizing between consecutive helpers was not obvious.

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "When two helpers produce/consume the same intermediate CB in sequence within the same work unit (e.g., sub -> reduce on c_25), the intermediate CB MUST hold the full block size (Wt tiles for REDUCE_ROW, Ht tiles for REDUCE_COL), not just 2 pages. TRISC tile_regs synchronization prevents interleaved execution."
- **ttnn-kernel-writer-tdd**: When encountering a deadlock that passes for Wt=1 but fails for Wt>1, the first hypothesis should be "intermediate CB is too small for the tile_regs synchronization pattern."

### Issue 2: binary_op_init_common in design caused device hangs

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w |
| Agent | ttnn-operation-architect (design) + ttnn-kernel-writer-tdd (debugging) |
| Retries Consumed | 2 hard attempts (hangs) |
| Time Cost | ~6m (from first hang at 10:24:30 to agent relaunch at 10:30:39) |

**Problem**: The architect's design (op_design.md, Part 2, "Compute Kernel" section) says: "Startup: `compute_kernel_hw_startup(c_0, c_1, c_16)` then `binary_op_init_common(c_0, c_24, c_16)`". The kernel writer included this `binary_op_init_common` call. The device hung with TRISC0=UABD TRISC1=MWDD TRISC2=K state.

Hypothesis H2 (medium confidence): binary_op_init_common corrupts hardware state for the reduce helper that runs first. The kernel writer removed it, but the hang persisted (same TRISC state). The real issue turned out to be related (namespace/policy issues), but removing `binary_op_init_common` was the correct fix since binary helpers handle their own initialization internally.

**Root Cause**: The architect incorrectly included `binary_op_init_common` as a required startup call. The binary op helpers (sub, mul) have `init=true` as their default template parameter and call `binary_op_init_common` internally. Calling it explicitly at the top of the kernel with different CB arguments than what the first actual operation uses causes state corruption.

**Fix for agents**:
- **ttnn-operation-architect**: Remove `binary_op_init_common` from design templates when using binary_op_helpers.hpp. The helpers manage their own initialization. Add to design checklist: "binary_op helpers have init=true by default -- do NOT call binary_op_init_common separately."
- **ttnn-kernel-writer-tdd**: If the design says to call `binary_op_init_common` but helpers are being used, skip it. Add a validation: "binary_op helpers handle their own init; explicit binary_op_init_common is only needed for raw binary operations."

### Issue 3: Namespace qualification confusion on NoAccumulation and binary_op_init_common

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 2 free retries (compile errors) |
| Time Cost | ~3m (10:21:08 to 10:22:52) |

**Problem**: First attempt used `NoAccumulation` without namespace qualification. Compiler suggested `compute_kernel_lib::NoAccumulation`. Second attempt over-corrected by also adding `compute_kernel_lib::` prefix to `binary_op_init_common`, which is actually in the `ckernel` (global) namespace.

**Root Cause**: The kernel writer lacked a reference table of which APIs live in which namespace. `NoAccumulation` is in `compute_kernel_lib::`, but `binary_op_init_common` is in the global/ckernel namespace. The compiler error message was clear for the first error but the writer generalized incorrectly.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Add a quick-reference note: "Helper library types (NoAccumulation, ReduceInputBlockShape, BinaryInputBlockShape, etc.) use `compute_kernel_lib::` namespace. Low-level LLK functions (binary_op_init_common, reduce_init, etc.) are in global/ckernel namespace. When a compile error says 'not declared in this scope', check the specific namespace -- do not blindly add compute_kernel_lib:: to all identifiers."

### Issue 4: tensor_accessor.hpp include path in builder stubs

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry (compile error) |
| Time Cost | ~3m (10:04:01 to 10:07:45) |

**Problem**: The builder generated kernel stubs with `#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` which does not exist. TensorAccessor is available through `api/dataflow/dataflow_api.h`. The builder documented this as upstream feedback for the pipeline.

**Root Cause**: The builder's instruction set or helper-to-include mapping table contains an incorrect include path for TensorAccessor. This same issue was flagged in the builder's execution log Section 7.

**Fix for agents**:
- **ttnn-generic-op-builder**: Remove or correct the TensorAccessor include path in the helper-to-include mapping. TensorAccessor does not need an explicit include in kernel stubs -- it is available via `api/dataflow/dataflow_api.h`.

### Issue 5: Kernel writer was relaunched mid-stage (softmax_dim_w)

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w |
| Agent | orchestrator / ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt (full agent relaunch) |
| Time Cost | ~3m (10:28 to 10:31, gap between launches) |

**Problem**: The kernel writer's first launch hit 3 consecutive failures in softmax_dim_w (2 compile errors + 2 hangs). At 10:27:53 it logged hypothesis H3 but then the agent was relaunched at 10:30:39 with a fresh start. The new launch had to re-read the design, re-implement the kernels, and re-discover the BinaryInputPolicy fix. The successful fix (CB sizing) came from the second launch.

The breadcrumb log shows a discontinuity: the first launch ends with H3 hypothesis, then a new `"event":"start"` appears at 10:30:39. This indicates the orchestrator killed and relaunched the kernel writer.

**Root Cause**: Likely the orchestrator's retry/timeout mechanism triggered after consecutive hard failures. The relaunch lost context from the first session's debugging (H1, H2, H3 hypotheses). However, the fresh start also meant the second launch approached the problem cleanly, which ultimately led to the correct BinaryInputPolicy pattern.

**Fix for agents**:
- **Orchestrator**: When relaunching a kernel writer mid-stage, pass a summary of the previous session's hypotheses and fixes as context. This prevents re-discovering the same issues.
- **ttnn-kernel-writer-tdd**: On relaunch, read the existing breadcrumbs from the previous session to avoid duplicate work.

### Issue 6: TDD stage test files not created by architect

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2 (Architect) -> Phase 3 (Builder) |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 (builder compensated) |
| Time Cost | ~2m (builder had to create test files) |

**Problem**: The architect registered 4 TDD stages in `.tdd_state.json` but did not create the corresponding `test_stage_*.py` files. The builder flagged this as a deviation (breadcrumb at 10:02:34: "Architect created .tdd_state.json but not test_stage_*.py files") and created them itself.

**Root Cause**: The architect's instructions likely do not include creating test files -- that responsibility falls to the builder. However, the builder was briefly confused because the `.tdd_state.json` referenced test files that did not yet exist.

**Fix for agents**:
- **ttnn-operation-architect**: Either create the test files (preferred -- the architect has the shape/tolerance/reference data) or explicitly note in the handoff that test files still need to be created by the builder.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~3m | 0 free, 0 hard | PASS | Clean -- reader/writer/compute all correct on first try |
| exp_passthrough | ~1m | 0 free, 0 hard | PASS | Clean -- trivial delta from previous stage (add exp post_op) |
| softmax_dim_w | ~20m | 2 free, 3 hard | PASS | 2 compile errors (namespace), 3 hangs (binary_op_init, BinaryInputPolicy, CB sizing deadlock). Agent relaunched once. |
| softmax_dim_h | ~21m | 0 free, 0 hard | PASS | Clean first-attempt pass. ~21m wall time likely includes reading/implementing the full dim=-2 path + waiting for 4-shape test suite (6.60s test runtime). |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | softmax_dim_w debugging | ttnn-kernel-writer-tdd | ~20m | 20% | 2 compile errors + 3 hangs + 1 agent relaunch. CB deadlock was hardest to diagnose. | 5 | Architect's CB sizing error + binary_op_init_common in design + namespace unfamiliarity |
| 2 | Analysis phase wall time | ttnn-operation-analyzer | ~31m | 30% | 3 analyzers ran in parallel but reduce_h took 31m (longest). Dominated by DeepWiki queries and comprehensive analysis writing. | 0 | Analysis thoroughness -- not necessarily wasteful, but long. |
| 3 | Build phase | ttnn-generic-op-builder | ~11m | 11% | Includes one build_metal.sh run + tensor_accessor.hpp fix + re-test. | 1 | tensor_accessor.hpp incorrect include path |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd (launch 1) | Full implementation of softmax_dim_w with compile fixes and hang debugging | Agent was relaunched; second launch had to re-implement. However, some fixes (removing binary_op_init_common) carried over since kernel files persisted on disk. | Pass previous session hypothesis summary to relaunched agent |
| ttnn-kernel-writer-tdd (launch 1) | Added binary_op_init_common per design doc | Had to be removed -- helpers handle their own init | Architect should not include it in designs using helpers |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | `softmax_tt_train_analysis.md`, `reduce_w_analysis.md`, `reduce_h_analysis.md` |
| Quality | GOOD |
| Issues | None significant. All three analyses were thorough and the architect explicitly cited them. |
| Downstream Impact | Positive -- architect completed design in 5 minutes using the analyses. |
| Suggestion | None needed. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md`, `.tdd_state.json` |
| Quality | ADEQUATE |
| Issues | (1) TDD stage test files not created by architect -- builder had to create them. (2) Design included `binary_op_init_common` as a startup call which was incorrect. (3) CB_EXP sizing of 2 pages was wrong for multi-tile shapes. |
| Downstream Impact | (1) Minor delay (~2m). (2) No direct impact on builder (it only creates stubs). (3) No impact on builder (CB sizing was correctly set to 2 per design; the bug manifested later in kernel writer). |
| Suggestion | Architect should validate CB sizes against tile_regs synchronization constraints. |

### Handoff 3: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, `softmax_program_descriptor.py`, test files, `op_design.md` (Part 2) |
| Quality | GOOD |
| Issues | (1) Stubs initially had incorrect tensor_accessor.hpp include but this was fixed before handoff. (2) CB_EXP was set to 2 pages per architect design -- this was a latent bug passed through. |
| Downstream Impact | (1) None -- fixed before handoff. (2) Required kernel writer to fix program_descriptor.py upstream during Stage 3. |
| Suggestion | Add a cross-validation step between architect's CB design and builder's CB allocation before kernel writing begins. |

### Handoff 4: ttnn-kernel-writer-tdd launch 1 -> launch 2

| Field | Value |
|-------|-------|
| Artifact Passed | Kernel files on disk (partially modified from launch 1 debugging), breadcrumbs from launch 1 |
| Quality | POOR |
| Issues | No explicit handoff -- the second launch started from scratch reading op_design.md. It did not read launch 1's breadcrumbs or hypotheses. The kernel files on disk retained some fixes from launch 1 (binary_op_init_common removal), but the second launch reimplemented anyway. |
| Downstream Impact | ~3m wasted re-reading + re-implementing before getting to the actual unsolved problem (CB deadlock). |
| Suggestion | Orchestrator should pass a "previous session summary" to the relaunched agent, or the agent should read its own prior breadcrumbs on startup. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder | Remove tensor_accessor.hpp from helper-to-include mapping; TensorAccessor is in dataflow_api.h | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Do not include binary_op_init_common when using binary_op helpers -- they manage own init | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | CB sizing for intermediate buffers between consecutive helpers must account for tile_regs sync | HIGH | HIGH |
| ttnn-kernel-writer-tdd | ttnn-kernel-writer-tdd | Add namespace quick-reference: compute_kernel_lib:: for helper types, global/ckernel for LLK functions | MEDIUM | LOW |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD / relaunch | Agent relaunch loses context from previous debugging session | Pass hypothesis summary or breadcrumb digest to relaunched agents | HIGH |
| Design / CB validation | CB sizing error was a design-time issue that only manifested at kernel runtime | Add a design-time validation rule for intermediate CB sizing under tile_regs sync constraints | HIGH |
| Build / includes | tensor_accessor.hpp path does not exist | Fix helper-to-include mapping in builder instructions | MEDIUM |
| Analysis | 31m for 3 parallel analyzers seems long but produced high-quality output | Consider caching analysis results for common reference ops (reduce_w, reduce_h are reused across many ops) | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | No numerical mismatches in this run -- all failures were compile errors or hangs. This is a positive signal for the helper-based approach. |
| 2 | Too many planning stages | N/A (DONE) | Merged planner+designer into architect worked well here (5m design phase). |
| 3 | .tdd_state.json fragility | PARTIALLY | TDD state was used correctly but `attempts` field shows 0 for all stages even though softmax_dim_w had multiple attempts. The kernel writer did not increment the counter. |
| 4 | No fast path for simple ops | NO | Softmax is a complex operation; full pipeline was appropriate. |
| 6 | Builder model choice | MAYBE | Builder hit one compile error (tensor_accessor include). Hard to say if a stronger model would have avoided this. |
| 7 | Discovery keyword matching | NO | Discovery correctly identified relevant references. |
| 9 | No architect/builder cross-validation | YES | CB_EXP sizing error in architect's design was passed through builder unchanged and only caught at kernel runtime. A cross-validation step would have caught this. |
| 11 | No incremental re-run capability | PARTIALLY | Agent was relaunched for softmax_dim_w but had to restart from scratch rather than resuming from the failing point. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| binary_op_init_common in architect designs with helpers | Architect includes explicit binary_op_init_common startup when binary helpers handle their own init internally. Causes device hangs. | HIGH |
| Intermediate CB sizing under tile_regs sync | When consecutive helpers share an intermediate CB (e.g., sub -> reduce on c_25), the CB must hold the full block, not just 2 pages. TRISC tile_regs synchronization prevents interleaved producer/consumer execution. Architect does not validate this. | HIGH |
| Agent relaunch loses debugging context | When the kernel writer is relaunched mid-stage, the new instance does not read prior breadcrumbs or hypotheses, wasting time re-discovering already-explored approaches. | MEDIUM |
| .tdd_state.json attempt counter not updated | The kernel writer does not increment the attempts field in .tdd_state.json, making it impossible to track retry counts from the state file alone. Breadcrumbs are the only reliable source. | LOW |

---

## 8. Actionable Recommendations

### Recommendation 1: Add tile_regs-aware CB sizing validation to architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a design checklist item: "For each intermediate CB that is produced by one helper and consumed by the next helper in sequence (same work unit), the CB MUST hold the full block dimension (Wt tiles for REDUCE_ROW operations, Ht tiles for REDUCE_COL operations). Double-buffering (2 pages) is insufficient because tile_regs synchronization prevents interleaved execution between consecutive helpers."
- **Expected Benefit**: Eliminates the class of deadlocks seen in softmax_dim_w CB_EXP sizing
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Remove binary_op_init_common from helper-based designs

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a note in the compute kernel design template: "When using binary_op_helpers.hpp (sub, mul, add), do NOT call binary_op_init_common at kernel startup. The helpers manage their own initialization (init=true default). Only call binary_op_init_common when using raw binary LLK operations."
- **Expected Benefit**: Eliminates device hangs from hardware state corruption
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Pass debugging context to relaunched kernel writers

- **Type**: pipeline_change
- **Target**: Orchestrator script / ttnn-kernel-writer-tdd
- **Change**: When the orchestrator relaunches a kernel writer for the same stage: (1) Read the previous session's breadcrumbs, (2) Extract hypotheses and their outcomes, (3) Include a "Previous Session Summary" section in the prompt to the new agent instance. Alternatively, have the kernel writer read its own breadcrumb file on startup if it exists and contains entries for the current stage.
- **Expected Benefit**: Avoids ~3-5m of wasted re-discovery per relaunch
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 4: Fix tensor_accessor include in builder instructions

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder prompt/instructions
- **Change**: Remove `tensor_accessor.hpp` from the helper-to-include mapping table. Add a note: "TensorAccessor is available via api/dataflow/dataflow_api.h -- no separate include needed."
- **Expected Benefit**: Eliminates one compile error cycle per builder run
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Cross-validate architect CB design with builder allocation

- **Type**: new_validation
- **Target**: Pipeline between Phase 2 and Phase 3 (or within Phase 3)
- **Change**: After the builder creates the program descriptor, run a static check that compares the architect's CB table (op_design.md Part 1) with the builder's actual CB allocation (program_descriptor.py). Check: CB IDs match, page counts match, data formats match. Flag discrepancies before kernel writing begins.
- **Expected Benefit**: Catches sizing mismatches (like CB_EXP) before expensive kernel debugging
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 6: Ensure kernel writer updates .tdd_state.json attempt counters

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd prompt/instructions
- **Change**: Add instruction: "After each test attempt (pass or fail), increment the `attempts` field for the current stage in `.tdd_state.json`. Also increment `free_retries` for compile errors and log hard attempts separately."
- **Expected Benefit**: Enables accurate retry tracking from TDD state alone, without needing breadcrumbs
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Found 3 highly relevant references. Slight penalty: no scoring of reference quality or deduplication. |
| Analysis quality | 5/5 | Three thorough analyses (~86KB total) that the architect used effectively. All patterns identified correctly. |
| Design completeness | 3/5 | CB layout, kernel args, TDD stages all excellent. Lost points for: incorrect binary_op_init_common, CB_EXP undersized, no test files created. |
| Build correctness | 4/5 | Program descriptor correct except for passing through architect's CB sizing error. One avoidable compile error (tensor_accessor include). |
| Kernel implementation | 4/5 | All 4 stages passed. Helper usage excellent. Lost one point for the 5-attempt debugging cycle in softmax_dim_w which was partly caused by upstream issues. |
| Inter-agent communication | 3/5 | Analysis-to-architect handoff excellent. Architect-to-builder adequate. Builder-to-kernel-writer good. Agent relaunch handoff poor (no context transfer). |
| Logging/observability | 4/5 | Breadcrumbs present for all agents with timestamps. Execution log available for builder. Missing: kernel writer execution log, Phase 0 breadcrumbs. |
| Helper usage compliance | 5/5 | 100% compliance. All 6 compute phases use helpers. No missed opportunities, no misuse, no redundant CB operations. |

### Top 3 Things to Fix

1. **Architect must validate intermediate CB sizing under tile_regs sync constraints.** The CB_EXP deadlock cost 8 minutes and was entirely preventable at design time. This is a systemic issue that will recur for any multi-phase compute kernel using helper sequences.

2. **Architect must not include binary_op_init_common when designs use binary_op helpers.** The helpers manage their own initialization. This caused 6 minutes of hang debugging and is a simple instruction fix.

3. **Agent relaunch must preserve debugging context.** The kernel writer was relaunched mid-stage with zero context from its previous session, wasting ~3 minutes re-discovering already-known information. This compounds with Issue 1 -- if the first launch had the CB sizing insight, the relaunch could have solved the problem immediately.

### What Worked Best

The **analysis-to-design pipeline** was the strongest aspect of this run. Three parallel analyzers produced comprehensive, high-quality analysis documents (~86KB) covering the exact patterns needed for softmax. The architect consumed all three analyses in under 5 minutes and produced a design that the kernel writer followed almost exactly. The helper library mapping in the design was particularly valuable -- it resulted in 100% helper compliance in the final kernel code, with zero numerical debugging issues (a stark contrast to the known Issue #1 about numerical debugging burning context). The helper-first approach completely eliminated the hardest class of bugs.

---

## 10. Helper Usage Audit

### Available Helpers

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `reduce_helpers_compute.hpp` | `reduce<PoolType, ReduceDim, Policy, ReconfigMode>(...)` | YES -- used for MAX reduce (find row/col max) and SUM reduce (sum exp tiles) |
| `reduce_helpers_dataflow.hpp` | `prepare_reduce_scaler<cb_id>(float)`, `calculate_and_prepare_reduce_scaler<...>()` | YES -- used for scaler tile generation in reader |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()`, `square<>()` | YES -- used for sub(x, max) and mul(exp, recip) |
| `copy_tile_helpers.hpp` | `copy_tiles(cb_in, cb_out, n, post_op)` | YES -- used for identity passthrough (stage 1) and exp passthrough (stage 2) |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT`, `get_dest_limit()`, `get_fp32_dest_acc_enabled()` | YES (implicitly) -- used by reduce and binary helpers internally |
| `cb_helpers.hpp` | `get_full_tile_size<>()`, `get_cb_num_pages()` | NO -- not directly needed; tile sizes handled via standard APIs |
| `tilize_helpers.hpp` | `tilize<>()` | NO -- input is already TILE_LAYOUT |
| `untilize_helpers.hpp` | `untilize<>()` | NO -- output is TILE_LAYOUT |
| `l1_helpers.hpp` | L1 memory utilities | NO -- not needed for this operation |
| `common_types.hpp` | Shared type definitions (BroadcastDim, BinaryInputPolicy, etc.) | YES (implicitly) -- types used throughout compute kernel |

### Per-Phase Helper Compliance

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| compute | Phase 1 (max reduce) dim=-1 | `reduce<MAX, REDUCE_ROW>` | `reduce<MAX, REDUCE_ROW, WaitAndPopPerTile, INPUT_AND_OUTPUT>` | Correct | Exact match to design |
| compute | Phase 2a (sub+exp) dim=-1 | `sub<COL>` with exp post_op | `sub<COL, WaitAndPopPerTile, WaitUpfrontNoPop>` with exp lambda | Correct | Design specified NoWaitNoPop for B; implementation uses WaitUpfrontNoPop (better -- explicit wait on first use) |
| compute | Phase 2b (sum+recip) dim=-1 | `reduce<SUM, REDUCE_ROW>` with recip post_op | `reduce<SUM, REDUCE_ROW, WaitAndPopPerTile, INPUT_AND_OUTPUT>` with recip lambda | Correct | Exact match to design |
| compute | Phase 3a (sub+exp) dim=-1 | `sub<COL>` with exp post_op | `sub<COL, WaitAndPopPerTile, NoWaitNoPop>` with exp lambda | Correct | NoWaitNoPop for B (c_24 already waited in Phase 2a) |
| compute | Phase 3b (mul recip) dim=-1 | `mul<COL>` | `mul<COL, WaitAndPopPerTile, WaitUpfrontNoPop>` | Correct | Correct policy for c_26 (freshly pushed in Phase 2b) |
| compute | Phases 1-3 dim=-2 | Same as dim=-1 with REDUCE_COL + ROW broadcast | REDUCE_COL + BroadcastDim::ROW | Correct | Clean mirror of dim=-1 |
| reader | Scaler generation | `prepare_reduce_scaler<c_1>(1.0f)` | `dataflow_kernel_lib::prepare_reduce_scaler<c_1>(1.0f)` | Correct | Exact match |
| reader | Data reads | Raw TensorAccessor reads (no helper exists) | Raw `noc_async_read` with TensorAccessor | Raw Justified | No helper for multi-pass DRAM streaming -- raw code is correct |
| writer | Data writes | Raw TensorAccessor writes (no helper exists) | Raw `noc_async_write` with TensorAccessor | Raw Justified | No helper for DRAM writes -- raw code is correct |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | 12 (6 per dimension for compute) + 2 (reader scaler + data) + 2 (writer data) = 16 |
| Phases using helpers correctly | 12 (all compute phases + reader scaler) |
| Phases with justified raw code | 4 (reader data reads x2 dims, writer data writes x2 dims) |
| Phases with missed helpers | 0 |
| Phases with misused helpers | 0 |
| **Helper compliance rate** | **100%** |

### Redundant CB Operations Around Helpers

No redundant CB operations detected around helper calls. The compute kernel does not wrap any helper calls with explicit `cb_reserve_back`, `cb_push_back`, `tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, or `tile_regs_release`. The only explicit CB operations are the two `cb_pop_front` calls for c_24 and c_26 at the end of each work unit, which are legitimately required because these CBs use NoPop policies in the helpers and need manual cleanup.

### Missed Helper Opportunities

All available helpers were used correctly. No missed opportunities.
