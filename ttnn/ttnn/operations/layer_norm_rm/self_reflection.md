# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 10 (this run: eab6422 through 4027efa) |
| Total Pipeline Duration | ~51 minutes (13:35 - 14:27 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m | SUCCESS | 3 references identified: tilize (input_stage), untilize (output_stage), batch_norm (compute_core) |
| 1: Analysis | 3x ttnn-operation-analyzer | ~8m (13:35-13:43) | SUCCESS | Ran in parallel. Untilize finished first (~4m), tilize second (~6m), batch_norm last (~8m) |
| 2: Design | ttnn-operation-architect | ~7m (13:45-13:52) | SUCCESS | Hybrid mode detected. Produced op_design.md with 15 CBs, 5 TDD stages, detailed per-phase helper specifications |
| 3: Build | ttnn-generic-op-builder | ~7m (13:54-14:01) | SUCCESS | Created Python infra, 3 stub kernels, 6 test files. All import/validation tests passed. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~21m (14:03-14:24) | SUCCESS | All 5 stages passed on first attempt, 0 retries. 25/25 test cases passed. |
| 5: Report | orchestrator | ~3m (14:24-14:27) | SUCCESS | REPORT.md generated |
| **Total** | | **~52m** | | Earliest start: 13:35, latest end: 14:27 |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (untilize) | 13:36:02 | 13:39:18 | 3m 16s | 0 | ~3m active |
| ttnn-operation-analyzer (tilize) | 13:35:36 | 13:41:23 | 5m 47s | 0 | ~6m active |
| ttnn-operation-analyzer (batch_norm) | 13:38:50 | 13:43:14 | 4m 24s | 0 | ~4m active |
| ttnn-operation-architect | 13:45:01 | 13:51:45 | 6m 44s | 0 | ~7m active |
| ttnn-generic-op-builder | 13:54:24 | 14:01:22 | 6m 58s | 0 | ~7m active |
| ttnn-kernel-writer-tdd | 14:03:47 | 14:24:22 | 20m 35s | 0 | ~21m active, 0m debugging |

**Duration calculation method**: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps used for all agents. All agents had both start and complete events.

### Duration Visualization

```
Phase 0  |#|                                            (~1m)
Phase 1  |########|                                     (~8m) 3 analyzers in parallel
Phase 2       |#######|                                 (~7m)
Phase 3             |#######|                           (~7m)
Phase 4                     |#####################|     (~21m) -- longest
Phase 5                                            |###| (~3m)
         0    5    10   15   20   25   30   35   40   50 min

Longest phase: Phase 4 (21m) -- kernel implementation across 5 TDD stages, all clean
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~9m | 17% | 3 parallel analyzers |
| Design (Phase 2) | ~7m | 13% | Single architect |
| Build (Phase 3) | ~7m | 13% | Single builder |
| Kernel implementation (Phase 4) | ~21m | 40% | 5 TDD stages |
| -- Productive coding | ~21m | 40% | All stages first-attempt pass |
| -- Debugging/retries | 0m | 0% | No debugging cycles |
| Reporting (Phase 5) | ~3m | 6% | |
| Inter-phase gaps | ~5m | 10% | Agent spawn/teardown overhead |
| **Total** | **~52m** | **100%** | |

---

## 2. What Went Well

### 1. Zero retries across all 5 TDD stages

**Phase/Agent**: Phase 4 -- ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `"attempts": 0, "free_retries": 0` for all 5 stages. Breadcrumbs confirm: every `test_run` event has `"status":"pass"` and `"hard_attempts":0`. The git history shows exactly 5 kernel commits (one per stage), with zero fixup or WIP commits.
**Why it worked**: The architect's design document was exceptionally detailed -- it specified every compute phase with exact helper calls, CB policies (WaitUpfrontNoPop, NoWaitNoPop, etc.), manual pop requirements, CB state tables after each phase, and binary broadcast dimension verification. The kernel writer could essentially translate the design into code. The helper library's clean API (binary_op helpers with typed policies, reduce helpers with input block shapes) eliminated ambiguity about CB synchronization.

### 2. Architect's CB layout required zero corrections

**Phase/Agent**: Phase 2 (Architect) and Phase 4 (Kernel Writer)
**Evidence**: The design specified 15 CBs with exact page counts and lifetime annotations. The final kernel code uses exactly these CBs with exactly these sizes. The builder's `program_descriptor.py` allocated CBs matching the design, and the kernel writer consumed them without any CB-related bugs. No CB sizing mismatches, no deadlocks, no incorrect page counts.
**Why it worked**: The architect built CB state tables after each phase (e.g., "CB state after Phase 2: c_1: Wt tiles, waited, not popped") that traced tile lifetimes through the full compute pipeline. This made double-buffering requirements (c_1 at 2*Wt, c_25 at 2*Wt) explicit rather than leaving them for the kernel writer to figure out.

### 3. Helper library achieved full coverage of compute phases

**Phase/Agent**: Phase 4 -- kernel implementation
**Evidence**: The compute kernel uses exactly 4 helper headers: `tilize_helpers.hpp`, `untilize_helpers.hpp`, `reduce_helpers_compute.hpp`, `binary_op_helpers.hpp`. Every compute phase (tilize, reduce, sub, square, add+rsqrt, mul, untilize) maps to a helper call. Only two raw `cb_pop_front` calls exist, and these are for the explicit manual-pop pattern documented in the design (c_1 after Phase 3, c_25 after Phase 7). See Section 10 for full audit.
**Why it worked**: The architect's design validated each phase against the helper library's API before handing off, specifying exact template parameters and policy enums.

### 4. Clean inter-phase handoffs

**Phase/Agent**: All transitions
**Evidence**: No `upstream_feedback` breadcrumb events from the architect or builder. The kernel writer had exactly 1 upstream fix: a minor TensorAccessorArgs placeholder correction (`[0]` changed to `[0,0]`). No design ambiguity complaints, no missing information.
**Why it worked**: Each artifact was complete and self-contained. The architect's `op_design.md` had two clear parts (Part 1: Architecture for builder, Part 2: Kernel Implementation for kernel writer). The builder produced correct infrastructure on the first try.

### 5. Analysis phase was well-scoped

**Phase/Agent**: Phase 1 -- 3 parallel analyzers
**Evidence**: Each analyzer was given a specific role directive (input_stage, output_stage, compute_core). The analyses total approximately 3 focused documents rather than one monolithic analysis. The architect breadcrumbs show all 3 references were used ("mode":"Hybrid", "references":["tilize_analysis.md","untilize_analysis.md","batch_norm_analysis.md"]").
**Why it worked**: Role-scoping kept each analysis focused and reduced total context for the architect.

---

## 3. Issues Found

### Issue 1: Design document contains unresolved deliberation text

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2 (Design) |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 |
| Time Cost | 0 (kernel writer navigated it successfully) |

**Problem**: The design document `op_design.md` Part 2, Stage 2 (reduce_mean) contains visible deliberation text showing the architect thinking out loud about implementation alternatives:

> "For testability, output the mean expanded: use sub(input, mean, output) but negate, or better: just skip sub and output the mean broadcast across W using the add helper with broadcast COL, adding mean to a zero input. Simplest: compute tilizes input, reduces to mean in c_24, then for each of Wt output tiles, copies the mean tile."

This paragraph explores 4 different approaches in-line without resolving to a clear recommendation. The kernel writer ignored this deliberation and used a `unary_bcast COL` approach (per breadcrumb: "Used unary_bcast COL instead of binary_op for mean broadcast"). The kernel writer's approach was simpler and correct, but the deliberation text could have confused a less capable model.

**Root Cause**: The architect did not clean up exploratory reasoning before committing the design. The final approach ("Use NoWaitNoPop on c_24 to read it Wt times without popping") was buried at the end of the paragraph rather than stated clearly at the top.

**Fix for agents**:
- **ttnn-operation-architect**: Add an instruction to "resolve all alternatives before writing the final design. If multiple approaches were considered, state only the chosen approach in the design. Move deliberation to a 'Design Notes' appendix if needed for context."

### Issue 2: Design doc compute CT args mismatch with implementation

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2 (Design) vs Phase 3 (Build) vs Phase 4 (TDD) |
| Agent | ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Retries Consumed | 0 |
| Time Cost | 0 (silently resolved) |

**Problem**: The design document specifies compute compile-time args as `[Wt, num_tile_rows, has_gamma, has_beta]` (4 args, index 0-3). The builder made `num_tile_rows` a runtime arg instead of a CT arg (see `program_descriptor.py` lines 395-399, where compute CT args are `[Wt, has_gamma, has_beta]` -- only 3 args). The kernel writer then used `get_compile_time_arg_val(0)` for Wt, `get_compile_time_arg_val(1)` for has_gamma, `get_compile_time_arg_val(2)` for has_beta, and `get_arg_val<uint32_t>(0)` for num_tile_rows as a runtime arg.

This means the indices shifted: the design says `has_gamma` is CT arg 2, but the actual code has it at CT arg 1. The builder correctly realized that `num_tile_rows` cannot be a compile-time arg when it differs per core group, and silently corrected this. The kernel writer then matched the builder's actual CT arg order.

This worked fine because everyone adapted, but it represents a design-to-implementation drift that went undocumented.

**Root Cause**: The architect did not consider that `num_tile_rows` varies per core (cliff cores get fewer tile-rows), making it unsuitable for a compile-time arg. The design says "Compute Compile-Time Args: Index 1: num_tile_rows" which is architecturally wrong for multi-core work distribution.

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "Any argument that varies per core MUST be a runtime arg, not a compile-time arg. Review work distribution to identify which values differ across cores."

### Issue 3: TensorAccessorArgs placeholder size was wrong in builder output

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) -> Phase 4 Stage 1 (data_pipeline) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (fixed by kernel writer before first test) |
| Time Cost | ~1m |

**Problem**: The builder initially used `[0]` (single element) as placeholder compile-time args for absent gamma/beta TensorAccessorArgs. The kernel writer's breadcrumb (ts 14:05:22) records: `"upstream_fix": "Fixed TensorAccessorArgs placeholder from [0] to [0,0] for absent gamma/beta"`. TensorAccessorArgs requires exactly 2 compile-time args for interleaved tensors.

**Root Cause**: The builder did not know the exact number of compile-time args that TensorAccessorArgs produces for interleaved tensors. The design document specifies `TensorAccessorArgs | varies` in the CT args table but does not state the exact count.

**Fix for agents**:
- **ttnn-operation-architect**: When specifying TensorAccessorArgs in the CT args table, explicitly state the number of CT args produced (e.g., "TensorAccessorArgs: 2 CT args for interleaved")
- **ttnn-generic-op-builder**: When creating placeholder args for absent TensorAccessorArgs, query the actual size from the API or use a documented constant rather than guessing.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~8m (14:04-14:12) | 0 free, 0 hard | PASS | Clean. Includes all 3 kernel implementations + upstream fix. |
| reduce_mean | ~3m (14:12-14:15) | 0 free, 0 hard | PASS | Clean. Delta: add reduce in compute. |
| subtract_mean | ~3m (14:16-14:18) | 0 free, 0 hard | PASS | Clean. Delta: add sub(COL) in compute. |
| variance_rsqrt | ~3m (14:18-14:22) | 0 free, 0 hard | PASS | Clean. Delta: add 4 phases in compute. |
| affine | ~3m (14:22-14:24) | 0 free, 0 hard | PASS | Clean. Delta: add gamma/beta tilize + mul/add in compute. |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | data_pipeline stage | ttnn-kernel-writer-tdd | 8m | 15% | Initial stage implements all 3 kernels from scratch | 0 | Expected -- first stage is largest scope |
| 2 | Build phase | ttnn-generic-op-builder | 7m | 13% | Program descriptor with 15 CBs, 3 kernels, complex RT args | 0 | High CB count increases boilerplate |
| 3 | Design phase | ttnn-operation-architect | 7m | 13% | 10 compute phases with full CB state tracking | 0 | Thoroughness paid off in zero retries |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| (none) | -- | -- | -- |

No wasted work was identified. All code written was used in the final implementation. The only upstream fix (TensorAccessorArgs placeholder) was a 1-minute correction.

---

## 5. Inter-Agent Communication Issues

### Handoff 1: orchestrator -> ttnn-operation-analyzer (x3)

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation paths + role directives |
| Quality | GOOD |
| Issues | None observed |
| Downstream Impact | None |
| Suggestion | None needed |

### Handoff 2: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| Quality | GOOD |
| Issues | None. Architect breadcrumbs confirm all 3 analyses were consumed. |
| Downstream Impact | None |
| Suggestion | None needed |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 1 |
| Quality | ADEQUATE |
| Issues | (1) Compute CT args listed `num_tile_rows` as CT arg, but builder correctly made it a runtime arg. (2) TensorAccessorArgs placeholder size unspecified. |
| Downstream Impact | Minor: builder had to infer correct approach for num_tile_rows, and guessed wrong placeholder size. |
| Suggestion | Architect should flag which args vary per core. Architect should specify TensorAccessorArgs CT arg counts. |

### Handoff 4: ttnn-operation-architect + ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 + stub kernels + program_descriptor.py + test files |
| Quality | GOOD |
| Issues | (1) Single TensorAccessorArgs placeholder size fix needed. (2) Deliberation text in Stage 2 design. |
| Downstream Impact | Minimal. Kernel writer fixed placeholder in ~1m and ignored deliberation text. |
| Suggestion | See Issue 1 and Issue 3 recommendations. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Specify TensorAccessorArgs CT arg count per memory layout | H | M |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Mark args that vary per core as runtime-only | H | M |
| ttnn-operation-architect | Self-reflection | Remove deliberation text from final design documents | M | L |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Design completeness | CT arg index mismatch between design and implementation | Add validation step comparing design CT arg tables vs builder output | M |
| Logging | All agents produced clean breadcrumbs with timestamps | No changes needed -- current observability is sufficient | -- |
| TDD stages | 5-stage progression was well-calibrated for layer norm | The stage granularity (data_pipeline, reduce_mean, subtract_mean, variance_rsqrt, affine) isolates each algorithmic step effectively | -- |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Zero debugging cycles. All stages passed first attempt. |
| 3 | `.tdd_state.json` coupling fragility | NO | Builder and kernel writer consumed the state file without issues. |
| 4 | No fast path for simple operations | NO | Layer norm is a medium-complexity op; full pipeline was appropriate. |
| 6 | Builder runs on Sonnet while everything else uses Opus | MAYBE | Builder produced correct output with only 1 minor placeholder issue. Hard to say if Opus would have caught the TensorAccessorArgs size. |
| 7 | Discovery phase uses keyword matching | NOT TESTED | Discovery worked correctly for this op. |
| 9 | No validation between architect output and builder output | YES (mild) | The CT arg mismatch (num_tile_rows as CT vs RT arg) went undetected between phases. A cross-check would have flagged it. |
| 11 | No incremental re-run capability | NOT NEEDED | Full pipeline ran to completion without needing re-runs. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect includes deliberation text in final design | Unresolved alternative approaches left in design doc Stage 2 could confuse downstream agents | L |
| TensorAccessorArgs placeholder size not documented | Builder must guess the number of CT args for absent tensors; should be documented in architect's CT arg table | M |

---

## 8. Actionable Recommendations

### Recommendation 1: Architect should separate deliberation from final design

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: Add rule: "In the final op_design.md, each stage's implementation section must contain ONLY the chosen approach. If alternatives were considered, either omit them or move them to a clearly-labeled 'Design Notes' appendix."
- **Expected Benefit**: Downstream agents receive unambiguous implementation guidance
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 2: Architect should validate CT arg core-variance

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: Add validation rule: "For each compile-time arg listed in kernel arg tables, verify it has the same value across all cores. Any arg that differs between core groups (e.g., num_tile_rows in work distribution) MUST be listed as a runtime arg instead."
- **Expected Benefit**: Eliminates the class of bugs where CT args are used for per-core-varying values
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Document TensorAccessorArgs CT arg counts

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: When specifying TensorAccessorArgs in the kernel CT args table, include a note like "TensorAccessorArgs: N CT args for {interleaved|sharded}" based on the tensor's memory layout. This enables the builder to create correct-sized placeholders for absent tensors.
- **Expected Benefit**: Eliminates TensorAccessorArgs placeholder sizing bugs
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Add architect-builder CB cross-validation

- **Type**: new_validation
- **Target**: Orchestrator script (between Phase 2 and Phase 3 completion)
- **Change**: After the builder produces the program descriptor, compare CB allocations (indices, page counts, data formats) against the architect's CB table in op_design.md. Flag any mismatches before Phase 4 begins.
- **Expected Benefit**: Catches CB sizing discrepancies before expensive device tests
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5 | All 3 references (tilize, untilize, batch_norm) were appropriate and useful |
| Analysis quality | 5 | Each analysis was role-scoped and focused. Architect consumed all three. |
| Design completeness | 4 | Excellent CB state tracking and helper specification; minor issues with deliberation text and CT arg classification |
| Build correctness | 4 | Correct infrastructure with only 1 minor TensorAccessorArgs placeholder issue |
| Kernel implementation | 5 | 5/5 stages passed on first attempt. Zero retries. Zero debugging cycles. |
| Inter-agent communication | 4 | Clean handoffs with minor friction at architect-builder boundary (CT arg mismatch, placeholder sizing) |
| Logging/observability | 5 | All agents produced complete breadcrumbs with timestamps. Full timeline reconstruction was straightforward. |
| Helper usage compliance | 5 | 100% compliance. All compute phases use helpers. No missed helpers, no misused helpers. See Section 10. |

### Top 3 Things to Fix

1. **Architect should validate CT arg core-variance** -- prevents the class of bugs where per-core-varying values are incorrectly specified as compile-time args
2. **Document TensorAccessorArgs CT arg counts in design** -- prevents placeholder sizing bugs in the builder
3. **Add architect-builder CB cross-validation** -- catches configuration drift between design and infrastructure (known issue #9)

### What Worked Best

The architect's detailed per-phase CB state tables were the single most impactful element of this pipeline run. By tracing tile lifetimes through all 10 compute phases and explicitly specifying WaitUpfrontNoPop / NoWaitNoPop policies with manual pop annotations, the design eliminated the most common class of kernel bugs (CB synchronization errors). The result: zero debugging cycles across a complex 10-phase compute kernel with 15 circular buffers and 6 different binary operation types.

---

## 10. Helper Usage Audit

### Available Helpers

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `tilize_helpers.hpp` | `tilize<>()` | YES -- input RM sticks must be tilized for compute |
| `untilize_helpers.hpp` | `untilize<>()` | YES -- output must be untilized back to RM sticks |
| `reduce_helpers_compute.hpp` | `reduce<PoolType, ReduceDim, Policy>()` | YES -- row-wise mean and variance reduction |
| `reduce_helpers_dataflow.hpp` | `calculate_and_prepare_reduce_scaler<>()`, `prepare_reduce_scaler<>()` | YES -- reduce scaler (1/W) and epsilon tile generation |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()`, `square<>()` | YES -- sub mean, square, add eps, mul rstd, mul gamma, add beta |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT`, `get_dest_limit()` | YES (implicit) -- used by reduce and binary_op helpers internally |
| `copy_tile_helpers.hpp` | `copy_tiles()` | NO -- not needed; no CB-to-CB tile copies in this op |
| `cb_helpers.hpp` | `get_cb_num_pages()`, `get_full_tile_size()` | NO -- not directly used by kernel code |
| `l1_helpers.hpp` | `zero_faces<>()`, `addr_to_l1_ptr()` | NO -- not needed; zero-padding done manually in reader |
| `common_types.hpp` | `NoOp`, `NoAccumulation` | YES (implicit) -- used as defaults by helper templates |

### Per-Phase Helper Compliance

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| reader | RM stick reads | Raw TensorAccessor | Raw TensorAccessor | Raw Justified | No helper exists for RM stick DRAM reads |
| reader | Reduce scaler gen | `calculate_and_prepare_reduce_scaler` | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, AVG, REDUCE_ROW, 32, Wt*32>()` | Correct | Line 43-48 of reader |
| reader | Epsilon tile gen | `prepare_reduce_scaler` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_float)` | Correct | Line 66 of reader |
| reader | Gamma/beta reads | Raw with zero-padding | Raw TensorAccessor + manual zero-pad loop | Raw Justified | No helper for single-stick read + zero-pad |
| compute | Phase 0: Tilize gamma/beta | `tilize<Wt, c_2, c_30>(1)` | `compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma_tilized>(1)` | Correct | Line 54 of compute |
| compute | Phase 1: Tilize input | `tilize<Wt, c_0, c_1>(1)` | `compute_kernel_lib::tilize<Wt, cb_input_rm, cb_tilized>(1)` | Correct | Line 63 of compute |
| compute | Phase 2: Reduce mean | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(...)` | Correct | Line 67-69 of compute |
| compute | Phase 3: Sub mean | `sub<COL, NoWaitNoPop, WaitAndPopPerTile>` | `compute_kernel_lib::sub<COL, NoWaitNoPop, WaitAndPopPerTile>(...)` | Correct | Line 72-76 of compute |
| compute | Phase 3: Manual pop c_1 | `cb_pop_front(c_1, Wt)` | `cb_pop_front(cb_tilized, Wt)` | Raw Justified | Required by NoWaitNoPop pattern; no helper for manual pop |
| compute | Phase 4: Square | `square<WaitUpfrontNoPop>` | `compute_kernel_lib::square<WaitUpfrontNoPop>(...)` | Correct | Line 83-84 of compute |
| compute | Phase 5: Reduce variance | `reduce<SUM, REDUCE_ROW>` | `compute_kernel_lib::reduce<SUM, REDUCE_ROW>(...)` | Correct | Line 87-88 of compute |
| compute | Phase 6: Add eps + rsqrt | `add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>` with rsqrt post-op | `compute_kernel_lib::add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>(..., [](uint32_t dst_idx){ rsqrt_tile_init(); rsqrt_tile(dst_idx); })` | Correct | Line 91-98 of compute |
| compute | Phase 7: Mul rstd | `mul<COL, NoWaitNoPop, WaitAndPopPerTile>` | `compute_kernel_lib::mul<COL, NoWaitNoPop, WaitAndPopPerTile>(...)` | Correct | Line 101-105 of compute |
| compute | Phase 7: Manual pop c_25 | `cb_pop_front(c_25, Wt)` | `cb_pop_front(cb_centered, Wt)` | Raw Justified | Required by NoWaitNoPop pattern |
| compute | Phase 8: Mul gamma | `mul<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>` (if gamma) | `compute_kernel_lib::mul<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>(...)` | Correct | Line 114-118 of compute |
| compute | Phase 9: Add beta | `add<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>` (if beta) | `compute_kernel_lib::add<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>(...)` | Correct | Line 127-131 of compute |
| compute | Phase 10: Untilize | `untilize<Wt, cb_final, c_16>(1)` | `compute_kernel_lib::untilize<Wt, cb_untilize_src, cb_out_rm>(1)` | Correct | Line 135 of compute |
| writer | RM stick writes | Raw TensorAccessor | Raw TensorAccessor + noc_async_write | Raw Justified | No helper exists for RM stick DRAM writes |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | 18 |
| Phases using helpers correctly | 14 |
| Phases with justified raw code | 4 (RM reads, RM writes, 2x manual pop) |
| Phases with missed helpers | 0 |
| Phases with misused helpers | 0 |
| **Helper compliance rate** | **100%** |

### Redundant CB Operations Around Helpers

No redundant CB operations detected around helper calls. The only `cb_pop_front` calls are the two mandatory manual pops for c_1 (line 79) and c_25 (line 108), which are required because the NoWaitNoPop policy was used on earlier phases that pre-waited these CBs. These are not wrapping helpers -- they are inter-phase transitions explicitly required by the design.

The reader kernel's `cb_reserve_back`/`cb_push_back` calls for gamma/beta (lines 73/97 and 102/117) are not wrapping any helper -- they are raw dataflow operations with no corresponding helper.

### Missed Helper Opportunities

All available helpers were used correctly. No missed opportunities.

One minor observation: the reader's gamma/beta zero-padding loop (lines 89-95) manually writes zeros using `reinterpret_cast<volatile uint32_t*>`. The `l1_helpers.hpp` header provides `zero_faces<>()` which uses NoC reads from hardware zeros, which is faster for large regions. However, `zero_faces` operates on face-sized chunks (16x16), not stick-sized chunks, so it is not directly applicable here. The manual approach is correct for this use case.
