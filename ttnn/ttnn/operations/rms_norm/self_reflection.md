# Self-Reflection: rms_norm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `rms_norm` |
| Operation Path | `ttnn/ttnn/operations/rms_norm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd (x3 sessions) |
| Total Git Commits | 10 (on this branch for this run) |
| Total Pipeline Duration | ~87 minutes (08:58 to 10:25) |
| Overall Result | SUCCESS -- all 4 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m | COMPLETE | Selected 3 references: tilize (input_stage), untilize (output_stage), batch_norm (compute_core). Correct choices for a normalization op with RM/TILE support. |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~12m | COMPLETE | 3 parallel analyzers produced tilize_analysis.md (20KB), untilize_analysis.md (23KB), batch_norm_analysis.md (33KB). Thorough DeepWiki research on TensorAccessor, reduce APIs, and SFPU ops. |
| 2: Design | ttnn-operation-architect | ~10m | COMPLETE | Produced op_design.md (352 lines) covering two-pass data flow, 12 CBs, 8 compute phases, 4 TDD stages. Validated helper library applicability. |
| 3: Build | ttnn-generic-op-builder | ~8m | COMPLETE | Created 11 files (3 kernel stubs, entry point, program descriptor, 5 tests, init). Hit 1 compile error (kernel_lib includes not on JIT path), fixed in 1 retry. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd (x3 sessions) | ~48m | COMPLETE | All 4 stages passed. 3 sessions due to context window limits. 2 hangs encountered and resolved. Multiple CB sizing fixes applied upstream. |
| 5: Report | orchestrator | ~3m | COMPLETE | Generated REPORT.md summarizing full pipeline run. |
| **Total** | | **~87m** | | From earliest analyzer start (08:58) to report commit (10:25). |

### Agent Duration Breakdown

Duration calculation method: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps where available, supplemented by git commit timestamps.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 08:58:58 | 09:10:12 | ~11m | 0 | ~11m active (reading, DeepWiki queries, writing analysis) |
| ttnn-operation-analyzer (untilize) | 08:58:48 | 09:10:12 | ~11m | 0 | ~11m active (parallel with tilize/batch_norm analyzers) |
| ttnn-operation-analyzer (batch_norm) | 08:58:57 | 09:06:59 | ~8m | 0 | ~8m active |
| ttnn-operation-architect | 09:10:00 | 09:20:21 | ~10m | 0 | ~10m active (read analyses, validated helpers, wrote design) |
| ttnn-generic-op-builder | 09:22:41 | 09:30:18 | ~8m | 1 (compile error) | ~6m active, ~2m fixing kernel_lib include issue |
| ttnn-kernel-writer-tdd (session 1) | 09:33:57 | ~09:49 | ~15m | 2 (compile, hang) | ~8m coding, ~4m test runs, ~3m debugging hang |
| ttnn-kernel-writer-tdd (session 2) | 09:50:24 | ~09:58 | ~8m | 1 (hang on square_reduce) | ~4m coding, ~4m debugging cb_x_sq sizing hang |
| ttnn-kernel-writer-tdd (session 3) | 09:59:09 | 10:22:06 | ~23m | 2 (compile, hang) | ~10m coding, ~6m test runs, ~7m debugging gamma hang + cb_norm sizing |

### Duration Visualization

```
Phase 0  |#|                                                  (~1m)
Phase 1  |############|                                        (~12m) 3 analyzers in parallel
Phase 2              |##########|                              (~10m)
Phase 3                        |########|                      (~8m)
Phase 4                                 |################################| (~48m) 3 sessions, 4 stages
Phase 5                                                                  |###| (~3m)
         0    5    10   15   20   25   30   35   40   45   50   55   60 min

Longest phase: Phase 4 (~48m) -- kernel implementation with 3 context-window-exhausting sessions
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~13m | 15% | 3 parallel analyzers |
| Design (Phase 2) | ~10m | 11% | Single architect agent |
| Build (Phase 3) | ~8m | 9% | 1 compile-error retry |
| Kernel implementation (Phase 4) | ~48m | 55% | 3 sessions, 4 TDD stages |
|   Productive coding | ~22m | 25% | Writing kernel code that passed |
|   Test execution | ~14m | 16% | Running tests (~3m per test run x ~5 runs) |
|   Debugging/retries | ~12m | 14% | Hypothesis-fix-retest cycles for hangs |
| Reporting (Phase 5) | ~3m | 3% | |
| Inter-session gaps | ~6m | 7% | Time between agent sessions (context switch overhead) |
| **Total** | **~87m** | **100%** | |

---

## 2. What Went Well

### 1. Exceptional Helper Library Compliance

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd) and Phase 2 (ttnn-operation-architect)
**Evidence**: Every single compute phase uses a helper function. The final compute kernel (143 lines) contains zero raw `tile_regs_acquire`/`tile_regs_commit`/`cb_reserve_back`/`cb_push_back` calls. 8 compute phases all use helpers: `tilize<>()`, `square<>()`, `reduce<>()`, `add<>()`, `mul<>()`, `untilize<>()`. The only manual CB operations are 2 `cb_pop_front` calls for tiles held by `WaitUpfrontNoPop` policies, which are correct and required.
**Why it worked**: The architect explicitly mapped every compute phase to a helper function in op_design.md Part 2 with exact template parameters and policies. The kernel writer followed these mappings faithfully.

### 2. Zero Numerical Debugging

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: All 4 TDD stages passed without any numerical mismatch issues. The `.tdd_state.json` shows 0 "attempts" (hard failures) across all stages. The breadcrumbs contain zero `hypothesis` entries related to numerical accuracy. Every failure was either a compilation error (2 occurrences) or a hang (3 occurrences) -- both structural issues, not mathematical correctness issues.
**Why it worked**: The TDD stages were well-designed with incremental complexity (identity -> square_reduce -> normalize -> gamma). Each stage isolated a specific mathematical transformation, making it easy to verify correctness in isolation. The tolerances (rtol=0.05, atol=0.2 for later stages) were appropriate for bfloat16 precision.

### 3. Zero Device Hangs Requiring External Recovery

**Phase/Agent**: All of Phase 4
**Evidence**: While 3 hangs occurred (data_pipeline multi-batch, square_reduce multi-tile, gamma_scale multi-tile), all were detected by the test timeout mechanism and identified by the kernel writer within 1-2 hypotheses. None required manual device reset or external intervention. The root causes were all CB sizing issues (structural, not data-dependent).
**Why it worked**: The test parametrization over multiple shapes (including multi-batch and multi-tile shapes) was effective at catching CB sizing bugs that only manifest when Wt > 2 or num_rows > 1.

### 4. Effective Two-Pass Data Flow Design

**Phase/Agent**: Phase 2 (ttnn-operation-architect)
**Evidence**: The two-pass reader design (pass 1: read for square+reduce, pass 2: re-read for normalization) worked correctly on first implementation for all shapes. The `start_stick_id`/`row_start_tile` tracking pattern was clean and bug-free. No issues with data re-reading across the entire TDD progression.
**Why it worked**: The architect clearly documented the two-pass strategy with explicit CB flow diagrams. The reader kernel implementation directly followed this design.

### 5. Clean TDD Stage Progression

**Phase/Agent**: Phase 2 (architect) + Phase 4 (kernel writer)
**Evidence**: The 4-stage TDD plan (data_pipeline -> square_reduce -> rms_normalize -> gamma_scale) provided clean incremental build-up. Each stage added exactly one functional block. Stage 2 required a temporary output shape change that was cleanly reverted in stage 3. The stage isolation meant that bugs in later stages never required reworking earlier code.

---

## 3. Issues Found

### Issue 1: Architect Specified 2 Pages for CBs That Need Wt Pages

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- square_reduce, gamma_scale |
| Agent | ttnn-operation-architect (root cause), ttnn-kernel-writer-tdd (discovered and fixed) |
| Retries Consumed | 2 hangs (1 in square_reduce for cb_x_sq, 1 in gamma_scale for cb_norm) + 1 upstream fix for cb_gamma |
| Time Cost | ~8 minutes total debugging across the two hang incidents |

**Problem**: The architect's op_design.md specified 2 pages for cb_x_sq (c_2), cb_gamma (c_4), and cb_norm (c_26). All three needed Wt pages because they serve as intermediate buffers between two sequential compute operations on the same RISC-V core. When both producer and consumer run on the same thread (compute), the producer must complete all Wt tiles before the consumer starts. With only 2 pages and Wt > 2, `cb_reserve_back` deadlocks because the CB is full and no consumer thread is running to drain it.

Specifically:
- cb_x_sq: `square` pushes Wt tiles, then `reduce` consumes them (breadcrumb H1 at 09:55:28)
- cb_gamma: `tilize` pushes Wt gamma tiles, then `mul<ROW>` consumes them (breadcrumb upstream_fix at 10:12:47)
- cb_norm: `mul<COL>` pushes Wt tiles, then `mul<ROW>` consumes them (breadcrumb at 10:20:25)

The issue only manifested on multi-tile shapes (Wt >= 4) -- single-tile shapes (Wt=1) passed fine.

**Root Cause**: The architect designed CB page counts based on "streaming 2-page double-buffer" reasoning, which is correct when producer and consumer run on different threads (e.g., reader -> compute, or compute -> writer). But for intra-compute CB transfers where producer and consumer are sequential operations on the same RISC core, the full Wt pages must be available. The design document's CB table (Part 1) lists cb_x_sq, cb_gamma, and cb_norm as "2 pages" with "per-tile streaming" lifetime, but the compute kernel pseudocode (Part 2) shows sequential phase execution on the same core.

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "For any CB that is both produced and consumed by sequential phases within the same compute kernel (no concurrent consumer), the page count MUST be >= the total tiles produced in a single batch (typically Wt for row-oriented operations). The 2-page streaming pattern only works when producer and consumer run on different RISC cores (reader/writer vs compute) or on concurrent unpack/pack threads." This rule should be part of the CB sizing checklist.

### Issue 2: Gamma TensorAccessorArgs Placeholder Size Mismatch

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- gamma_scale |
| Agent | ttnn-generic-op-builder (root cause), ttnn-kernel-writer-tdd (discovered and fixed) |
| Retries Consumed | 0 hard retries (fixed proactively before test run), but added debug time |
| Time Cost | ~2 minutes |

**Problem**: The builder generated a single `[0]` as the placeholder for absent gamma TensorAccessorArgs in `rms_norm_program_descriptor.py`. However, interleaved TensorAccessorArgs uses 2 compile-time args (args_config + aligned_page_size). The kernel unconditionally declares `gamma_ta_args = TensorAccessorArgs<offset>()` which reads 2 CT args from that offset. With only 1 placeholder, the second read is out of bounds.

The kernel writer identified this from breadcrumb at 10:10:41 and changed `[0]` to `[0, 0]`.

**Root Cause**: The builder template does not know the exact CT arg count for TensorAccessorArgs for different memory layouts. It used a single 0 as a generic placeholder. The comment added by the kernel writer explains: "TensorAccessorArgs for interleaved tensors uses 2 CT args."

**Fix for agents**:
- **ttnn-generic-op-builder**: When generating placeholder CT args for conditional TensorAccessor parameters (like optional gamma), always use `[0, 0]` (2 zeros) as the default, matching interleaved layout which requires 2 CT args. Document this in the builder's stub generation logic with a comment explaining why.

### Issue 3: Context Window Exhaustion Forced 3 Kernel Writer Sessions

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- all stages |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 (not a test failure, but pipeline efficiency loss) |
| Time Cost | ~6 minutes in inter-session gaps + context re-parsing overhead |

**Problem**: The TDD kernel writer required 3 separate agent sessions to complete 4 TDD stages:
- Session 1 (09:33:57 - ~09:49): Completed data_pipeline, started square_reduce but exhausted context
- Session 2 (09:50:24 - ~09:58): Completed square_reduce, started rms_normalize but exhausted context
- Session 3 (09:59:09 - 10:22:06): Completed rms_normalize and gamma_scale

Each new session had to re-read op_design.md, re-parse the current state of all 3 kernel files, and understand what was already done. This adds ~2-3 minutes of overhead per session.

**Root Cause**: Large analysis documents (76KB combined) are passed as context to the kernel writer. The writer also generates verbose breadcrumbs. For a complex 8-phase compute kernel with 2-pass reader, the total context (design doc + kernel code + test output + breadcrumbs) exceeds the context window after ~1.5 stages.

**Fix for agents**:
- **Pipeline orchestrator**: Consider summarizing analysis documents before passing to the kernel writer. The writer needs the CB table, kernel arg table, and compute phase pseudocode from op_design.md Part 2 -- not the full analysis documents (tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md). Stripping these from the kernel writer's context could save ~60KB.

### Issue 4: RM Tilize/Untilize Interleaving Not Specified in Design

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- data_pipeline |
| Agent | ttnn-operation-architect (omission), ttnn-kernel-writer-tdd (discovered and fixed) |
| Retries Consumed | 1 hang on multi-batch shape (4,2,64,64) |
| Time Cost | ~4 minutes |

**Problem**: The design document shows Phase 1 (tilize) and Phase 8 (untilize) as separate sequential operations. For the data_pipeline stage, the writer initially implemented them as bulk operations: tilize ALL rows, then untilize ALL rows. This deadlocked because cb_out holds only Wt pages -- after the first row, tilize blocks on cb_reserve_back since cb_out is full and untilize hasn't started draining it.

The fix was to interleave tilize+untilize per tile-row in a loop (breadcrumb H2 at 09:42:23).

**Root Cause**: The design document's Phase 1 and Phase 8 descriptions do not mention that they must be interleaved per tile-row for the identity/data_pipeline stage. The full compute pipeline (stages 2-4) naturally interleaves because intermediate phases (square, reduce, normalize) consume cb_in before the next tilize, but the identity passthrough has no intermediate consumer.

**Fix for agents**:
- **ttnn-operation-architect**: In the Stage 1 (data_pipeline) design section, explicitly note: "For RM identity passthrough, tilize and untilize MUST be interleaved per tile-row in a loop, not called as bulk operations, because cb_out capacity is Wt pages." This is a data_pipeline-specific concern that doesn't apply to later stages.

### Issue 5: Builder Include Paths Incompatible with JIT Compilation

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 -- Build |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 1 free retry (compile error, quick fix) |
| Time Cost | ~2 minutes |

**Problem**: The builder initially used `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` include paths in stub kernels, which are not on the JIT kernel compile include path. The fix was to use `api/tensor/tensor_accessor.h` instead. (Builder breadcrumb H1 at 09:28:23.)

**Root Cause**: The builder's include mapping table did not match the JIT compile environment. The builder execution log (Section 7) explicitly recommends fixing this in instructions.

**Fix for agents**:
- **ttnn-generic-op-builder**: Update the include mapping table in the builder's prompt to use only `api/` prefixed includes that are known to work on the JIT compile path. This was already recommended by the builder itself.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~10m | 1 free (compile), 1 hard (hang) | PASS | RM tilize/untilize interleaving hang on multi-batch |
| square_reduce | ~13m (across 2 sessions) | 0 free, 1 hard (hang) | PASS | cb_x_sq sizing hang on multi-tile, plus context switch between sessions |
| rms_normalize | ~9m | 1 free (compile -- rsqrt include) | PASS | Clean once include was fixed |
| gamma_scale | ~13m | 0 free, 1 hard (hang) | PASS | Gamma stick replication + cb_norm sizing hang |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | CB sizing hangs | kernel-writer | ~12m | 14% | 3 separate hangs from CB page counts of 2 where Wt was needed | 3 | Architect specified 2-page streaming for intra-compute CBs |
| 2 | Context window overhead | kernel-writer | ~6m | 7% | 3 sessions required, each re-reading design and kernel state | N/A | Large analysis documents in context |
| 3 | Test execution | kernel-writer | ~14m | 16% | ~5 test runs at ~3m each (8 test cases per run) | N/A | Inherent cost of TDD on hardware |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| kernel-writer (session 1) | Started implementing square_reduce stage | Session hit context limit before completing; session 2 had to re-implement | Better context management / analysis summarization |
| kernel-writer (all sessions) | Re-read op_design.md and all kernel files at start of each session | Repeated context parsing (~2m per session) | Pass a condensed "current state summary" to continuation sessions |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md (76KB total) |
| Quality | GOOD |
| Issues | None significant. Analyses were comprehensive and well-structured. |
| Downstream Impact | Architect used all three analyses effectively, mapping tilize/untilize helpers to input/output stages and batch_norm patterns to compute. |
| Suggestion | Consider length limits -- 76KB of analysis is substantial for downstream context windows. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md (352 lines) |
| Quality | GOOD |
| Issues | Minor: CB page counts for cb_x_sq, cb_gamma, cb_norm were 2 instead of Wt. This didn't affect the builder (which just allocates the specified sizes) but propagated to Phase 4. |
| Downstream Impact | Builder faithfully implemented the specified CB sizes. The kernel writer had to fix them during TDD. |
| Suggestion | The architect should validate CB sizing against helper requirements -- helpers like `tilize<>()` push full blocks of Wt tiles, and sequential same-RISC phases need Wt-page intermediates. |

### Handoff 3: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | 11 files (stubs, program descriptor, tests) + op_design.md (passed through) |
| Quality | ADEQUATE |
| Issues | (1) Gamma TA placeholder was `[0]` instead of `[0, 0]`. (2) kernel_lib includes not on JIT path (fixed by builder before handoff, but handoff notes warned about it). (3) Handoff notes were excellent -- clearly documented cb_out pages for RM vs TILE, conditional CBs, and scaler format. |
| Downstream Impact | Gamma TA placeholder caused 1 upstream fix by kernel writer. The well-documented handoff notes helped the writer understand conditional CB allocation. |
| Suggestion | Builder should use `[0, 0]` for interleaved TA placeholders. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | ttnn-generic-op-builder (self) | Update include mapping table to use `api/` prefixed paths instead of `ttnn/cpp/ttnn/kernel_lib/` paths for JIT compilation | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-kernel-writer-tdd (via upstream fixes) | Add validation rule for intra-compute CB sizing: sequential phases on same RISC need >= Wt pages, not 2 | HIGH | HIGH |
| ttnn-generic-op-builder | ttnn-kernel-writer-tdd (via upstream fix) | Use `[0, 0]` not `[0]` for interleaved TensorAccessorArgs placeholder | HIGH | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD / context | Kernel writer consumed 3 sessions for 4 stages | Summarize analysis docs before passing to kernel writer; strip raw analyses from context | HIGH |
| Design / CB validation | 3 CBs had wrong page counts causing hangs | Add automated CB sizing validation: any CB that is both source and sink of sequential compute phases must have >= Wt pages | HIGH |
| Build / templates | Gamma TA placeholder was wrong size | Document correct placeholder sizes per memory layout in builder templates | MEDIUM |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Zero numerical debugging needed in this run. All issues were structural (hangs, compile errors). |
| 2 | Long leash (planner/designer gap) | NO (DONE) | Merged architect worked well. Single design doc was clear and complete. |
| 3 | `.tdd_state.json` coupling fragility | NO | State file worked correctly throughout. No schema issues. |
| 4 | No fast path for simple operations | NO | RMS norm is medium complexity; full pipeline was appropriate. |
| 6 | Builder model choice (Sonnet vs Opus) | MAYBE | Builder hit the kernel_lib include issue. A stronger model might have caught this from reference code, but it was a quick fix. |
| 7 | Discovery keyword matching | NO | Discovery found correct references (tilize, untilize, batch_norm). |
| 9 | No architect/builder cross-validation | YES | CB sizing errors in architect's design propagated through builder to kernel writer. If a static cross-check existed, it could have caught the 2-page vs Wt-page mismatch for intra-compute CBs. |
| 11 | No incremental re-run capability | NO | Pipeline completed successfully without needing re-runs. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect CB sizing rule for same-RISC sequential phases | Architect uses 2-page "streaming" pattern for all intermediate CBs, but intra-compute sequential phases need Wt pages because producer must finish before consumer starts on the same thread. Caused 3 hangs in this run. | HIGH |
| Context window management for complex ops | 8-phase compute kernel with 76KB analysis context caused 3 context exhaustion events. Need analysis summarization before kernel writer phase. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Add Intra-Compute CB Sizing Validation to Architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add the following rule to the CB sizing section: "For any CB that serves as an intermediate between two sequential compute phases on the same RISC core (where the producer phase must complete all tiles before the consumer phase starts), allocate >= Wt pages (or the total tiles produced per batch unit). The 2-page double-buffer pattern is ONLY valid when producer and consumer run on different threads (reader->compute, compute->writer) or on concurrent unpack/pack pipelines."
- **Expected Benefit**: Eliminates the most common hang pattern observed (3 instances in this run, ~8 minutes wasted)
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Summarize Analysis Documents Before Kernel Writer Phase

- **Type**: pipeline_change
- **Target**: Pipeline orchestrator (Phase 4 context preparation)
- **Change**: Before launching the kernel writer, extract a condensed summary from each analysis document (CB tables, key patterns, include paths) and pass only the summary + op_design.md Part 2. Drop the full analysis documents (~76KB) from the kernel writer's context.
- **Expected Benefit**: Reduces context consumption by ~60KB, potentially allowing all 4 TDD stages in 2 sessions instead of 3
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 3: Fix Builder TensorAccessorArgs Placeholder

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent prompt
- **Change**: In the stub generation logic for conditional TensorAccessor parameters, change the placeholder from `[0]` to `[0, 0]` for interleaved memory layout. Add a comment: "Interleaved TensorAccessorArgs uses 2 CT args (args_config, aligned_page_size). Placeholder must match."
- **Expected Benefit**: Prevents 1 upstream fix per operation that has optional tensor parameters
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Add CB Sizing Cross-Validation Between Architect and Builder

- **Type**: new_validation
- **Target**: Pipeline orchestrator (between Phase 2 and Phase 3, or within Phase 3)
- **Change**: After the builder creates the program descriptor, validate that CB page counts match the architect's design. For CBs that serve intra-compute phases, additionally check that page count >= Wt. This could be a simple Python script that parses op_design.md's CB table and compares against the program descriptor's CB allocation.
- **Expected Benefit**: Catches CB sizing mismatches before any kernel code runs, saving entire TDD debug cycles
- **Priority**: MEDIUM
- **Effort**: MEDIUM

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5 | Correct references selected: tilize (input), untilize (output), batch_norm (compute). All were relevant and useful. |
| Analysis quality | 4 | Thorough analyses (76KB total) with DeepWiki research. Slightly too verbose for downstream context windows. |
| Design completeness | 4 | Comprehensive 352-line design with helper mappings and CB flow. Deducted 1 point for 3 CB sizing errors. |
| Build correctness | 4 | All infrastructure correct except gamma TA placeholder (`[0]` vs `[0, 0]`) and JIT include paths. Both fixed quickly. |
| Kernel implementation | 5 | All 4 stages passed with zero numerical issues. Helper compliance was perfect. Hangs were from upstream CB sizing, not kernel logic. |
| Inter-agent communication | 4 | Handoff notes were excellent. CB sizing errors propagated from architect through builder to kernel writer, but this is an architect issue, not a communication issue. |
| Logging/observability | 4 | Breadcrumbs were detailed with timestamps, hypotheses, and fix tracking. Execution log from builder was thorough. Missing: kernel writer execution logs (only breadcrumbs available). |
| Helper usage compliance | 5 | 100% helper compliance. Every compute phase uses the appropriate helper. Zero raw LLK calls. |

### Top 3 Things to Fix

1. **Architect CB sizing rule for intra-compute sequential phases**: This caused 3 hangs and ~8 minutes of debugging. A simple rule addition to the architect's prompt would eliminate this entirely.
2. **Context window management for kernel writer**: 3 sessions for 4 stages is expensive. Summarizing analysis documents could save ~6 minutes and one full session.
3. **Builder TensorAccessorArgs placeholder size**: Use `[0, 0]` for interleaved layout placeholders to prevent downstream fixes.

### What Worked Best

The helper library compliance was outstanding. The architect mapped every compute phase to a specific helper with exact template parameters and policies (tilize, square, reduce, add with rsqrt post-op, mul with COL/ROW broadcast). The kernel writer followed these mappings exactly, producing a clean 143-line compute kernel with zero raw LLK calls. This is the gold standard for how the pipeline should work: design specifies helpers, implementation uses them faithfully, and the resulting kernel is clean, correct, and maintainable. The zero numerical debugging is a direct consequence of this helper compliance -- the helpers abstract away tile register management, CB synchronization, and broadcast semantics that are the primary sources of numerical bugs.

---

## 10. Helper Usage Audit

### Available Helpers

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `tilize_helpers.hpp` | `tilize<block_width, in_cb, out_cb>(num_blocks)` | YES -- RM input requires tilize, gamma is always RM |
| `untilize_helpers.hpp` | `untilize<block_width, in_cb, out_cb>(num_blocks)` | YES -- RM output requires untilize |
| `reduce_helpers_compute.hpp` | `reduce<PoolType, ReduceDim, Policy>(in_cb, scaler_cb, out_cb, shape)` | YES -- row reduction for mean(x^2) |
| `reduce_helpers_dataflow.hpp` | `prepare_reduce_scaler<cb_id>(scaler_f)` | YES -- scaler and epsilon tile preparation |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()`, `square<>()` | YES -- square, add+rsqrt, normalize multiply, gamma multiply |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT`, `get_dest_limit()` | YES -- used internally by all helpers above |
| `copy_tile_helpers.hpp` | `copy_tiles(in_cb, out_cb, N)` | NO -- not needed in final design (TILE identity was replaced by full compute pipeline) |
| `cb_helpers.hpp` | `get_cb_num_pages()`, `get_full_tile_size()` | NO -- not directly used, but available if needed |
| `l1_helpers.hpp` | L1 memory utilities | NO -- not relevant to this op |

### Per-Phase Helper Compliance

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| compute | Phase 1: Tilize (RM) | `tilize<Wt, cb_in_rm, cb_in>()` | `compute_kernel_lib::tilize<Wt, cb_in_rm, cb_in, InitAndUninit, WaitBlock>(1)` | Correct | Line 55-60 of rms_norm_compute.cpp |
| compute | Phase 2: Square | `square<WaitAndPopPerTile>()` | `compute_kernel_lib::square<WaitAndPopPerTile>(cb_in, cb_x_sq, BlockShape::of(1, Wt))` | Correct | Line 64-65 |
| compute | Phase 3: Reduce Row | `reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>()` | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>(cb_x_sq, cb_scaler, cb_reduce_out, ReduceInputBlockShape::row(Wt, 1))` | Correct | Lines 69-71 |
| compute | Phase 4: Add + Rsqrt | `add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>()` with rsqrt post-op | `compute_kernel_lib::add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>(cb_reduce_out, cb_eps, cb_rms_inv, single(), rsqrt_lambda)` | Correct | Lines 76-87 |
| compute | Phase 5: Re-tilize (RM) | Same as Phase 1 | Same helper call | Correct | Lines 90-97 |
| compute | Phase 6: Normalize | `mul<COL, WaitAndPopPerTile, WaitUpfrontNoPop>()` | `compute_kernel_lib::mul<COL, WaitAndPopPerTile, WaitUpfrontNoPop>(cb_in, cb_rms_inv, cb_norm_or_out, BlockShape::of(1, Wt))` | Correct | Lines 103-107 |
| compute | Phase 7a: Gamma tilize | `tilize<Wt, cb_gamma_rm, cb_gamma>()` | `compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma, InitAndUninit, WaitBlock>(1)` | Correct | Lines 115-120 |
| compute | Phase 7b: Gamma mul | `mul<ROW, WaitAndPopPerTile, WaitAndPopPerTile>()` | `compute_kernel_lib::mul<ROW, WaitAndPopPerTile, WaitAndPopPerTile>(cb_norm, cb_gamma, cb_out, BlockShape::of(1, Wt))` | Correct | Lines 123-127 |
| compute | Phase 8: Untilize (RM) | `untilize<Wt, cb_out, cb_out_rm>()` | `compute_kernel_lib::untilize<Wt, cb_out, cb_out_rm, InitAndUninit, WaitBlock>(1)` | Correct | Lines 132-137 |
| reader | Scaler prep | `prepare_reduce_scaler<cb_scaler>()` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_val)` | Correct | Line 54 |
| reader | Eps prep | `prepare_reduce_scaler<cb_eps>()` | `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_conv.f)` | Correct | Line 65, design deviation: used prepare_reduce_scaler instead of custom broadcast fill, functionally equivalent for SCALAR broadcast |
| reader | Input read | Raw TensorAccessor + NoC reads | Raw code | Raw Justified | No helper exists for DRAM read via TensorAccessor |
| reader | Gamma read | Raw TensorAccessor + NoC reads | Raw code | Raw Justified | No helper exists for stick replication pattern |
| writer | Output write | Raw TensorAccessor + NoC writes | Raw code | Raw Justified | No helper exists for DRAM write via TensorAccessor |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | 14 (9 compute + 2 reader prep + 2 reader I/O + 1 writer I/O) |
| Phases using helpers correctly | 11 |
| Phases with justified raw code | 3 (reader input I/O, reader gamma I/O, writer output I/O) |
| Phases with missed helpers | 0 |
| Phases with misused helpers | 0 |
| **Helper compliance rate** | **100%** (11/11 helper-eligible phases use helpers; 3/3 raw phases have no available helper) |

### Redundant CB Operations Around Helpers

The compute kernel contains exactly two manual `cb_pop_front` calls:
- Line 110: `cb_pop_front(cb_rms_inv, 1)` -- required to release the tile held by `WaitUpfrontNoPop` policy on the B input of `mul<COL>`. This is NOT redundant; the helper deliberately leaves the tile in the CB for reuse across all Wt iterations, and the caller must pop it after the loop.
- Line 142: `cb_pop_front(cb_eps, 1)` -- required to release the persistent epsilon tile held by `WaitUpfrontNoPop` across all tile-rows. This is NOT redundant; it's the documented cleanup pattern for persistent CBs.

No redundant CB operations detected around helper calls.

### Missed Helper Opportunities

All available helpers were used correctly. No missed opportunities.

The reader's epsilon fill used `prepare_reduce_scaler` (a dataflow helper for filling scaler tiles) instead of a custom broadcast fill implementation. While the design originally considered a different approach, `prepare_reduce_scaler` fills row 0 of each tile face, which is exactly what SCALAR broadcast reads. This is a valid and arguably better approach since it reuses an existing helper rather than writing custom code.
