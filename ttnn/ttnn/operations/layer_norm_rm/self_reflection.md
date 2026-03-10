# Self-Reflection: layer_norm_rm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 12 (this run: `85e7b63`..`ded1a40`) |
| Total Pipeline Duration | ~46 minutes (08:56 - 09:42 UTC, Mar 10 2026) |
| Overall Result | SUCCESS |

**Context**: This was the 6th+ iteration of the `layer_norm_rm` pipeline across multiple days (Feb 23 - Mar 10). Previous runs on Feb 23-25, Mar 3, Mar 5, Mar 6, and Mar 9 show progressive refinement of both the pipeline tooling and the operation itself. This analysis focuses on the **March 10 run** (the latest fully automated run on the current branch), but references earlier runs for context.

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m | PASS | Selected tilize, untilize, batch_norm as references. Replaced softmax (used in earlier runs) with batch_norm for better gamma/beta pattern match. |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~6m (08:56-09:02) | PASS | 3 parallel analyses: tilize (20KB), untilize (22KB), batch_norm (33KB). Total 75KB of analysis output. |
| 2: Design | ttnn-operation-architect | ~7m (09:03-09:10) | PASS | Produced 412-line op_design.md with 13 CBs, 8 compute phases, 4 TDD stages. Clean execution with no revisions needed. |
| 3: Build | ttnn-generic-op-builder | ~13m (09:11-09:24) | PASS | 840 lines of scaffolding across 11 files. Ran 8/8 integration tests to verify import and basic structure. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~14m (09:25-09:39) | PASS | All 4 stages passed first try, 0 hard retries. 411 total kernel lines. |
| 5: Report | orchestrator | ~3m (09:39-09:42) | PASS | Generated REPORT.md |

### Agent Duration Breakdown

Timing source: breadcrumb `"event":"start"` and `"event":"complete"` timestamps (ISO 8601). Where `complete` was missing, used last breadcrumb timestamp.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| analyzer (tilize) | 08:56:31 | 09:00:35 | 4m 4s | 0 | ~4m active (file reads + analysis writing) |
| analyzer (untilize) | 08:56:38 | 09:01:00 | 4m 22s | 0 | ~4m active |
| analyzer (batch_norm) | 08:56:46 | 09:02:12 | 5m 26s | 0 | ~5m active (largest analysis) |
| architect | 09:03:21 | 09:09:48 | 6m 27s | 0 | ~6m active, 0m debugging |
| builder | 09:11:17 | 09:23:28 | 12m 11s | 0 | ~12m active (file generation + test run) |
| kernel-writer-tdd | 09:25:00 | 09:38:33 | 13m 33s | 0 | ~13m active, 0m debugging |

**Duration calculation method**: Breadcrumb start/complete events for all agents. All agents had both events available.

### Duration Visualization

```
Phase 0  |#|                                              (~1m)
Phase 1  |#####|                                          (~6m) 3 analyzers in parallel
Phase 2       |######|                                    (~7m)
Phase 3             |############|                        (~13m)
Phase 4                          |#############|          (~14m)
Phase 5                                        |###|      (~3m)
         0    5    10   15   20   25   30   35   40   45 min

Longest phase: Phase 4 (~14m) -- kernel implementation across 4 TDD stages
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~7m | 15% | 3 analyzers in parallel |
| Design (Phase 2) | ~7m | 15% | Single architect, clean pass |
| Build (Phase 3) | ~13m | 28% | 11 files created, integration tests run |
| Kernel implementation (Phase 4) | ~14m | 30% | 4 TDD stages, all first-try |
| -- Productive coding | ~14m | 30% | All time was productive (zero debugging) |
| -- Debugging/retries | 0m | 0% | No debugging cycles at all |
| Reporting (Phase 5) | ~3m | 7% | REPORT.md generation |
| Inter-phase gaps | ~2m | 5% | Agent spawn/handoff overhead |
| **Total** | **~46m** | **100%** | |

---

## 2. What Went Well

### 1. Zero Hard Retries Across All 4 TDD Stages

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `"attempts": 0, "free_retries": 0, "failure_history": []` for all 4 stages. Breadcrumbs confirm linear progression: `data_pipeline` at 09:28, `reduce_mean` at 09:30, `variance_normalize` at 09:33, `affine_transform` at 09:38. No `hypothesis` or `deviation` events in any breadcrumb file.
**Why it worked**: Three factors converged:
1. The architect's design document was detailed enough that the kernel writer could implement directly from it (8 compute phases with exact helper signatures, CB lifecycle tables, explicit push/pop balance).
2. The helper library (`tilize_helpers.hpp`, `reduce_helpers_compute.hpp`, `binary_op_helpers.hpp`, `untilize_helpers.hpp`) abstracted away low-level tile register and DEST management.
3. The `cb_in_rm page_size = tile_size` requirement (a recurring historical bug) was explicitly called out in both the design doc and MEMORY.md, preventing the most common first-stage failure.

### 2. Reference Operation Selection Improvement

**Phase/Agent**: Phase 0, orchestrator
**Evidence**: Previous runs (Mar 3, Mar 5, Mar 6) used `softmax` as the compute_core reference. The Mar 10 run switched to `batch_norm`, which provides a much closer pattern match: gamma/beta affine handling, reduce-based normalization, scalar CB setup for epsilon. The `phase0_discovery.md` explicitly documents this choice.
**Why it worked**: batch_norm shares the same gamma/beta broadcast pattern (per-feature scale and shift), the same reduce-for-statistics flow, and the same epsilon scalar handling. The architect's breadcrumbs show it extracted `"FPU binary ops, binary_dest_reuse for sub+mul fusion, epsilon scalar fill, conditional affine transform"` from the batch_norm analysis -- all directly applicable.

### 3. CB Layout Correct on First Pass

**Phase/Agent**: Phase 2, ttnn-operation-architect
**Evidence**: The architect's execution log states `"No architecture revisions needed"` and `"Pass 1 CB layout was compatible with all helpers"`. The final kernel code uses exactly the 13 CBs designed (IDs 0-9, 16, 17, 24). No CB was added, removed, or resized during Phase 4 (aside from the `cb_var_input` sizing expansion from 1 to Wt tiles, which was a program descriptor bug, not a design bug).
**Why it worked**: The architect validated CB requirements against helper signatures in Pass 2 before writing the design document. This two-pass approach (architecture first, then helper validation) catches incompatibilities before they reach the builder.

### 4. Clean Inter-Phase Handoffs

**Phase/Agent**: All phases
**Evidence**: No `upstream_feedback` breadcrumb events in any agent's log. The architect's execution log Section 3 (Recovery Summary) states `"No errors occurred"` and Section 7 (Instruction Improvement Recommendations) states `"None"`. The builder's breadcrumbs show `"8/8 integration tests pass"` on first run.
**Why it worked**: The hybrid mode with 3 role-specific analyses (input_stage, compute_core, output_stage) gave the architect exactly the patterns it needed. The design document's two-part structure (Part 1: Architecture for builder, Part 2: Kernel Implementation for kernel writer) separated concerns cleanly.

### 5. Pipeline Speed -- 46 Minutes End-to-End

**Phase/Agent**: All
**Evidence**: First commit at 08:56 (analyzer), last at 09:42 (report). Compare with earlier runs: Feb 24 run took multiple hours with debugging, Mar 3 run took ~90 minutes, Mar 6 run took ~4 hours. This run is the fastest by far.
**Why it worked**: The accumulation of improvements across 6+ iterations: better reference selection, more precise design documents, proven helper library patterns, and (most importantly) no debugging cycles consuming time.

---

## 3. Issues Found

### Issue 1: Builder Uses Stick-Sized Pages for cb_in_rm Despite Design Specifying Tile Size

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) / Phase 4 Stage 0 (data_pipeline) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (kernel writer caught and fixed inline) |
| Time Cost | ~1-2 minutes of kernel writer's time |

**Problem**: The REPORT.md explicitly documents this as Upstream Fix #1: `"cb_in_rm page_size: Changed from stick_size to tile_page_size."` The architect's design (op_design.md line 168) specifies `"page_size MUST be tile_size"` in bold, and the hardware constraints checklist (line 147) repeats this. Yet the builder's initial stub used `stick_size`. The kernel writer silently fixed the `page_size=tile_size` in the program descriptor during Stage 0 implementation (commit `4e814aa`, line change in `layer_norm_rm_program_descriptor.py`).

**Root Cause**: This is a **known recurring issue** documented in MEMORY.md ("Tilize CB Page Size Requirement", discovered 2026-02-27). The builder (running on Sonnet per pipeline-improvements.md #6) does not reliably read and apply the architect's explicit CB page_size specifications. The design document stated the requirement in 3 separate places, yet the builder still defaulted to `stick_size`.

**Fix for agents**:
- **ttnn-generic-op-builder**: Add a mandatory validation step after CB generation: for any CB that feeds a `tilize` helper (identifiable from the design as the first CB in a tilize pipeline), assert `page_size == tile_size`. This could be a post-generation lint check.
- **Orchestrator**: Consider adding a static cross-check between op_design.md's CB table and the builder's generated program_descriptor.py before launching Phase 4 (relates to pipeline-improvements.md #9).

### Issue 2: cb_var_input Sizing Mismatch Between Design and Builder

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) / Phase 4 Stage 2 (variance_normalize) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (kernel writer caught and fixed inline) |
| Time Cost | <1 minute |

**Problem**: The REPORT.md documents Upstream Fix #4: `"cb_var_input sizing: Expanded from 1 tile to Wt tiles."` The design document (op_design.md line 87) specifies `cb_var_input` as `"1 page, streaming"` but the actual compute flow requires Wt pages because `square()` produces all Wt tiles before the reduce consumes them. The design document itself was slightly inconsistent -- it says "1 tile" in the CB table (line 87) but the Phase 4 pseudocode (line 297-298) implies `NoWaitNoPop` on cb_centered which means all Wt squared tiles must be buffered.

**Root Cause**: The architect's CB table listed `cb_var_input` as 1 page based on the streaming assumption, but the implementation pattern (square produces a full block before reduce consumes it) requires Wt pages. The design document should have specified this more precisely.

**Fix for agents**:
- **ttnn-operation-architect**: When specifying "streaming" CBs, explicitly verify that the producer and consumer operate at the same granularity. If the producer writes Wt tiles in a block before the consumer begins, the CB needs Wt pages even if conceptually it is "streaming."
- **ttnn-generic-op-builder**: When a CB is marked "streaming" with 1 page but its producer writes multiple tiles, flag a potential sizing issue.

### Issue 3: Epsilon Fill Location Confusion (Design vs Implementation)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2 (Design) / Phase 4 Stage 2 (variance_normalize) |
| Agent | ttnn-operation-architect |
| Retries Consumed | 0 |
| Time Cost | <1 minute (kernel writer resolved immediately) |

**Problem**: The design document (op_design.md lines 176, 401) presents multiple alternatives for epsilon handling without clearly selecting one: `"Filled by compute via fill_with_val"` (CB table), `"reader fills cb_eps once at startup (program lifetime)"` (Note 4), `"compute fills cb_eps locally each iteration using L1 write"` (Note 4 alternative). The REPORT.md documents Upstream Fix #3: `"Epsilon fill location: Moved from compute kernel to reader kernel. Compute kernels lack get_write_ptr() access."` The final implementation (reader kernel line 73-77) fills cb_eps in the reader.

**Root Cause**: The architect knew that `fill_with_val` / `get_write_ptr()` is a dataflow-only API but left deliberation text in the design document rather than resolving to a single approach. The CB table says "Filled by compute" but Note 4 discusses alternatives. This is exactly the kind of ambiguity that causes downstream confusion.

**Fix for agents**:
- **ttnn-operation-architect**: All "Notes" in the design document that discuss alternatives must resolve to a single chosen approach. Deliberation text (e.g., `"Alternative: ..."`, `"Better: ..."`) should be removed or moved to a separate rationale section. The kernel writer should receive a single unambiguous instruction per implementation decision.

### Issue 4: nblocks_per_core as Compile-Time vs Runtime Arg

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 2-3 (Design/Build) / Phase 4 Stage 0 |
| Agent | ttnn-operation-architect, ttnn-generic-op-builder |
| Retries Consumed | 0 |
| Time Cost | <1 minute |

**Problem**: The design document (op_design.md line 116) lists `nblocks_per_core` as a compile-time arg for the compute kernel. The REPORT.md documents Upstream Fix #2: `"nblocks_per_core: Moved from compile-time define to runtime arg. Cliff cores get different nblocks."` The compute kernel (line 53) reads it as `get_arg_val<uint32_t>(0)`.

**Root Cause**: The architect correctly identified that cliff cores handle remainder rows but placed `nblocks_per_core` in the compile-time args table. Since a single KernelDescriptor with one set of compile-time args is shared across all cores, per-core variation requires runtime args. This is a structural constraint of the `generic_op` ProgramDescriptor API.

**Fix for agents**:
- **ttnn-operation-architect**: Any parameter that varies per-core (nblocks, start offsets) must be in the runtime args section, never compile-time. Add a design-time check: "Does this value differ between core groups? If yes, it must be a runtime arg."

### Issue 5: Analysis Documents Are Excessively Large (75KB Total)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 1 (Analysis) |
| Agent | ttnn-operation-analyzer |
| Retries Consumed | N/A |
| Time Cost | ~1-2 minutes extra context consumption in architect |

**Problem**: The three analysis files total 75KB (tilize: 20KB/326 lines, untilize: 22KB/436 lines, batch_norm: 33KB/639 lines). These are consumed by the architect as input context. The REPORT.md Pain Points section notes: `"Even with role-based focus, analyses are 20-33KB each. Could be trimmed further."` The architect's breadcrumbs show it extracted ~5-6 key findings per analysis, meaning the signal-to-noise ratio is low.

**Root Cause**: The analyzer's instruction set does not impose a length cap or structured output format. Each analysis is free-form markdown that includes full code snippets, extensive explanations, and repeated context.

**Fix for agents**:
- **ttnn-operation-analyzer**: Add a target size limit (e.g., 8KB per analysis) or switch to a structured output format (JSON or table-based markdown with defined sections). The architect only needs: CB layout, key helper signatures, data flow pattern, and work distribution strategy -- not full source code listings.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~3.5m (09:25-09:28:49) | 0 free, 0 hard | PASS | Clean -- implemented reader, compute (tilize+untilize), writer in one pass |
| reduce_mean | ~1.5m (09:28:49-09:30:21) | 0 free, 0 hard | PASS | Clean -- added reduce+sub phases, only modified compute kernel |
| variance_normalize | ~3.7m (09:30:21-09:34:03) | 0 free, 0 hard | PASS | Clean -- most complex stage (square, reduce, add+rsqrt, mul), plus reader epsilon fill |
| affine_transform | ~4.3m (09:34:03-09:38:21) | 0 free, 0 hard | PASS | Clean -- added gamma/beta read + tilize + affine compute. Longest due to reader refactoring (stick replication). |

### Time Sinks

There were no significant time sinks in this run. Every stage passed on first attempt. The distribution across stages is proportional to their complexity.

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Build phase | builder | 12m 11s | 26% | Longest single agent. Generates 11 files, runs integration tests. | 0 | Builder is inherently slow: lots of file generation + test validation. |
| 2 | Affine stage | kernel-writer | 4m 18s | 9% | Most complex TDD stage -- reader changes + compute changes + CB routing. | 0 | Complexity proportional to delta (gamma/beta reading, L1-to-L1 stick replication). |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| (None in this run) | N/A | N/A | N/A |

No work was discarded in this run. All code written passed tests on the first attempt. However, across the broader history (6+ runs), the same operation was rebuilt from scratch each time the pipeline was re-run. The earlier runs on Feb 24 (with debugging), Mar 3, Mar 5, Mar 6, and Mar 9 represent significant cumulative effort. The pipeline has no ability to reuse artifacts from prior runs.

---

## 5. Inter-Agent Communication Issues

### Handoff 1: orchestrator -> ttnn-operation-analyzer

| Field | Value |
|-------|-------|
| Artifact Passed | Reference operation paths + role assignments |
| Quality | GOOD |
| Issues | None -- roles (input_stage, compute_core, output_stage) were clear |
| Downstream Impact | None |
| Suggestion | None needed |

### Handoff 2: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | `tilize_analysis.md`, `untilize_analysis.md`, `batch_norm_analysis.md` |
| Quality | ADEQUATE |
| Issues | Analysis documents are 75KB total with low signal-to-noise ratio. The architect extracted ~15 key findings across all three. |
| Downstream Impact | Extra context consumption but no confusion -- architect breadcrumbs show smooth extraction of key patterns. |
| Suggestion | Reduce analysis size to ~8KB per document by using structured output format. |

### Handoff 3: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md` (Part 1: Architecture) |
| Quality | GOOD (with caveats) |
| Issues | Builder ignored explicit `page_size = tile_size` requirement for cb_in_rm despite it being stated 3 times. Builder also used 1-tile sizing for cb_var_input despite the data flow implying Wt tiles. |
| Downstream Impact | Two bugs in program descriptor that the kernel writer had to fix (Issues 1 and 2 above). No retries consumed since kernel writer caught them inline. |
| Suggestion | Add automated cross-validation between design CB table and builder's generated CBDescriptor calls. |

### Handoff 4: ttnn-operation-architect -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md` (Part 2: Kernel Implementation) |
| Quality | GOOD |
| Issues | Minor ambiguity in epsilon fill location (Issue 3) and nblocks compile-time vs runtime (Issue 4). Both were resolved instantly by the kernel writer. |
| Downstream Impact | Negligible -- kernel writer made the correct choices without delay. |
| Suggestion | Resolve all alternative approaches in the design document before handoff. |

### Handoff 5: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files |
| Quality | ADEQUATE |
| Issues | Two program descriptor bugs (cb_in_rm page_size, cb_var_input sizing). Stub kernels were minimal but structurally correct. |
| Downstream Impact | Kernel writer had to fix program descriptor during Stage 0 and Stage 2 (Issues 1 and 2). |
| Suggestion | Builder should validate CB configurations against the design document's explicit constraints. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

No agents filed `upstream_feedback` breadcrumb events in this run. The architect's execution log Section 7 states `"None -- instructions were sufficient."` This is the cleanest run observed.

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-generic-op-builder | REPORT.md | Respect architect's explicit CB page_size specifications | HIGH | HIGH |
| ttnn-operation-architect | REPORT.md | Resolve epsilon fill location ambiguity before handoff | MEDIUM | LOW |
| ttnn-operation-analyzer | REPORT.md | Reduce analysis document size (~8KB target) | MEDIUM | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Build | Builder ignores architect's CB constraints | Add static cross-validation step between Phase 2 output and Phase 3 output | HIGH |
| Design | Unresolved alternatives in design doc | Architect instructions should mandate single-choice resolution for all decisions | MEDIUM |
| Analysis | 75KB total analysis for ~15 key findings | Structured output format with size cap | MEDIUM |
| TDD | Epsilon fill convention re-discovered each run | Codify as a standard pattern: "scalar constants are filled by reader, not compute" | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | NO | Zero debugging in this run. The helper library eliminated numerical issues entirely. |
| 2 | Too many planning stages | NO (DONE) | Merged architect working well -- single agent produced design in 7 minutes. |
| 3 | .tdd_state.json coupling fragile | NO | TDD state was correct throughout. No schema issues. |
| 4 | No fast path for simple operations | N/A | layer_norm_rm is medium complexity (appropriate for full pipeline). |
| 6 | Builder runs on Sonnet | YES | Builder still missed explicit CB page_size requirements. This aligns with the concern about Sonnet's detail sensitivity. |
| 7 | Discovery keyword matching | NO | Discovery chose correct references. |
| 9 | No architect/builder cross-validation | YES | cb_in_rm page_size and cb_var_input sizing mismatches were not caught between Phase 2 and Phase 3. Kernel writer caught them in Phase 4. |
| 11 | No incremental re-run | N/A | Not needed in this run since all stages passed. But across 6+ runs of this operation, re-running from scratch each time is wasteful. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Unresolved design alternatives confuse downstream agents | The architect leaves deliberation text in the design document (e.g., "Alternative: ...", "Better: ...") instead of resolving to a single approach. Example: epsilon fill location in op_design.md Note 4 presents 3 options without choosing one. | MEDIUM |
| Per-core-varying args placed in compile-time section | The architect placed `nblocks_per_core` (which varies between core groups) in the compile-time args table. This is structurally incorrect for the generic_op API where a single KernelDescriptor shares compile-time args across all cores. | LOW |

---

## 8. Actionable Recommendations

### Recommendation 1: Add Post-Build CB Cross-Validation

- **Type**: new_validation
- **Target**: Orchestrator script (between Phase 3 and Phase 4)
- **Change**: After the builder commits, parse both `op_design.md` (CB table) and the generated `program_descriptor.py` (CBDescriptor calls). Verify that: (a) all CB IDs match, (b) page_size values match, (c) total_size values are compatible. Flag mismatches before launching Phase 4.
- **Expected Benefit**: Catches Issues 1 and 2 automatically, preventing kernel writer from needing to fix infrastructure bugs. Would have saved ~2 minutes in this run, but could save 10+ minutes (or hard retries) in runs where the mismatch causes hangs or silent numerical errors.
- **Priority**: HIGH
- **Effort**: MEDIUM

### Recommendation 2: Mandate Single-Choice Resolution in Design Documents

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add instruction: "Every design decision must resolve to exactly ONE approach. If multiple alternatives are considered during design, document the chosen approach only. Move alternatives to a separate 'Design Rationale' appendix if needed. The kernel writer must never encounter text like 'Alternative: ...' or 'Better: ...' in the main design sections."
- **Expected Benefit**: Eliminates epsilon-fill-style ambiguities. Reduces cognitive load on kernel writer.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Cap Analysis Document Size

- **Type**: instruction_change
- **Target**: ttnn-operation-analyzer agent prompt
- **Change**: Add instruction: "Analysis output must not exceed 10KB. Use a structured format with defined sections: (1) CB Layout Table, (2) Key Helper Signatures, (3) Data Flow Diagram, (4) Work Distribution Strategy, (5) Critical Implementation Notes. Do not include full source code listings -- reference file paths and line ranges instead."
- **Expected Benefit**: Reduces context consumption in the architect from ~75KB to ~30KB. Faster architect startup, lower token cost.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Codify Scalar Constant Fill Convention

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt, kernel_lib documentation
- **Change**: Add standard pattern documentation: "Scalar constants (epsilon, reduce scalers, etc.) that need to be written to a CB must be filled by the **reader kernel** (dataflow), not the compute kernel. Compute kernels do not have `get_write_ptr()` access. Use `prepare_reduce_scaler<cb_id>(value)` for reduce scalers and a custom `fill_cb_with_val_bfloat16()` function in the reader for other scalars."
- **Expected Benefit**: Prevents rediscovery of "compute can't write to L1" on every operation. Saves 1-2 minutes per new operation and eliminates a potential debugging cycle.
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 5: Add Compile-Time vs Runtime Arg Classification Rule

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Add design rule: "A kernel argument MUST be a runtime arg if it varies between cores (e.g., `nblocks_per_core`, `start_stick_id`). It may be a compile-time arg only if the value is identical across all cores in all core groups. The generic_op ProgramDescriptor uses a single KernelDescriptor per kernel type, so compile-time args are shared."
- **Expected Benefit**: Eliminates the nblocks compile-time/runtime confusion. Prevents potential cliff-core bugs.
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5/5 | Correctly selected batch_norm over softmax as compute reference. All 3 references were directly useful. |
| Analysis quality | 4/5 | Content quality is high (architect extracted all needed patterns), but documents are too large (75KB). |
| Design completeness | 4/5 | Comprehensive 412-line design covering all 8 phases, 13 CBs, 4 TDD stages. Minor issues: unresolved alternatives (epsilon), nblocks placement. |
| Build correctness | 3/5 | Structural scaffolding was correct (11 files, imports, test structure). Two CB configuration bugs (page_size, var_input sizing) were carried to Phase 4. This is a recurring issue. |
| Kernel implementation | 5/5 | Zero retries, zero debugging, all 4 stages first-try. 411 lines of kernel code produced in 14 minutes. Best-case scenario. |
| Inter-agent communication | 4/5 | No upstream_feedback events. Design-to-implementation fidelity was high. Two builder bugs were minor and caught immediately. |
| Logging/observability | 3/5 | Breadcrumbs present for all 4 agents with timestamps. However: no execution logs for builder or kernel writer (only architect has one), breadcrumbs lack detail (no per-file-read logging for builder/kernel-writer), and there are no test output breadcrumbs (just pass/fail). |

### Top 3 Things to Fix

1. **Add post-build CB cross-validation** (Recommendation 1) -- the builder's CB configuration bugs are the most consistent failure pattern across runs. Automated validation would catch them before Phase 4, saving time and preventing potential hard retries in more complex operations.

2. **Mandate single-choice resolution in design documents** (Recommendation 2) -- unresolved alternatives create unnecessary ambiguity. This is cheap to fix (instruction change only) and improves every downstream handoff.

3. **Cap analysis document size** (Recommendation 3) -- 75KB of analysis for ~15 key findings is inefficient. Structured output with a size cap would reduce context consumption by 50%+ without losing signal.

### What Worked Best

The **kernel helper library** was the single strongest contributor to this run's success. The `tilize_helpers`, `untilize_helpers`, `reduce_helpers_compute`, and `binary_op_helpers` abstractions allowed the kernel writer to express all 8 compute phases using high-level helper calls (tilize, reduce, sub, square, add, mul, untilize) without managing tile registers, DEST accumulator, or CB sync at the low level. This is what enabled 0 retries across all 4 TDD stages and reduced the entire Phase 4 to 14 minutes. For context, earlier runs of the same operation (before the helper library matured) took 30-90 minutes on Phase 4 with multiple debugging cycles.
