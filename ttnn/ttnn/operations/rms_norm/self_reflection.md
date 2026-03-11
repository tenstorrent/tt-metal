# Self-Reflection: rms_norm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `rms_norm` |
| Operation Path | `ttnn/ttnn/operations/rms_norm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 10 |
| Total Pipeline Duration | ~148 min (17:04 - 19:32 UTC) |
| Overall Result | PARTIAL -- 3/4 TDD stages passed; gamma stage unresolved after 87 minutes of debugging |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~4m | COMPLETE | Hybrid mode selected (tilize + reduce_w + untilize) |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~13m (parallel) | COMPLETE | All 3 analyses completed (tilize, reduce_w, untilize) |
| 2: Design | ttnn-operation-architect | ~8m | COMPLETE | op_design.md produced with 4 TDD stages, 10 CBs |
| 3: Build | ttnn-generic-op-builder | ~9m | COMPLETE | 9 files created, 1 compile fix, all stubs pass |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~111m | PARTIAL | 3/4 stages passed, gamma stage in_progress (hung) |
| 5: Report | orchestrator | ~2m | COMPLETE | REPORT.md generated |

### Agent Duration Breakdown

Duration method: Breadcrumb `"event":"start"` to `"event":"complete"` timestamps. For the kernel writer, `complete` was never emitted -- last breadcrumb timestamp (19:31:49) used as end time.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 17:04:45 | 17:13:37 | ~9m | 0 | ~9m active |
| ttnn-operation-analyzer (reduce_w) | 17:04:33 | 17:16:44 | ~12m | 0 | ~12m active |
| ttnn-operation-analyzer (untilize) | 17:04:26 | 17:17:15 | ~13m | 0 | ~13m active |
| ttnn-operation-architect | 17:19:18 | 17:27:19 | ~8m | 0 | ~8m active |
| ttnn-generic-op-builder | 17:29:44 | 17:38:39 | ~9m | 1 (compile fix) | ~8m active, ~1m debugging |
| ttnn-kernel-writer-tdd | 17:40:40 | 19:31:49 | ~111m | 7 total | ~24m active, ~87m debugging gamma |

### Duration Visualization

```
Phase 0  |##|                                                            (~4m)
Phase 1  |############|                                                  (~13m, 3 analyzers parallel)
Phase 2       |########|                                                 (~8m)
Phase 3            |#########|                                           (~9m)
Phase 4                      |###############################################| (~111m)
             |--data_pipeline--|                                          (~6m)
                    |--sq_reduce_rsqrt--|                                 (~12m, 3 failures)
                            |--normalize--|                              (~6m)
                                    |------------ gamma (UNRESOLVED) ------->  (~87m, 4+ failures)
Phase 5                                                                  |##| (~2m)
         0    10   20   30   40   50   60   70   80   90  100  110  120  min

Longest phase: Phase 4 (111m) -- gamma stage debugging consumed 78% of Phase 4 time.
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~17m | 11% | 3 analyzers, parallel execution |
| Design (Phase 2) | ~8m | 5% | |
| Build (Phase 3) | ~9m | 6% | 1 retry (TensorAccessor include path) |
| Kernel implementation (Phase 4) | ~111m | 75% | 4 TDD stages |
| -- Productive coding | ~24m | 16% | Stages 1-3 implementation + gamma initial impl |
| -- Debugging/retries | ~87m | 59% | Gamma stage: 4+ hypothesis-fix-retest cycles |
| Reporting (Phase 5) | ~2m | 1% | |
| **Total** | **~148m** | **100%** | |

---

## 2. What Went Well

### 1. Stages 1 and 3 Passed First Attempt

**Phase/Agent**: Phase 4 -- ttnn-kernel-writer-tdd
**Evidence**: `.tdd_state.json` shows `data_pipeline` and `normalize` both have `"attempts": 0` (0 failures) and `"status": "passed"`. The breadcrumbs confirm single test runs at 17:46:04 and 18:04:40, both passing on first attempt.
**Why it worked**: The design document was highly specific for these stages. The data_pipeline stage was a straightforward tilize/copy/untilize chain. The normalize stage only required changing the square phase from `WaitAndPopPerTile` to `WaitUpfrontNoPop` and adding a `mul<COL>` phase -- both precisely described in op_design.md Phase 5 specification (lines 305-326).

### 2. High-Quality Analysis Documents

**Phase/Agent**: Phase 1 -- ttnn-operation-analyzer (x3)
**Evidence**: The `reduce_w_analysis.md` is 595 lines covering both the matmul and reduce helper paths, with correct identification of `prepare_reduce_scaler` as the recommended approach (vs `generate_mm_scaler`). The `tilize_analysis.md` correctly documented the 32-stick batching pattern. The `untilize_analysis.md` correctly documented the raw L1 pointer extraction pattern for RM sticks. All three directly informed the architect's design decisions.
**Why it worked**: The analyzers focused on their assigned roles (input_stage, compute_core, output_stage) and extracted the specific patterns needed for a new operation.

### 3. Clean CB Layout for Stages 1-3

**Phase/Agent**: Phase 2 (architect) and Phase 4 (kernel writer)
**Evidence**: The original 10-CB layout (c_0 through c_17) required zero CB-related bugs for the first 3 TDD stages. The only CB issue was c_3 page count (2 vs Wt), which is a sizing issue, not a layout issue. The architect's CB state tables after each phase (op_design.md lines 234-326) enabled the kernel writer to verify push/pop balance at each stage (breadcrumb `cb_sync_check` events at 17:43:58, 17:48:53, 18:00:34 all show `balanced:true`).
**Why it worked**: The architect systematically documented CB state after each phase, making balance verification straightforward.

### 4. Efficient Hypothesis-to-Fix Cycles in Stage 2

**Phase/Agent**: Phase 4, stage square_reduce_rsqrt -- ttnn-kernel-writer-tdd
**Evidence**: All 3 failures in stage 2 were diagnosed with HIGH confidence hypotheses. H1 (ttnn.Shape slice issue, 17:50:02) was fixed in 14 seconds. H2 (writer start_id using input Wt, 17:51:52) was fixed in 20 seconds. H3 (c3 page count, 17:55:47) was fixed in 18 seconds. Total debugging time for 3 failures: ~8 minutes including test runs.
**Why it worked**: Each failure had a clear error signature. The TypeError was self-explanatory. The multi-core-only crash pointed directly to per-core offset computation. The watcher triage output identified the stuck CB (c3) and operation (square/reduce).

### 5. Builder's Proactive Test Fix

**Phase/Agent**: Phase 3 -- ttnn-generic-op-builder
**Evidence**: The builder identified and fixed 5 issues in the architect's stage test templates (phase3_builder.md, Section 1): Markup syntax errors, undefined defaults, relative imports, undefined variable references, and extraneous `layout=layout` parameters. This prevented the kernel writer from encountering broken test files.
**Why it worked**: The builder ran the integration test suite before committing, caught the compile error, and proactively reviewed test files for correctness.

---

## 3. Issues Found

### Issue 1: CB Reuse Design Caused 87-Minute Gamma Debugging Spiral

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- gamma |
| Agent | ttnn-kernel-writer-tdd (implementation), ttnn-operation-architect (root cause) |
| Retries Consumed | 4 hard attempts consumed |
| Time Cost | ~87 minutes (18:05:17 to 19:31:49) |

**Problem**: The architect's design specified that gamma multiply output should go to c_4 (reusing the reduce output CB): "c_4 is reused as gamma output (2 pages, double-buffered streaming)" (op_design.md line 343). This created a complex CB reuse pattern where c_4 served as reduce output in Phase 3, was freed, then served as gamma multiply output in Phase 6. The kernel writer implemented this, but it produced consistent numerical mismatches (max diff 6.4375 across 3 attempts at 18:16:45, 18:18:30, 18:51:04). The design itself acknowledged the complexity: "Alternatively, add a dedicated cb_gamma_out. For simplicity, the kernel writer should use c_4 as the gamma multiply output" (line 343). This "simplicity" proved to be anything but simple.

The kernel writer went through 6+ hypotheses and approaches:
1. ROW broadcast row-0 semantics (18:17:00) -- replicated gamma sticks 32 times
2. Changed ROW to NONE broadcast (18:23:40)
3. Added DPRINT debugging (18:26:01-18:47:27, ~25 min)
4. Diagnostic test passed but official test failed (18:48:55 vs 18:51:04) -- suspected kernel cache
5. Changed input policy to WaitUpfrontPopAtEnd (19:01:51)
6. Introduced CB8 as a new intermediate (19:24:04) -- this caused a hang (19:31:49)

**Root Cause**: The design's CB reuse pattern (c_4 for both reduce output and gamma output) was fragile. When the kernel writer attempted to implement it, the interplay between Phase 3's reduce output (pushing 1 tile to c_4), Phase 4's epsilon add (popping that tile from c_4), and Phase 6's gamma multiply (pushing Wt tiles to c_4 with only 2 pages allocated) created sizing and synchronization issues. The design noted c_4 needed only 2 pages, but gamma output requires Wt pages. This mismatch was not caught at design time.

Additionally, the diagnostic test passing while the official test failed (timestamps 18:48:55 vs 18:51:04) strongly suggests that DPRINT side effects (extra cb_wait_front calls) were masking a real synchronization issue. The kernel writer correctly identified this possibility (breadcrumb 19:01:35) but could not resolve it.

**Fix for agents**:
- **ttnn-operation-architect**: When a CB is reused across phases with different page count requirements, explicitly note the maximum page count needed and add a "CB reuse validation" section. For this case, c_4 needed max(2, Wt) = Wt pages. Better yet, prefer dedicated CBs over reuse when the reuse crosses distinct computational phases. The design should have recommended a dedicated cb_gamma_out (c_8) from the start rather than suggesting c_4 reuse "for simplicity."
- **ttnn-kernel-writer-tdd**: When DPRINT causes a test to pass that otherwise fails, this is a strong signal of a synchronization/timing issue. The agent should immediately focus on CB wait/pop policy mismatches rather than continuing to modify broadcast types or other parameters. Add an explicit rule: "If adding debug prints changes test outcome, diagnose the CB synchronization boundary, not the data flow."

### Issue 2: Design Specified c_3 with 2 Pages but Sequential Square+Reduce Requires Wt Pages

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- square_reduce_rsqrt |
| Agent | ttnn-operation-architect (design), ttnn-kernel-writer-tdd (fixed it) |
| Retries Consumed | 2 hard attempts (hangs at 17:51:44, 17:55:40) |
| Time Cost | ~8 minutes |

**Problem**: The design doc specified c_3 (cb_sq) with "2 | Streaming" pages (op_design.md line 76). The kernel writer implemented square with `PerTile` output policy and reduce with `WaitAndPopPerTile` input policy, which are supposed to stream. However, the square helper pushes all Wt tiles before reduce begins consuming (they are sequential calls, not concurrent). With only 2 pages, the pack thread blocks at `cb_reserve_back(c3)` after pushing 2 tiles when Wt > 2 (e.g., Wt=4 for shape 1x1x64x128). The watcher triage confirmed: "TRISC2(pack) stuck at cb_reserve_back(c3) in square. TRISC0(unpack) stuck at cb_wait_front(c3) in reduce" (breadcrumb 17:55:40).

**Root Cause**: The architect labeled c_3 as "Streaming" with 2 pages, which is correct for true streaming (when producer and consumer run concurrently on different threads). But when both square and reduce are sequential helper calls within the same compute kernel, they execute on the same set of TRISC threads. The pack thread from square cannot interleave with the unpack thread from reduce -- they run in sequence. The architect's streaming assumption was incorrect for sequential helper calls.

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "If two helper calls are sequential (not concurrent on different kernels), their intermediate CB must hold the full output of the producer. Streaming (2-page) sizing only works when producer and consumer are concurrent (e.g., reader and compute, or compute and writer)." This should be a checklist item in the design template.

### Issue 3: Broadcast Mode Confusion (NONE vs ROW) Wasted 25+ Minutes of DPRINT Debugging

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- gamma |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | Part of the 87-minute gamma debugging |
| Time Cost | ~25 minutes (18:23:40 to 18:49:49) on DPRINT investigation |

**Problem**: The design specified `BroadcastDim::ROW` for gamma multiply (op_design.md line 329). The kernel writer initially implemented ROW broadcast but got numerical mismatches. The writer then hypothesized that ROW broadcast "was producing mostly zeros" (breadcrumb 18:23:40) and switched to `BroadcastDim::NONE`. A diagnostic test with DPRINT passed with NONE broadcast (max diff 0.09375), but the official test failed with NONE broadcast (max diff 6.4375). The writer then switched back to ROW (breadcrumb 19:06:39). This back-and-forth consumed significant time.

**Root Cause**: The kernel writer lacked a definitive understanding of what ROW broadcast does at the hardware level. The hypotheses alternated between "ROW reads only row 0" and "ROW reads per-row data from B tile faces." The fact that the diagnostic test passed while the official test failed was not a broadcast mode issue at all -- it was the DPRINT-induced synchronization side effect (Issue 1). The broadcast mode changes were a red herring.

**Fix for agents**:
- **ttnn-operation-architect**: In the Binary Op Broadcast Verification table, add a comment for each broadcast mode explaining the hardware semantics: "ROW broadcast: LLK reads row 0 of each B tile face and replicates it across all rows during unpack. B tile must have valid data in row 0. Other rows are ignored." This would have prevented the confusion.
- **ttnn-kernel-writer-tdd**: When a code change does not alter test results (max diff 6.4375 before and after), stop investigating that parameter and focus on other dimensions. Add a rule: "If switching between two approaches produces identical error magnitude, the bug is elsewhere."

### Issue 4: Device Contention Extended Pipeline by 60+ Minutes

| Field | Value |
|-------|-------|
| Severity | HIGH (infrastructure) |
| Phase / TDD Stage | Phase 4 -- gamma (and post-pipeline verification) |
| Agent | N/A (infrastructure) |
| Retries Consumed | 0 (but blocked test execution) |
| Time Cost | ~60+ minutes of device lock waits |

**Problem**: The REPORT.md documents that another agent process on the same machine held the device lock for 60+ minutes during golden tests, blocking the TDD agent's test runs and the orchestrator's post-pipeline verification. The TDD agent's context window was partially exhausted by device lock waits between test attempts (REPORT.md line 138-139).

**Root Cause**: The `tt-test.sh` script uses `flock` for device access, but there is no cooperative scheduling mechanism. A long-running golden test suite on another agent can monopolize the device for over an hour.

**Fix for agents**:
- **Infrastructure**: Implement a device scheduling queue with timeout-based preemption. If an agent holds the device for more than N minutes, the lock should be released and re-acquired per test case rather than held for the entire suite.
- **Orchestrator**: When spawning multiple agents on the same machine, stagger Phase 4 starts or assign agents to different devices if available.

### Issue 5: Builder Set c_3 to 2 Pages Despite Design Saying "Streaming"

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 0 (caught in Phase 4) |
| Time Cost | Indirect: contributed to Issue 2 |

**Problem**: The builder set c_3 (cb_sq) to 2 pages (phase3_builder.md, CB Configuration table). This matched the architect's design doc specification (op_design.md line 76: "c_3 | cb_sq | Squared tiles | Compute (square) | Compute (reduce) | 2 | Streaming"). The builder faithfully implemented the design, but the design was wrong.

**Root Cause**: No cross-validation between the architect's CB sizing assumptions and the actual helper call sequence. The architect assumed streaming (2 pages) but the helpers execute sequentially.

**Fix for agents**:
- **Pipeline infrastructure (known issue #9)**: Add a static cross-check between the CB table in `op_design.md` and the actual helper call patterns. If a CB is labeled "Streaming" but both its producer and consumer are compute-kernel helpers (not reader/writer concurrent), flag it as requiring full-buffer sizing.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~6m (17:40:40-17:46:10) | 0 free, 0 hard | PASS | Clean -- first attempt pass |
| square_reduce_rsqrt | ~12m (17:46:51-17:58:56) | 0 free, 3 hard | PASS | CB hang debugging (c_3 sizing) |
| normalize | ~6m (17:59:35-18:04:48) | 0 free, 0 hard | PASS | Clean -- first attempt pass |
| gamma | ~87m (18:05:17-19:31:49) | 0 free, 4+ hard | IN_PROGRESS | CB reuse + broadcast confusion + kernel cache |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Gamma debugging | ttnn-kernel-writer-tdd | ~87m | 59% | Oscillating between broadcast modes, DPRINT investigation, CB architecture redesign | 4+ hard | CB reuse design flaw + DPRINT side effects masking root cause |
| 2 | Device contention | Infrastructure | ~60m+ | ~40% (overlapping) | Lock waits for device during gamma stage and post-pipeline | N/A | No cooperative device scheduling |
| 3 | DPRINT debugging | ttnn-kernel-writer-tdd | ~25m | 17% | Adding/removing debug prints across compute and writer kernels | 0 | Kernel writer had no better diagnostic tool |
| 4 | Stage 2 hangs | ttnn-kernel-writer-tdd | ~8m | 5% | Two CB deadlocks from c_3 under-sizing | 2 hard | Design specified 2 pages, needed Wt |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | Changed ROW to NONE broadcast (18:23:40) then back to ROW (19:06:39) | The broadcast mode was not the problem; DPRINT side effects masked the real issue | Document broadcast hardware semantics in design; add rule about identical error magnitudes |
| ttnn-kernel-writer-tdd | Added extensive DPRINT to both compute and writer kernels (18:26:01-18:47:27) | While the diagnostic test passed, removing DPRINT caused re-failure, proving DPRINT was a side effect not a diagnostic | Need a synchronization-safe debug methodology that does not add extra CB waits |
| ttnn-kernel-writer-tdd | Implemented CB8 architecture redesign (19:24:04-19:25:56) | Caused a new hang (19:31:49), adding a new problem without solving the original | Validate CB push/pop balance before testing architectural changes; the original c_4 reuse approach might have worked with correct page sizing |
| ttnn-kernel-writer-tdd | Gamma stick replication fix (18:17:40) | Did not change the error at all (same 6.4375 max diff), proving the issue was not in gamma data loading | When a fix produces identical error magnitude, immediately discard that hypothesis |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, reduce_w_analysis.md, untilize_analysis.md |
| Quality | GOOD |
| Issues | None significant. All three analyses were thorough and well-structured. |
| Downstream Impact | Architect made correct design decisions for tilize/untilize helpers and reduce pattern. |
| Suggestion | No changes needed. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md + .tdd_state.json |
| Quality | ADEQUATE |
| Issues | (1) c_3 CB sized at 2 pages with "Streaming" label, which was wrong for sequential helper calls. (2) Stage test templates had 5 syntax/logic errors that the builder had to fix. (3) The gamma multiply CB reuse (c_4) with only 2 pages was a design-time error that propagated downstream. |
| Downstream Impact | Builder faithfully implemented the wrong c_3 sizing. The c_4 reuse pattern was handed to the kernel writer who spent 87 minutes unable to make it work. |
| Suggestion | Add a "CB Page Count Validation" section to the design template where the architect explicitly traces the maximum concurrent pages for each CB across all phases. Add automated validation of stage test templates before handoff. |

### Handoff 3: ttnn-operation-architect -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 (Kernel Implementation) |
| Quality | ADEQUATE for stages 1-3, POOR for stage 4 |
| Issues | (1) Phase 6 gamma multiply design had unresolved ambiguity: "Alternatively, add a dedicated cb_gamma_out" (line 343) left the kernel writer to decide between c_4 reuse and a new CB. (2) The design's "Revised approach" paragraph (lines 343-344) contradicted the earlier c_4 specification without clearly resolving which approach to use. (3) No explicit documentation of ROW broadcast hardware semantics. |
| Downstream Impact | The kernel writer spent 87 minutes trying to make the c_4 reuse work, then independently arrived at the CB8 approach (which the architect hinted at as an alternative), but the architectural change introduced a new hang. |
| Suggestion | Architect must resolve all design alternatives before handoff. Remove deliberation text ("Alternatively...") from the final design. If there are two approaches, choose one and document why. |

### Handoff 4: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files |
| Quality | GOOD |
| Issues | Minor: the builder's execution log correctly documented all CB indices and runtime args, which the kernel writer relied on. The builder proactively fixed test templates. |
| Downstream Impact | Kernel writer had clean test files and correct program descriptor structure. The only builder-originated issue was c_3 sizing (2 pages), which was the architect's specification. |
| Suggestion | No changes needed to builder handoff. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-kernel-writer-tdd (breadcrumbs) | Resolve all design alternatives before handoff; remove deliberation text ("Alternatively...", "For simplicity...") | H | H |
| ttnn-operation-architect | ttnn-kernel-writer-tdd (stage 2 fix) | Add sequential-helper CB sizing rule: intermediate CBs between sequential compute helpers need full producer output capacity, not streaming 2-page capacity | H | H |
| ttnn-operation-architect | Analysis of gamma debugging | Document broadcast hardware semantics in Binary Op Broadcast Verification table: what ROW/COL/SCALAR/NONE actually do at the LLK unpack level | M | M |
| ttnn-kernel-writer-tdd | Self-analysis of gamma debugging | Add diagnostic rules: (1) identical error magnitude across fixes = wrong hypothesis dimension; (2) DPRINT-sensitive test = synchronization issue, not data issue; (3) max 2 broadcast mode changes before escalating to CB architecture review | H | H |
| ttnn-generic-op-builder | Builder execution log | No changes needed; builder correctly implemented spec and proactively fixed test issues | - | - |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD | Gamma stage consumed 87 min (59% of total) with no resolution | Add a "complexity escalation" rule: after 3 hard failures on the same stage with the same error magnitude, the kernel writer should stop and emit an upstream_feedback event requesting design clarification | H |
| Infrastructure | Device contention blocked testing for 60+ min | Implement cooperative device scheduling with per-test-case lock acquisition instead of per-suite | H |
| Design | CB reuse across phases with different page requirements caused a cascading debugging spiral | Add explicit CB lifecycle analysis to design template: for each CB, list all phases that use it and the maximum pages needed across all phases | M |
| Logging | No execution_log.md files were generated for analyzer, architect, or kernel writer agents | Ensure all agents produce execution_log.md with structured recovery summaries | M |
| Diagnostics | DPRINT debugging took 25 min and produced misleading results (side effects changed test outcome) | Develop a synchronization-safe diagnostic methodology: e.g., post-test CB state dump that does not affect runtime behavior | M |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | YES | Gamma stage: 87 minutes, multiple hypothesis cycles, DPRINT investigation. Exactly the pattern described: "Numerical mismatches are extremely hard to debug, this is where the time burns." |
| 3 | `.tdd_state.json` coupling fragile | NO | TDD state file worked correctly throughout. |
| 4 | No fast path for simple operations | NO | rms_norm is a medium-complexity op; full pipeline appropriate. |
| 6 | Builder runs on Sonnet | PARTIALLY | Builder had one compile error (TensorAccessor include path) but recovered quickly. The 5 test template fixes suggest the builder did detailed work. |
| 7 | Discovery uses keyword matching | NO | Discovery correctly selected hybrid mode with 3 references. |
| 9 | No validation between architect and builder output | YES | c_3 CB sizing mismatch: architect specified 2 pages, which was wrong, and builder faithfully implemented it. A cross-validation check would have caught "Streaming CB between two compute helpers needs Wt pages, not 2." |
| 11 | No incremental re-run capability | YES | After the gamma stage hung at 19:31:49, there was no way to resume from the gamma stage with the corrected kernels. The entire TDD phase would need to be restarted. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| DPRINT side effects mask synchronization bugs | Adding `cb_wait_front` calls via DPRINT changes the timing/synchronization of CB operations, causing tests to pass that would otherwise fail. The kernel writer spent 25 minutes investigating a "fix" that only worked because of DPRINT side effects. Need a diagnostic approach that does not modify CB synchronization. | H |
| Design deliberation text propagates to implementation as ambiguity | The architect's "Alternatively, add a dedicated cb_gamma_out. For simplicity..." text left an unresolved design choice in the handoff document. The kernel writer tried both approaches, wasting significant time. Design documents must be decisional, not deliberative. | M |
| Gamma CB reuse pattern is a recurring trap | Reusing a CB across phases (c_4 for reduce output then gamma output) with different page requirements is error-prone. The design should either use dedicated CBs or explicitly validate page count compatibility across phases. | M |
| Device contention during TDD extends context window | When the device is locked by another agent, the TDD agent's context window fills with wait/retry messages rather than productive debugging. The agent cannot checkpoint and resume later. | M |

---

## 8. Actionable Recommendations

### Recommendation 1: Eliminate CB Reuse Across Phases with Different Page Requirements

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt / design template
- **Change**: Add a design rule: "Never reuse a CB across phases if the phases require different page counts. Assign a dedicated CB for each distinct usage. CB IDs 0-31 are available; there is no need to conserve them." Add a CB lifecycle table to the design template where each CB row lists all phases that use it and the maximum pages across all phases.
- **Expected Benefit**: Eliminates the class of bugs where Phase N's push/pop state interferes with Phase M's usage of the same CB. Would have prevented the 87-minute gamma debugging spiral.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add Sequential-Helper CB Sizing Rule

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt
- **Change**: Add to the architect's CB sizing checklist: "If two compute helpers are called sequentially within the same kernel (not reader+compute or compute+writer), the intermediate CB must hold the full output of the producer. Streaming (2-page double-buffered) sizing only works for concurrent producer-consumer pairs on different threads." Include examples showing correct vs incorrect sizing.
- **Expected Benefit**: Prevents CB deadlocks caused by under-sized intermediate buffers. Would have prevented the 2 hangs in stage 2 (8 minutes saved).
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Add Numerical Debugging Escalation Rules for Kernel Writer

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd prompt
- **Change**: Add debugging rules: (1) "If two different code changes produce the identical error magnitude (same max diff), the bug is in a different dimension -- stop investigating that parameter." (2) "If adding DPRINT causes a test to pass, this is a synchronization bug, not a data bug. Focus on CB wait/pop policies." (3) "After 3 hard failures on the same stage with the same error, emit an upstream_feedback breadcrumb and request design clarification rather than continuing to modify code."
- **Expected Benefit**: Prevents the broadcast-mode oscillation pattern and DPRINT wild-goose-chase observed in gamma debugging. Could save 30-60 minutes per similar incident.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 4: Remove Deliberation Text from Design Documents

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt
- **Change**: Add rule: "The design document must be decisional, not deliberative. Do not include phrases like 'Alternatively...', 'For simplicity...', or 'The kernel writer should decide...'. Choose one approach, document it precisely, and explain why it was chosen. If you considered alternatives, document them in a separate 'Alternatives Considered' section clearly marked as rejected."
- **Expected Benefit**: Eliminates ambiguity in handoff to kernel writer. The kernel writer should not need to resolve design decisions.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Cooperative Device Scheduling

- **Type**: tool_improvement
- **Target**: `scripts/tt-test.sh` or infrastructure scheduler
- **Change**: Modify device lock acquisition to release the lock between test cases rather than holding it for the entire test suite. Alternatively, implement a scheduling queue where agents register their expected test duration and are scheduled to minimize contention.
- **Expected Benefit**: Reduces device contention from 60+ minute blocks to seconds. Enables faster TDD iteration.
- **Priority**: HIGH
- **Effort**: LARGE

### Recommendation 6: Add Broadcast Mode Hardware Semantics to Design Template

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt / Binary Op Broadcast Verification table template
- **Change**: Add a reference table to the architect's prompt documenting what each broadcast mode does at the LLK level: "ROW: unpacks row 0 of each face of B tile, broadcasts across all rows during mul. B tile row 0 must contain valid data. COL: unpacks col 0, broadcasts across all cols. SCALAR: unpacks element [0,0], broadcasts everywhere. NONE: full element-wise, both tiles must have identical valid regions."
- **Expected Benefit**: Prevents broadcast mode confusion. The kernel writer would not have wasted time switching between ROW and NONE if the semantics were documented.
- **Priority**: MEDIUM
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Correctly identified hybrid mode with 3 appropriate references |
| Analysis quality | 5/5 | All 3 analyses were thorough, well-structured, and directly useful |
| Design completeness | 3/5 | Good for stages 1-3, but gamma stage had unresolved alternatives and incorrect CB sizing |
| Build correctness | 4/5 | One compile fix needed, but proactively fixed 5 test template issues |
| Kernel implementation | 2/5 | 3/4 stages passed smoothly, but gamma stage was an 87-minute unresolved debugging spiral |
| Inter-agent communication | 3/5 | Good for phases 0-3, but design-to-kernel-writer handoff for gamma was poor |
| Logging/observability | 3/5 | Breadcrumbs were detailed and timestamped for kernel writer and builder; missing execution_log.md files; no execution logs for analyzers or architect |

### Top 3 Things to Fix

1. **CB reuse across phases**: The architect must not reuse CBs across phases with different page requirements. This single design decision caused 87 minutes of unresolved debugging (59% of total pipeline time). Assigning a dedicated CB would have cost nothing.

2. **Numerical debugging escalation**: The kernel writer needs explicit rules for when to stop investigating a parameter (identical error magnitude = wrong dimension) and when to escalate to design review (3+ failures with same error). The current approach of unlimited hypothesis cycling within a single session burns context without convergence.

3. **Sequential-helper CB sizing**: The design template must distinguish between concurrent streaming (reader/compute, 2 pages OK) and sequential helpers (same kernel, needs full buffer). This rule would prevent the class of CB deadlocks seen in stage 2 and potentially in the gamma stage.

### What Worked Best

The reference analysis phase (Phase 1) was the single strongest aspect of this pipeline run. The three parallel analyzers produced comprehensive, role-specific analyses of tilize, reduce_w, and untilize that directly enabled the architect to make correct design decisions for 75% of the operation (stages 1-3). The reduce_w analysis in particular correctly identified `prepare_reduce_scaler` as the recommended approach (avoiding the matmul path) and documented the `post_reduce_op` lambda pattern for rsqrt -- both of which were used exactly as analyzed in the final kernel. This phase took only 13 minutes and set up the rest of the pipeline for success on the non-gamma stages.
