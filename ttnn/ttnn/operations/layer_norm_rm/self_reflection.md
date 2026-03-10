# Self-Reflection: layer_norm_rm (v2 run2)

## Metadata
| Field | Value |
|-------|-------|
| Operation | `layer_norm_rm` |
| Operation Path | `ttnn/ttnn/operations/layer_norm_rm` |
| Pipeline Phases Executed | Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd (x2 sessions) |
| Total Git Commits | 13 (this run, 2026-03-10 11:52 - 13:45) |
| Total Pipeline Duration | ~113 minutes |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~0m | PASS | Not explicitly logged; analyzer targets were pre-determined (tilize, untilize, batch_norm) |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~12m | PASS | 3 parallel analyzers for tilize (input_stage), untilize (output_stage), batch_norm (compute_core). All produced comprehensive analyses. |
| 2: Design | ttnn-operation-architect | ~11m | PASS | Hybrid mode design, 11+4 CBs, 10 compute phases, 5 TDD stages. Clean single-pass design with no revisions. |
| 3: Build | ttnn-generic-op-builder | ~10m | PASS | 3 attempts (2 quick fixes: CoreRange API, kernel include paths). All 7 integration tests passed. |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~74m | PASS | 5 stages. Stages 1,2,4,5 clean first-attempt passes. Stage 3 (square_centered) consumed ~57m with 8 hard attempts across 2 writer sessions before tolerance fix. |
| 5: Report | orchestrator | ~0m | N/A | No REPORT.md found for this run |

### Agent Duration Breakdown

Duration calculation method: Breadcrumb `"event":"start"` and `"event":"complete"` timestamps, cross-referenced with git commit timestamps.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 2026-03-10T11:52:39 | 2026-03-10T12:02:05 | ~9m | 0 | ~9m active |
| ttnn-operation-analyzer (untilize) | 2026-03-10T11:52:31 | 2026-03-10T12:04:27 | ~12m | 0 | ~12m active |
| ttnn-operation-analyzer (batch_norm) | 2026-03-10T11:53:15 | 2026-03-10T12:03:15 | ~10m | 0 | ~10m active |
| ttnn-operation-architect | 2026-03-10T12:05:50 | 2026-03-10T12:16:14 | ~10m | 0 | ~10m active |
| ttnn-generic-op-builder | 2026-03-10T12:19:16 | 2026-03-10T12:29:00 | ~10m | 2 (free) | ~7m active, ~3m fixing CoreRange + includes |
| ttnn-kernel-writer-tdd (session 1) | 2026-03-10T12:31:02 | ~2026-03-10T12:47:00 | ~16m | 6 hard (budget exhausted on square_centered) | ~5m active (stages 1-2), ~11m debugging square_centered |
| ttnn-kernel-writer-tdd (session 2) | 2026-03-10T12:47:07 | 2026-03-10T13:45:54 | ~59m | 2 hard (square_centered before fix) | ~17m active (stages 3-5 post-fix), ~42m debugging square_centered |

### Duration Visualization

```
Phase 1  |============|                                           (~12m) 3 analyzers in parallel
Phase 2              |==========|                                 (~11m)
Phase 3                         |=========|                       (~10m)
Phase 4                                    |==============================...==========| (~74m) <- longest
             data_pipeline(3m) subtract_mean(2m) square_centered(57m!!) full_norm(5m) gamma(7m)
         0    10   20   30   40   50   60   70   80   90  100  110 min

Longest phase: Phase 4 (74m) -- stage 3 (square_centered) alone consumed 57m
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 1) | ~12m | 11% | 3 analyzers in parallel |
| Design (Phase 2) | ~11m | 10% | Single pass, no revisions |
| Build (Phase 3) | ~10m | 9% | 2 quick free-retry fixes |
| Kernel implementation (Phase 4) | ~74m | 65% | 5 TDD stages |
| -- Productive coding | ~17m | 15% | Stages 1,2,4,5 + final stage 3 implementation |
| -- Debugging/retries (square_centered) | ~57m | 50% | 8 hard attempts on bf16 precision issue |
| Reporting (Phase 5) | ~0m | 0% | No report generated |
| Inter-phase gaps | ~6m | 5% | Agent startup/handoff overhead |
| **Total** | **~113m** | **100%** | |

---

## 2. What Went Well

### 1. Clean First-Attempt Passes on 4 of 5 TDD Stages

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: Stages data_pipeline, subtract_mean, full_normalize, and gamma_beta all passed on first attempt with 0 free retries and 0 hard attempts (confirmed in `.tdd_state.json` failure_history being empty for these stages). Total time for these 4 stages combined was ~17 minutes for 10 compute phases, 3 kernel files, and 15 CBs.
**Why it worked**: The architect's design document was exceptionally detailed. Every compute phase had explicit helper call signatures with CB IDs, policies (WaitUpfrontNoPop, NoWaitPopAtEnd, etc.), and CB state tables after each phase. The kernel writer could directly translate the pseudocode into working C++ without interpretation ambiguity.

### 2. High-Quality Reference Analyses

**Phase/Agent**: Phase 1, ttnn-operation-analyzer
**Evidence**: The architect's breadcrumbs show rapid reference extraction: reading all 3 analyses and extracting key findings (tilize: RM stick pattern, untilize: untilize helper, batch_norm: rsqrt pipeline) took ~2 minutes (12:05:50 to 12:06:14). No upstream feedback was filed about analysis quality.
**Why it worked**: Each analysis was focused on a specific role (input_stage, output_stage, compute_core) and included concrete API signatures, CB sizing formulas, and explicit "for layer_norm_rm" reuse guidance. The batch_norm analysis correctly identified the key difference (pre-computed vs in-kernel mean/var) and documented reduce API patterns even though batch_norm does not use them.

### 3. Architect Helper Validation Prevented CB Mismatches

**Phase/Agent**: Phase 2, ttnn-operation-architect
**Evidence**: The architect's execution log shows explicit helper analysis for 6 helper files (tilize_helpers.hpp, untilize_helpers.hpp, reduce_helpers_compute.hpp, binary_op_helpers.hpp, dest_helpers.hpp, reduce_helpers_dataflow.hpp). The "Architecture Revisions (Pass 2 corrections)" section states "None required - Pass 1 CB layout was compatible with all helpers." Throughout all 5 TDD stages, there were zero CB synchronization bugs -- every push/pop/wait was correctly balanced.
**Why it worked**: The architect validated every helper's requirements against the CB layout before committing the design. The Binary Op Broadcast Verification table in op_design.md explicitly mapped each phase's CB valid regions against broadcast dimensions.

### 4. Builder Produced Correct Infrastructure with Minimal Retries

**Phase/Agent**: Phase 3, ttnn-generic-op-builder
**Evidence**: 2 quick fixes (CoreRange API attribute name, kernel include path) resolved in seconds. All 15 CBs matched the architect's design exactly. 7/7 integration tests passed. The kernel writer never filed upstream feedback about program descriptor issues (except adding gamma/beta TensorAccessorArgs in stage 5, which was an expected extension).
**Why it worked**: The architect's design document Part 1 provided explicit CB tables with page sizes, page counts, and data formats. The builder had a clear specification to implement against.

### 5. Final Kernel Code is Clean and Well-Structured

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd
**Evidence**: The final compute kernel (`layer_norm_rm_compute.cpp`) is 153 lines with 10 clearly labeled phases, compile-time constexpr CB routing, and no dead code or commented-out debug logic. The reader and writer kernels are similarly clean (107 and 46 lines respectively). No residual artifacts from the square_centered debugging remain in the final code.
**Why it worked**: Despite the extensive debugging in stage 3, the kernel writer cleanly reverted all experimental changes (matmul approach, fp32 DEST) before the final fix, leaving only the design-specified reduce helper approach.

---

## 3. Issues Found

### Issue 1: Architect Set Tolerance Too Tight for Intermediate Stage (square_centered)

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- square_centered (Stage 3) |
| Agent | ttnn-operation-architect (root cause), ttnn-kernel-writer-tdd (impact) |
| Retries Consumed | 8 hard attempts across 2 kernel-writer sessions (6 in session 1 exhausting budget, 2 more in session 2) |
| Time Cost | ~57 minutes (~50% of total pipeline time) |

**Problem**: The architect set the square_centered stage tolerance to `rtol=0.02, atol=0.1`. The kernel writer's implementation was mathematically correct, producing `max_diff=0.375` consistently across all approaches (reduce helper, matmul_tiles, separate SUM+multiply). The issue was that bf16 reduce accumulation introduces ~0.06 mean error, and squaring amplifies this by factor `|a+b| ~ 7`, giving theoretical max error of ~0.42 -- which exceeds `atol=0.1`.

The kernel writer explored 12 hypotheses over ~57 minutes:
- H1: Enable fp32_dest_acc_en (failed: zeroed rows from pack_untilize incompatibility)
- H2: AVG PoolType (not attempted -- compile-time template limitation)
- H3: fp32 + WaitUpfront untilize mode (failed: still zeroed rows)
- H4: Separate SUM then multiply by 1/W (same 0.375 error)
- H5: bf16 accumulation in reduce is inherently imprecise (correct diagnosis, but tried wrong fix)
- H6: Replace reduce with matmul_tiles (same 0.375 error, plus introduced state corruption)
- H6 revised: matmul causes mm_init state corruption (incorrect -- same error proves otherwise)
- H7: Re-enable fp32_dest_acc_en after removing matmul (zeroed rows again)
- H8-H9: Standard untilize_block instead of pack_untilize (still zeroed rows with fp32)
- H10: Mathematical analysis of error amplification (correct root cause)
- H11: Post-reduce scaler multiplication (same error)
- H12: Tolerance is a design error (final correct fix)

The fix was trivial: change tolerance from `atol=0.1` to `atol=0.5` in the test file and `.tdd_state.json`. This took seconds to implement but ~57 minutes to reach.

**Root Cause**: The architect did not account for error amplification through the squaring operation when setting intermediate stage tolerances. The stage 2 tolerance (`atol=0.1`) was borderline for bf16 reduce precision, and squaring mathematically amplifies this error beyond any achievable atol with bf16 DEST accumulation.

**Fix for agents**:
- **ttnn-operation-architect**: Add a mandatory tolerance analysis step for intermediate stages that involve squaring, division, or exponentiation. For any stage where the reference includes `pow(2)` or similar nonlinear operations on bf16 intermediate results, the tolerance must account for error amplification. Concrete rule: if stage N tolerance allows max error `e`, and stage N+1 squares the output, stage N+1 tolerance must be at least `2 * max_value * e` where `max_value` is the expected magnitude of the intermediate.
- **ttnn-kernel-writer-tdd**: When a consistent numerical mismatch persists across 3+ fundamentally different approaches (e.g., reduce helper, matmul, separate SUM+multiply all give identical max_diff), the agent should immediately suspect a tolerance specification error rather than continuing to modify kernel code. Add a heuristic: "If the same max_diff value appears across 3+ structurally different approaches, file upstream feedback about tolerance and proceed with a tolerance adjustment rather than consuming remaining hard attempts."

### Issue 2: Kernel Writer Exhausted Hard Attempt Budget Requiring Session Restart

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- square_centered (Stage 3) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 6/6 hard attempts in session 1 (budget_exhausted=true in .tdd_state.json) |
| Time Cost | ~16 minutes wasted in session 1 before budget reset in session 2 |

**Problem**: The `.tdd_state.json` shows 8 entries in `failure_history` for square_centered. The first 6 entries (attempts 1-6 in session 1) all have `cost: "HARD"` and show the budget depleting from `remaining_attempts: 5` down to `remaining_attempts: 0` with `budget_exhausted: true`. Then 2 more entries appear (attempts 1-2 in session 2) -- the budget was apparently reset for a new kernel-writer session.

The kernel writer in session 1 spent 6 hard attempts on approaches that all yielded the same `max_diff=0.375` or `9.0` (fp32 variant). None of the 6 approaches changed the fundamental precision limitation of bf16 DEST accumulation.

**Root Cause**: The hard attempt budget (6) was consumed by a problem that required a specification change, not a code change. The kernel writer has no mechanism to "escalate" a tolerance issue back to the architect without consuming hard attempts.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Add an "upstream escalation" path that does not consume hard attempts. When the agent identifies a consistent numerical mismatch that persists across 2+ structurally different approaches, it should be able to flag the tolerance as potentially incorrect and either (a) adjust the tolerance with documented justification, or (b) escalate without consuming the hard attempt budget.
- **Pipeline orchestrator**: Consider adding a "tolerance review" checkpoint that triggers when more than 3 hard attempts are consumed on a numerical mismatch with a consistent max_diff value.

### Issue 3: fp32_dest_acc_en Incompatibility with pack_untilize Not Documented in Design

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- square_centered (Stage 3) |
| Agent | ttnn-operation-architect (missing constraint), ttnn-kernel-writer-tdd (wasted attempts) |
| Retries Consumed | 4 hard attempts on fp32 DEST approaches (H1, H3, H7-H9) |
| Time Cost | ~15 minutes on fp32-related approaches that could not work |

**Problem**: The kernel writer attempted to enable `fp32_dest_acc_en=True` to improve reduce precision (hypotheses H1, H3, H7, H8, H9). Each time, the output contained zeroed rows from pack_untilize incompatibility with fp32 DEST. This hardware limitation was not documented in the architect's design, the reference analyses, or the hardware constraints checklist in op_design.md.

From breadcrumbs: H1 at 12:38:07 produced `max_diff=9.0` with zeroed rows. H3 at 12:41:07 tried WaitUpfront mode, same result. H7 at 13:20:16 re-enabled fp32 after eliminating matmul, same zeroed rows. H9 at 13:22:43 tried standard untilize_block instead of pack_untilize, still zeroed rows.

**Root Cause**: The architect's Hardware Constraints Checklist includes `[x] DEST register holds max 8 tiles (bf16 half-sync) -- Wt can exceed this; helpers auto-chunk` but does not mention that `fp32_dest_acc_en=True` is incompatible with pack_untilize on Wormhole B0. This is a known hardware limitation that should be documented in reference analyses or the design.

**Fix for agents**:
- **ttnn-operation-architect**: Add to the Hardware Constraints Checklist: "fp32_dest_acc_en=True is incompatible with pack_untilize (produces zeroed rows for block_width > DEST_LIMIT/2). If fp32 accumulation is needed for precision, either avoid untilize or use manual tile-by-tile pack." This constraint should be checked whenever the design includes both reduce operations and untilize.
- **ttnn-operation-analyzer**: When analyzing untilize reference operations, document the fp32 DEST limitation explicitly in the output stage analysis.

### Issue 4: Builder Used Incorrect CoreRange Attribute Names and Include Paths

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder |
| Retries Consumed | 2 free retries |
| Time Cost | ~3 minutes |

**Problem**: The builder initially used `.start_coord`/`.end_coord` instead of `.start`/`.end` on CoreRange objects (breadcrumb H1), and used host-side include path `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` instead of device-side `api/tensor/tensor_accessor.h` (breadcrumb H2).

**Root Cause**: The builder's instructions do not document CoreRange API attribute names, and the include mapping table references host-side paths that are invalid for device kernel compilation. The builder's execution log Recommendation 1 and Recommendation 2 both flag these issues.

**Fix for agents**:
- **ttnn-generic-op-builder instructions**: Add note: "CoreRange attributes are `.start` and `.end` (NOT `.start_coord`/`.end_coord`)." Add device-side include mapping: `TensorAccessor -> #include "api/tensor/tensor_accessor.h"`.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~3m | 0 free, 0 hard | PASS | Clean -- all 3 kernels implemented and passed first attempt |
| subtract_mean | ~2m | 0 free, 0 hard | PASS | Clean -- 2 compute phases added, passed immediately |
| square_centered | ~57m | 0 free, 8 hard (6+2 across sessions) | PASS | bf16 precision amplified through squaring; tolerance spec error |
| full_normalize | ~5m | 0 free, 0 hard | PASS | Clean -- 3 new phases (reduce_var, add_eps+rsqrt, mul_rsqrt) |
| gamma_beta | ~7m | 0 free, 0 hard | PASS | Clean -- reader changes + affine transform + CB routing |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | square_centered precision debugging | ttnn-kernel-writer-tdd | ~57m | 50% | Consistent 0.375 max_diff across all approaches; tolerance too tight | 8 hard | Architect did not account for bf16 error amplification through squaring |
| 2 | fp32_dest_acc_en exploration | ttnn-kernel-writer-tdd | ~15m | 13% | 4 attempts to use fp32 DEST, all failed due to pack_untilize incompatibility | 4 hard | Undocumented hardware limitation |
| 3 | matmul_tiles approach | ttnn-kernel-writer-tdd | ~18m | 16% | Replaced reduce with matmul; identical result but with state corruption | 2 hard | Red herring: matmul does not improve bf16 accumulation precision |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | Implemented matmul_tiles row reduction with ones-column CB (12:52-12:55) | matmul uses same bf16 DEST as reduce_tile; introduced mm_init state corruption that masked the real issue | Document in kernel-writer instructions that matmul_tiles does NOT improve accumulation precision over reduce_tile for bf16 |
| ttnn-kernel-writer-tdd | 4 attempts with fp32_dest_acc_en (12:38-12:41, 13:20-13:24) | fp32 DEST is fundamentally incompatible with pack_untilize on this hardware | Document this hardware constraint in the design/analysis phase |
| ttnn-kernel-writer-tdd | Session 1 entirely (12:31-12:47, 6 hard attempts) | Budget exhausted without resolving the fundamental tolerance issue; same work repeated in session 2 | Allow tolerance escalation without consuming hard budget |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer --> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | tilize_analysis.md, untilize_analysis.md, batch_norm_analysis.md |
| Quality | GOOD |
| Issues | batch_norm analysis does not mention fp32_dest_acc_en incompatibility with pack_untilize. The analysis documents that batch_norm uses SFPU kernel when fp32 is enabled (which avoids pack_untilize) but does not explicitly flag this as a constraint. |
| Downstream Impact | Architect did not include fp32/pack_untilize constraint in design. Kernel writer spent ~15 minutes on fp32 approaches that could never work. |
| Suggestion | When analyzing operations that use both fp32_dest_acc_en and untilize, explicitly document whether the operation uses pack_untilize vs standard untilize_block, and flag any incompatibilities. |

### Handoff 2: ttnn-operation-architect --> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | GOOD |
| Issues | Include path for TensorAccessor referenced host-side path (`ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp`) instead of device-side (`api/tensor/tensor_accessor.h`). Builder had to fix this. |
| Downstream Impact | 1 free retry (~1 minute) |
| Suggestion | Architect should use device-side include paths in kernel implementation sections. |

### Handoff 3: ttnn-operation-architect --> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 (Kernel Implementation) |
| Quality | GOOD (4/5 stages) to POOR (stage 3 tolerance) |
| Issues | Stage 3 tolerance (rtol=0.02, atol=0.1) was too tight for bf16 precision after squaring. The design document does not analyze error propagation through nonlinear operations. |
| Downstream Impact | 57 minutes and 8 hard attempts debugging a specification error |
| Suggestion | Add mandatory error budget analysis for intermediate stages, especially those involving squaring, division, or exponentiation of bf16 intermediate results. |

### Handoff 4: ttnn-generic-op-builder --> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Program descriptor, stub kernels, test files |
| Quality | GOOD |
| Issues | None significant. The builder's handoff notes correctly flagged device-side TensorAccessor include path and kernel_lib prefix. The kernel writer only needed to add gamma/beta TensorAccessorArgs in stage 5 (expected extension). |
| Downstream Impact | Minimal -- gamma/beta TensorAccessorArgs addition was clean (~2 minutes) |
| Suggestion | None needed for this handoff. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-kernel-writer-tdd (inferred) | Add tolerance error propagation analysis for intermediate stages with nonlinear operations | H | H |
| ttnn-generic-op-builder | ttnn-generic-op-builder (Rec 1) | Document CoreRange uses `.start`/`.end` not `.start_coord`/`.end_coord` | H | M |
| ttnn-generic-op-builder | ttnn-generic-op-builder (Rec 2) | Fix TensorAccessor include mapping to use device-side path `api/tensor/tensor_accessor.h` | H | M |
| ttnn-operation-architect | ttnn-generic-op-builder (upstream feedback) | Design should use device-side include paths in kernel implementation sections | M | M |
| ttnn-operation-architect | ttnn-operation-architect (Rec 1) | tdd_orchestrator.py test template needs `return` before reference_body | H | L |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD tolerance specification | Architect set tolerance without error propagation analysis; 50% of pipeline time was wasted debugging a tolerance spec error | Add mandatory error budget step to architect for intermediate stages | H |
| Hard attempt budget | 6 hard attempts were consumed on a problem that needed a spec change, not a code change; budget exhaustion forced session restart | Add "upstream escalation" path that does not consume hard attempts | H |
| Hardware constraints documentation | fp32_dest_acc_en + pack_untilize incompatibility was undocumented, costing 4 hard attempts | Document in analysis/design phase; add to hardware constraints checklist | M |
| Numerical debugging strategy | Kernel writer tried 12 hypotheses in sequence; did not recognize the "same max_diff across different approaches" pattern until hypothesis H12 | Add debugging heuristic: if same max_diff persists across 3+ approaches, suspect tolerance spec error | M |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | YES | The canonical example: 57 minutes on square_centered stage, 8 hard attempts, 12 hypotheses. Context burned on approaches that could not work (fp32 DEST, matmul). |
| 3 | `.tdd_state.json` coupling is fragile | PARTIAL | Budget was exhausted in session 1, then reset in session 2. The `.tdd_state.json` shows 8 failure_history entries with attempt counters resetting (1-6 then 1-2), suggesting the budget was manually or automatically reset between sessions. |
| 6 | Builder runs on Sonnet while everything else uses Opus | PARTIAL | Builder had 2 free retries (CoreRange API, include path) that Opus might have avoided. However, 2 free retries at ~1 min each is low cost. |
| 9 | No validation between architect output and builder output | NO | CB layout matched perfectly in this run. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Architect tolerance specification lacks error propagation analysis | The architect sets tolerances for intermediate TDD stages without analyzing how bf16 rounding errors propagate through nonlinear operations (squaring, division). This caused 50% of pipeline time to be spent debugging a correct implementation against an infeasible tolerance. | H |
| fp32_dest_acc_en + pack_untilize incompatibility undocumented | The hardware limitation where fp32_dest_acc_en=True causes pack_untilize to produce zeroed rows is not documented in any reference analysis, design document, or hardware constraints checklist. Kernel writer wasted 4 hard attempts on approaches that could never work. | M |
| Kernel writer lacks "same max_diff" pattern recognition heuristic | When a consistent numerical mismatch value (e.g., 0.375) persists across structurally different kernel implementations (reduce helper, matmul, separate SUM+multiply), this strongly indicates a tolerance/spec issue rather than a kernel bug. The kernel writer has no explicit heuristic for this pattern. | M |

---

## 8. Actionable Recommendations

### Recommendation 1: Add Error Propagation Analysis to Architect for Intermediate Tolerances

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent instructions
- **Change**: Add a mandatory "Tolerance Analysis" section to the TDD Stage Plan. For each intermediate stage, the architect must: (1) compute the expected bf16 rounding error for the stage's operations, (2) analyze how previous stage errors propagate through the current stage's operations (especially squaring, division, exponentiation), (3) set tolerance to at least 2x the computed maximum error. For squaring specifically: if stage N has tolerance `e` and max magnitude `M`, stage N+1 (squaring) must have tolerance >= `2 * M * e`.
- **Expected Benefit**: Eliminates the #1 time sink in this pipeline run (57 minutes, 50% of total time)
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add "Upstream Escalation" Path for Kernel Writer

- **Type**: pipeline_change
- **Target**: ttnn-kernel-writer-tdd agent instructions and pipeline orchestrator
- **Change**: When the kernel writer identifies a consistent numerical mismatch that persists across 2+ structurally different approaches (same max_diff value), allow it to adjust the tolerance with documented mathematical justification WITHOUT consuming hard attempts. The justification must include: (a) the consistent max_diff value, (b) the different approaches tried, (c) the mathematical analysis of why the tolerance is infeasible. This should be logged as an "upstream_fix" in breadcrumbs.
- **Expected Benefit**: Prevents budget exhaustion on spec errors; reduces time spent on infeasible debugging
- **Priority**: HIGH
- **Effort**: MEDIUM

### Recommendation 3: Document fp32_dest_acc_en + pack_untilize Incompatibility

- **Type**: instruction_change
- **Target**: ttnn-operation-architect hardware constraints checklist; ttnn-operation-analyzer instructions for untilize analysis
- **Change**: Add to the architect's Hardware Constraints Checklist: "fp32_dest_acc_en=True is incompatible with pack_untilize on Wormhole B0 (produces zeroed rows for block_width > DEST_LIMIT/2). If fp32 accumulation is needed, operations must avoid the pack_untilize helper or use manual tile-by-tile pack/write." Analyzer should flag this when documenting untilize patterns.
- **Expected Benefit**: Prevents kernel writer from spending attempts on approaches known to fail
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Add "Consistent Max Diff" Debugging Heuristic to Kernel Writer

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd agent instructions, numerical debugging section
- **Change**: Add heuristic: "If the same max_diff value (within 10%) appears across 3+ structurally different kernel implementations, the issue is almost certainly NOT in the kernel code. Likely causes: (1) tolerance is too tight for bf16 precision, (2) the reference computation uses higher precision (PyTorch fp32 internal accumulation). Action: perform mathematical error analysis and adjust tolerance if justified, rather than continuing to modify kernel code."
- **Expected Benefit**: Reduces mean time to correct diagnosis for precision-related failures
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Fix Builder CoreRange and Include Path Documentation

- **Type**: instruction_change
- **Target**: ttnn-generic-op-builder agent instructions
- **Change**: (1) Add note: "CoreRange attributes: `.start` and `.end` (NOT `.start_coord`/`.end_coord`)." (2) Update include mapping: `TensorAccessor -> #include "api/tensor/tensor_accessor.h"` for device kernels.
- **Expected Benefit**: Eliminates 2 common free retries per pipeline run
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

Rate each dimension (1-5):

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4 | Correct references selected (tilize, untilize, batch_norm). batch_norm analysis missed fp32/pack_untilize constraint. |
| Analysis quality | 4 | Comprehensive analyses with reuse guidance. Missing hardware limitation documentation. |
| Design completeness | 3 | Excellent CB layout and compute phase design. Failed on tolerance specification for intermediate stages -- the single most impactful gap. |
| Build correctness | 4 | 2 quick free retries, all infrastructure correct. Minor API documentation gaps. |
| Kernel implementation | 4 | 4/5 stages clean first-attempt passes. Final kernel code is well-structured and correct. Only dragged down by the tolerance issue (not a kernel quality issue). |
| Inter-agent communication | 3 | Design-to-kernel handoff was excellent for CB layout and compute phases but failed on tolerance specification. Analysis-to-design handoff missed hardware constraints. |
| Logging/observability | 4 | Breadcrumbs were detailed with timestamps, hypothesis tracking, test results, and CB sync checks. Execution logs were complete. Gap: no REPORT.md for this specific run. The 2-session kernel writer split was visible in breadcrumbs (new "start" event at 12:47:07). |

### Top 3 Things to Fix

1. **Add error propagation analysis to architect tolerance specification** -- would have saved 57 minutes (50% of pipeline time) and 8 hard attempts
2. **Add upstream escalation path for kernel writer** -- would have prevented budget exhaustion and session restart on a spec error
3. **Document fp32_dest_acc_en + pack_untilize incompatibility** -- would have saved 4 hard attempts and ~15 minutes of exploration on approaches known to fail

### What Worked Best

The architect's design document quality for CB layout and compute phase specification was the single strongest aspect. All 15 CBs were correctly sized, every helper was validated against the CB layout, and the detailed pseudocode with explicit policies (WaitUpfrontNoPop, NoWaitPopAtEnd, etc.) enabled the kernel writer to achieve 4/5 first-attempt passes. The kernel writer was able to implement 10 compute phases across 3 kernel files with zero CB synchronization bugs -- a testament to the thoroughness of the architect's design validation.
