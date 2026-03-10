# Self-Reflection: group_norm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `group_norm` |
| Operation Path | `ttnn/ttnn/operations/group_norm` |
| Pipeline Phases Executed | Phase 0 (Discovery), Phase 1 (Analysis), Phase 2 (Design), Phase 3 (Build), Phase 4 (TDD Kernels), Phase 5 (Report) |
| Agents Invoked | 3x ttnn-operation-analyzer, ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 10 (March 10 run) |
| Total Pipeline Duration | ~97 minutes (11:50 - 13:29 UTC) |
| Overall Result | SUCCESS -- All 4 TDD stages passed |

**Note**: Git history shows an earlier pipeline run (Feb 20-23) using the old planner/designer pipeline. The March 10 run analyzed here uses the new merged architect pipeline. The earlier run's artifacts were overwritten.

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m | PASS | 3 references selected: tilize, untilize, batch_norm |
| 1: Analysis | 3x ttnn-operation-analyzer | ~11m | PASS | 3 parallel analyzers, thorough analysis docs produced |
| 2: Design | ttnn-operation-architect | ~14m | PASS | Single-pass design, no errors, correct mode detection (Hybrid) |
| 3: Build | ttnn-generic-op-builder | ~10m | PASS | 1 compilation error (include paths), recovered cleanly |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~60m | PASS | 4 stages, 2 hard attempts, 2 free retries |
| 5: Report | orchestrator | ~2m | PASS | Report generated from pipeline data |

### Agent Duration Breakdown

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 11:50:42 | 12:01:30 | ~11m | 0 | ~11m active |
| ttnn-operation-analyzer (untilize) | 11:50:46 | 12:00:42 | ~10m | 0 | ~10m active |
| ttnn-operation-analyzer (batch_norm) | 11:50:52 | 12:00:51 | ~10m | 0 | ~10m active |
| ttnn-operation-architect | 12:02:45 | 12:16:07 | ~13m | 0 | ~13m active |
| ttnn-generic-op-builder | 12:18:46 | 12:26:59 | ~8m | 1 free | ~7m active, ~1m fixing includes |
| ttnn-kernel-writer-tdd | 12:28:02 | 13:27:02 | ~59m | 2 hard + 2 free | ~40m coding, ~19m debugging |

**Duration calculation method**: Used breadcrumb `"event":"start"` and `"event":"complete"` timestamps from all 4 breadcrumb files. The kernel-writer breadcrumbs show a session boundary at 13:04:52 (a second `start` event indicating a context/session restart mid-normalize stage).

### Duration Visualization

```
Phase 0-1  |||||||||||||                                       (~12m) 3 analyzers parallel
Phase 2         |||||||||||||                                  (~14m)
Phase 3                    ||||||||||                          (~10m)
Phase 4                              |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| (~60m)
Phase 5                                                                                      || (~2m)
           0    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80  min

Longest phase: Phase 4 (60m) -- 4 TDD stages including REDUCE_SCALAR scaler bug and test template bug
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~12m | 12% | 3 analyzers in parallel |
| Design (Phase 2) | ~14m | 14% | Single-pass, no revisions |
| Build (Phase 3) | ~10m | 10% | 1 include path fix |
| Kernel implementation (Phase 4) | ~60m | 62% | 4 TDD stages |
| -- Productive coding | ~40m | 41% | Writing kernel code that passed |
| -- Debugging/retries | ~19m | 20% | 2 hard attempts + 2 free retries |
| Reporting (Phase 5) | ~2m | 2% | |
| **Total** | **~97m** | **100%** | |

---

## 2. What Went Well

### 1. Group mean subtract stage passed on first test run (0 hard attempts)

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd, stage group_mean_subtract
**Evidence**: `.tdd_state.json` shows `"attempts": 0, "free_retries": 0` for this stage. The breadcrumb at 12:58:39 shows the stage passed with all 4 shapes on the first real test run (the one earlier failure at 12:57:36 was a free compilation error for `reduce_init_short` which was quickly fixed in <30 seconds).
**Why it worked**: The kernel writer correctly devised the binary group_scaler mask approach for handling sub-tile group boundaries, a non-trivial innovation that went beyond the architect's design. The writer also correctly used `cb_sq_sum` as an intermediate accumulator instead of `cb_mean` to avoid FIFO ordering issues -- showing strong understanding of CB semantics.

### 2. Analysis phase produced thorough, well-structured reference documents

**Phase/Agent**: Phase 1, 3x ttnn-operation-analyzer
**Evidence**: Three analysis files totaling ~600 lines: `tilize_single_core_analysis.md` (364 lines), `untilize_single_core_analysis.md` (425 lines), `batch_norm_analysis.md` (641 lines). Each contains complete CB tables, exact API signatures, memory access patterns, code snippets, and explicit "For Group Norm Reference" sections. The architect's execution log confirms: "All three analysis documents were thorough and provided the needed patterns."
**Why it worked**: The analyzers used DeepWiki queries effectively (11 queries total across 3 analyzers), read the actual helper library implementation files (tilize_helpers.inl, untilize_helpers.inl, dest_helpers.hpp), and focused on the specific role each reference plays (input_stage, output_stage, compute_core). The "downstream use" framing in the analysis docs was particularly effective.

### 3. Architect produced a single-pass design with no revisions

**Phase/Agent**: Phase 2, ttnn-operation-architect
**Evidence**: The architect breadcrumbs show a clean sequence: start -> 3 reference_reads -> mode_detection -> 3 design_decisions -> tdd_init -> git_commit -> complete. No hypothesis events, no recovery events. The execution log Section 3 confirms: "No errors encountered. Single-pass execution." The design document is 294 lines and covers all CB requirements, kernel arguments, phase details, and critical notes.
**Why it worked**: The merged architect agent (Improvement #2, DONE) eliminated the planner-designer handoff. The architect could validate helper compatibility immediately while designing the architecture, correctly identifying that the reduce helper cannot address column subsets.

### 4. Data pipeline stage passed cleanly (only 1 free retry)

**Phase/Agent**: Phase 4, ttnn-kernel-writer-tdd, stage data_pipeline
**Evidence**: The only failure was a trivial include path error (`compute_kernel_api/common.h` should be `api/compute/common.h`). This was classified as FREE, fixed in <1 minute, and the stage passed on retest with all 4 shapes. Total stage wall time: ~8 minutes.
**Why it worked**: The tilize/untilize helper patterns from the analysis docs mapped directly to the implementation. The writer/reader patterns were well-documented with exact TensorAccessor usage examples.

### 5. CB layout was mostly correct from the start (12 of 14 CBs needed zero changes)

**Phase/Agent**: Phase 2 (architect) -> Phase 3 (builder) -> Phase 4 (kernel-writer)
**Evidence**: Of the 14 CBs in the final implementation (CB 0-8, 16-17, 24-26), only 2 required changes during TDD: (1) cb_mean expanded from 1 to G pages, (2) cb_den expanded from 1 to G pages, and 2 new CBs were added (CB 8 = cb_scratch, CB 26 = cb_group_scaler). The original 12 CB IDs, page sizes, and data formats from op_design.md all survived unchanged.
**Why it worked**: The architect correctly identified all intermediate storage needs and lifetimes. The builder faithfully transcribed the CB table.

---

## 3. Issues Found

### Issue 1: REDUCE_SCALAR double-application of scaler (1/K vs 1/sqrt(K))

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- normalize (stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~8 minutes (13:09:41 numerical mismatch -> 13:17:12 fix committed) |

**Problem**: The `reduce_tile<SUM, REDUCE_SCALAR>` API applies the scaler at both row and column reduction stages, so the effective scaler is `scaler^2`. The architect's design (op_design.md line "Scaler = 1/K") and the kernel writer's initial implementation both used `1/K` as the scaler value. The result was that the mean was off by a factor of K, producing output values in the raw input range rather than the normalized range. The breadcrumb at 13:16:58 records: "DPRINT confirmed mean is K times too small."

**Root Cause**: This is an undocumented hardware behavior. Neither the architect nor the kernel writer knew that `REDUCE_SCALAR` applies the scaler twice. The architect's design note on line 83 ("Scaler CB c_5 is bf16") did not mention the squaring behavior. The batch_norm analysis document did not encounter this issue because batch_norm receives pre-computed mean/variance and does not use `reduce_tile` for stats computation.

**Fix for agents**:
- **ttnn-operation-architect**: Add to critical notes: "WARNING: `reduce_tile<SUM, REDUCE_SCALAR>` applies the scaler at BOTH row and column reduction stages. Effective scaler = scaler^2. If you need effective 1/K, pass 1/sqrt(K)."
- **ttnn-kernel-writer-tdd**: Add to debugging checklist: "If reduce output is wrong by a power-of-K factor, check whether REDUCE_SCALAR is applying the scaler twice."

### Issue 2: `add_tiles_bcast_scalar_init_short` does not exist (hallucinated API)

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- normalize (stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 free retry |
| Time Cost | ~1 minute |

**Problem**: The kernel writer used `add_tiles_bcast_scalar_init_short(...)` which does not exist in the API. The breadcrumb at 13:08:16 records the compilation error. The fix was to replace it with `init_bcast<ELWADD, SCALAR>(cb_tmp, cb_eps, cb_den)`.

**Root Cause**: The kernel writer hallucinated an API name by pattern-matching from other `_init_short` functions (e.g., `reduce_init_short`, `copy_tile_to_dst_init_short`). The actual broadcast init API uses the templated `init_bcast<BinaryType, BcastType>(...)` pattern.

**Fix for agents**:
- **ttnn-operation-architect**: In Part 2 kernel implementation notes, always specify the exact init function for broadcast operations: `init_bcast<ELWADD, SCALAR>(...)` not `add_tiles_bcast_scalar_init_short`.
- **ttnn-kernel-writer-tdd**: Add to known API patterns: "Broadcast operations use `init_bcast<BinaryType, BcastType>(cb_a, cb_b, cb_out)` and `op_tiles_bcast_scalar(cb_a, cb_b, i, j, dst)` -- there is no `_init_short` variant for broadcast."

### Issue 3: Auto-generated test_stage_affine.py had missing gamma/beta args

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- affine (stage 3) |
| Agent | ttnn-operation-architect (TDD template) + ttnn-kernel-writer-tdd |
| Retries Consumed | 1 hard attempt |
| Time Cost | ~3 minutes |

**Problem**: The auto-generated `test_stage_affine.py` created gamma and beta tensors in `extra_setup` but did not pass them to the `group_norm()` call. The breadcrumb at 13:23:33 records: "Test bug confirmed: test creates random gamma/beta for reference but does not pass them to group_norm call." The kernel writer had to manually fix the test file.

**Root Cause**: The TDD orchestrator's template system uses `extra_setup` (for variable creation) and `extra_args` (for function call arguments) as separate fields. The architect registered `extra_args` as `", num_groups=G, eps=1e-5"` but did not include `gamma=gamma, beta=beta`. The architect likely intended for gamma/beta to be optional defaults but the test reference body explicitly uses them.

**Fix for agents**:
- **ttnn-operation-architect**: When registering TDD stages with `extra_setup` that creates tensors used in the reference_body, MUST include those tensors in `extra_args`. Specifically: if `extra_setup` creates `gamma` and `beta` variables, and `reference_body` uses them, then `extra_args` must include `, gamma=gamma, beta=beta`.
- **tdd_orchestrator.py**: Consider adding validation: if `extra_setup` defines a variable name that appears in `reference_body`, warn if that variable is not present in `extra_args`.

### Issue 4: CB scratch conflict when cb_gamma became persistent

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- affine (stage 3) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 (caught proactively) |
| Time Cost | ~2 minutes |

**Problem**: During the normalize stage (stage 2), the kernel writer used `cb_gamma` (CB 2) as a scratch buffer for den_tile accumulation because gamma was not yet loaded. When the affine stage (stage 3) loaded gamma tiles into cb_gamma persistently, the scratch usage conflicted. The writer proactively added CB 8 (`cb_scratch`) and modified the program descriptor.

**Root Cause**: The architect's op_design.md did not anticipate this CB reuse conflict. The CB table listed cb_gamma as "Gamma tiles (TILE_LAYOUT), Ct pages, Program lifetime" but the kernel writer legitimately needed a scratch buffer in earlier stages. The architect also noted cb_gamma as the scratch candidate in the normalize stage implementation but did not account for the affine stage making it permanent.

**Fix for agents**:
- **ttnn-operation-architect**: When a CB is marked as "Program lifetime" (persistent), it should not be recommended as scratch space in any stage, even pre-affine stages. Add a dedicated scratch CB (e.g., CB 8) from the start if any stage needs scratch and a program-lifetime CB exists nearby.

### Issue 5: Compute kernel include paths wrong (compute_kernel_api/ vs api/compute/)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- data_pipeline (stage 0) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 free retry |
| Time Cost | ~1 minute |

**Problem**: The compute kernel initially used `#include "compute_kernel_api/common.h"` which does not exist. The correct path is `#include "api/compute/common.h"`. Breadcrumb at 12:33:07 records the compilation error.

**Root Cause**: The include path `compute_kernel_api/` is a common hallucination. The builder's execution log Recommendation 2 explicitly flags this: "Note that compute_kernel_api.h is the correct compute include." The kernel writer likely generated the path from general knowledge rather than consulting the builder's handoff notes.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Always use `#include "api/compute/*.h"` for compute kernel includes, never `compute_kernel_api/`. Add this to the agent's include path reference table.

### Issue 6: Session restart during normalize stage

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- normalize (stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 (no retries wasted, but ~5 min context re-loading) |
| Time Cost | ~5 minutes |

**Problem**: The kernel writer breadcrumbs show two `"event":"start"` entries: one at 12:28:02 and another at 13:04:52. This indicates a session restart (likely due to context limits). The second session had to re-parse the design and re-orient itself ("resuming_from_previous_session_stages_0_1_passed_stage_2_in_progress"). Between the first session's last breadcrumb (13:01:55) and the second session's start (13:04:52), there is a ~3 minute gap where no work was done.

**Root Cause**: The normalize stage involved heavy upstream modifications (program_descriptor changes, CB expansions, bf16 packing helpers) that consumed context window. This is a known issue (Pipeline Improvement #1: "Kernel writer burns massive context on numerical debugging").

**Fix for agents**:
- **Pipeline**: The kernel writer session handoff mechanism needs to preserve more state. The second session's breadcrumb only says "resuming" but there is no evidence it read the previous session's breadcrumbs to understand what was already attempted.

### Issue 7: Tolerance relaxation in affine stage (rtol=0.1, atol=0.7)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- affine (stage 3) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 (part of the test fix) |
| Time Cost | 0 (bundled with test fix) |

**Problem**: The affine stage required relaxing tolerance from the architect's specified `rtol=0.05, atol=0.2` to `rtol=0.1, atol=0.7`. The breadcrumb at 13:26:19 records this. The REPORT.md notes: "gamma multiplication amplifies bf16 normalization errors."

**Root Cause**: The original tolerance was set based on the normalize stage (without affine). When random gamma values (which can be >1.0) multiply already-imprecise bf16 normalized values, the absolute error grows proportionally. The architect did not account for error amplification through the affine transform.

**Fix for agents**:
- **ttnn-operation-architect**: When specifying tolerances for affine/scale stages, account for error amplification. If preceding stages have atol=X and the affine uses random gamma in [-2, 2], the affine stage needs atol >= 2*X. Or constrain test gamma values to be near 1.0 (e.g., `1 + 0.1 * randn`).

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~8m (12:28-12:35) | 1 free, 0 hard | PASS | Include path fix (trivial) |
| group_mean_subtract | ~24m (12:36-12:59) | 0 free, 0 hard | PASS | Complex implementation: binary mask approach, accumulator CB management, upstream program_descriptor modifications |
| normalize | ~19m (13:00-13:18) | 1 free, 1 hard | PASS | REDUCE_SCALAR scaler bug (8m debugging), session restart (~5m overhead), `add_tiles_bcast_scalar_init_short` hallucination (1m) |
| affine | ~9m (13:20-13:27) | 0 free, 1 hard | PASS | Test template bug (3m), CB scratch conflict (2m, caught proactively) |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | REDUCE_SCALAR scaler | kernel-writer-tdd | ~8m | 8% | Numerical mismatch requiring DPRINT debugging to discover scaler^2 behavior | 1 hard | Undocumented hardware behavior; no reference analysis covered reduce_tile stats |
| 2 | Session restart overhead | kernel-writer-tdd | ~5m | 5% | Context re-loading and state re-parsing between sessions during normalize stage | 0 | Context window exhaustion from upstream modifications |
| 3 | group_mean_subtract implementation | kernel-writer-tdd | ~24m | 25% | Novel binary mask approach requiring upstream changes to entry point, program descriptor, and multiple kernel modifications | 0 | Complex stage with design deviations; necessary innovation |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| kernel-writer-tdd | Initial normalize implementation with 1/K scaler | REDUCE_SCALAR applies scaler^2; had to change to 1/sqrt(K) | Document REDUCE_SCALAR squaring behavior in architect's critical notes |
| kernel-writer-tdd | Used cb_gamma as scratch in normalize | Had to be replaced with cb_scratch when affine stage loaded gamma | Allocate dedicated scratch CB from the start |
| kernel-writer-tdd | CB conflict analysis for cb_input_rm as scratch | Correctly rejected, then tried cb_gamma; eventually had to allocate new CB anyway | Two breadcrumbs (13:06:27, 13:07:05) show iterating through CB candidates |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | 3 analysis .md files (tilize, untilize, batch_norm) |
| Quality | GOOD |
| Issues | batch_norm analysis did not cover `reduce_tile` usage for stats computation (batch_norm receives pre-computed mean/var). This meant the architect and kernel writer had no reference for the REDUCE_SCALAR squaring behavior. |
| Downstream Impact | The architect designed with 1/K scaler, leading to Issue #1 (8 minutes debugging). |
| Suggestion | When the compute_core reference does not implement the same stats computation pattern as the target operation, the analyzer should note this gap explicitly: "batch_norm does NOT compute mean/var inline -- the reduce_tile stats pattern is novel for the target operation and has no reference." |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md, .tdd_state.json |
| Quality | GOOD |
| Issues | Minor: the architect's design referenced `"ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"` include path which does not exist for device-side kernels (builder's upstream feedback). |
| Downstream Impact | 1 free retry in builder phase to fix include paths in stubs. |
| Suggestion | The architect should not specify include paths -- leave that to the builder/kernel-writer who know the actual build system paths. |

### Handoff 3: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files |
| Quality | ADEQUATE |
| Issues | (1) The builder correctly noted the include path issue in handoff notes but the kernel writer still used wrong compute includes. (2) cb_mean and cb_den were allocated with 1 page each, requiring the kernel writer to expand to G pages. (3) The builder did not allocate CB 26 (group_scaler) or CB 8 (scratch) because the architect's design did not include them. |
| Downstream Impact | Kernel writer had to make 7 upstream_fix modifications to the program descriptor and entry point during stages 1-3. |
| Suggestion | The architect should be more conservative with CB sizing (allocate G pages for group-indexed CBs from the start). The builder should include all CBs from the design, even if some have conditional usage. |

### Handoff 4: ttnn-operation-architect -> ttnn-kernel-writer-tdd (design doc)

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2 |
| Quality | ADEQUATE |
| Issues | (1) Scaler = 1/K was wrong (should be 1/sqrt(K)). (2) The design left implementation details underspecified for Phase 3 (variance computation): "Simpler: pack mean^2 to cb_tmp, then sub_tiles(cb_sq_sum, cb_tmp)" -- the kernel writer deviated from this to use a more complex but correct approach. (3) Critical Note #4 on stats storage discussed alternatives without resolving them: "store stats in L1 directly (outside CBs) or recompute during normalize. Recomputation is simpler for v1." -- the kernel writer chose neither and instead stored all G stats in expanded CBs. |
| Downstream Impact | The kernel writer had to innovate beyond the design in multiple areas. This was ultimately successful but added complexity and time. |
| Suggestion | The architect should resolve all design alternatives in Part 2 rather than leaving options open. The phrase "The kernel writer should..." with multiple alternatives is a signal that a decision was deferred. |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-generic-op-builder | Remove device-side include paths from design docs; they are wrong for kernel compilation | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-operation-architect | Add column-subset reduce exception: "If reduction requires column-subset access within a CB, manual reduce_tile is acceptable" | HIGH | MEDIUM |
| tdd_orchestrator.py | ttnn-operation-architect | extra_args must include leading ", " -- document this or fix the template | HIGH | LOW |
| ttnn-kernel-writer-tdd | ttnn-kernel-writer-tdd | REDUCE_SCALAR squaring: add to known hardware behaviors | HIGH | HIGH |
| ttnn-generic-op-builder | ttnn-generic-op-builder | Standardize compute kernel include to `api/compute/compute_kernel_api.h` | MEDIUM | LOW |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| TDD | Kernel writer made 7 upstream fixes to program_descriptor.py and group_norm.py during stages 1-3 | Consider a "program descriptor evolution" mechanism where changes are tracked and validated, rather than having the kernel writer freely edit upstream files | MEDIUM |
| TDD | Session restart lost ~5 min in normalize stage | Add session handoff breadcrumb with structured state (current stage, what was attempted, what worked) | MEDIUM |
| Design | Architect left design alternatives unresolved (stats storage, Phase 3 details) | Require architect to make definitive decisions, even if the kernel writer may deviate; unresolved alternatives waste kernel writer time on evaluation | MEDIUM |
| Analysis | Compute_core reference (batch_norm) did not cover inline stats computation | When selecting references, verify the compute_core reference covers the core algorithm pattern, not just the post-stats normalization | HIGH |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | YES | The normalize stage consumed enough context to trigger a session restart. The REDUCE_SCALAR numerical debugging took ~8 minutes. However, the total debugging burden was moderate (~19m) compared to the worst cases described in the improvement tracker (~1 hour). |
| 2 | Too many planning stages (planner/designer) | RESOLVED | This run used the new merged architect agent. The March 10 run shows a clear improvement over the February 20-23 run, which used the old planner+designer pipeline (4 separate agents: planner, designer, builder, kernel-writer). |
| 3 | `.tdd_state.json` coupling fragility | NO | The .tdd_state.json worked correctly throughout this run. |
| 9 | No architect/builder cross-validation | YES | cb_mean was designed as 1 page but needed G pages. cb_den same issue. If a cross-validation step had compared the architect's "G pages" note (mentioned in Phase 1 pseudocode but not in the CB table) against the builder's allocation, this could have been caught earlier. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| REDUCE_SCALAR scaler squaring undocumented | `reduce_tile<SUM, REDUCE_SCALAR>` applies the scaler at both row and column reduction stages. This is not documented in the analysis docs, design docs, or any reference. Needs to be added to a known hardware behaviors reference. | HIGH |
| Test template missing function args from extra_setup | When `extra_setup` creates variables used in `reference_body`, those variables must also appear in `extra_args`. The template system does not validate this. | MEDIUM |
| Kernel writer CB scratch allocation pattern | When multiple TDD stages progressively repurpose CBs (e.g., using cb_gamma as scratch then later loading gamma into it), the kernel writer has to add new CBs mid-pipeline. A dedicated scratch CB should be in the initial design. | LOW |

---

## 8. Actionable Recommendations

### Recommendation 1: Document REDUCE_SCALAR squaring behavior

- **Type**: instruction_change
- **Target**: Architect agent system prompt, kernel-writer known hardware behaviors
- **Change**: Add a "Known Hardware Behaviors" section with: "`reduce_tile<SUM, REDUCE_SCALAR>` applies the scaler at both the row-reduction and column-reduction stages. The effective scaler is scaler^2. To achieve effective 1/K, pass 1/sqrt(K)."
- **Expected Benefit**: Prevents the 8-minute debugging cycle that occurred in this run and will recur for any op using reduce_tile for stats
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Require architect to resolve all design alternatives

- **Type**: instruction_change
- **Target**: Architect agent system prompt
- **Change**: Add instruction: "Part 2 must contain definitive implementation decisions. Do not leave alternatives for the kernel writer to evaluate. If you write 'Approach A or Approach B', choose one and justify it. If the kernel writer needs to deviate, they will, but the default path must be clear."
- **Expected Benefit**: Reduces kernel writer evaluation overhead; the group_mean_subtract stage took 24 minutes partly because the writer had to design a novel approach (binary mask) that went beyond the architect's under-specified plan
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Auto-validate extra_args covers extra_setup variables used in reference_body

- **Type**: new_validation
- **Target**: `tdd_orchestrator.py` or architect template generation
- **Change**: After TDD stage registration, parse `extra_setup` for variable assignments and check that any variable appearing in both `extra_setup` and `reference_body` also appears in `extra_args`. Warn if not.
- **Expected Benefit**: Prevents the test template bug that cost 1 hard attempt in the affine stage
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 4: Verify compute_core reference covers the target's core algorithm

- **Type**: pipeline_change
- **Target**: Discovery/orchestrator phase
- **Change**: After selecting a compute_core reference, validate that it implements the same compute pattern (e.g., "inline stats computation via reduce_tile" vs "receives pre-computed stats"). If the core algorithm pattern differs, add a note to the analysis or select an additional reference that covers the missing pattern.
- **Expected Benefit**: Would have flagged that batch_norm does not compute inline mean/var, prompting either a supplementary reference or explicit architect warning
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 5: Include dedicated scratch CB in initial design

- **Type**: instruction_change
- **Target**: Architect agent system prompt
- **Change**: Add instruction: "Always allocate at least one dedicated scratch CB (e.g., CB 8) in the initial design if any compute phase needs intermediate storage. Do not recommend reusing program-lifetime CBs as scratch in early stages."
- **Expected Benefit**: Prevents the CB conflict evolution seen in this run (cb_gamma -> cb_input_rm -> cb_gamma -> cb_scratch)
- **Priority**: LOW
- **Effort**: SMALL

### Recommendation 6: Standardize compute kernel include paths

- **Type**: instruction_change
- **Target**: Kernel-writer agent, builder agent
- **Change**: Add to both agents' reference tables: "Compute kernel includes: `#include \"api/compute/common.h\"`, `#include \"api/compute/bcast.h\"`, etc. NEVER use `compute_kernel_api/` prefix."
- **Expected Benefit**: Prevents the recurring free retry for wrong include paths
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Good reference selection; minor gap that batch_norm does not cover inline stats |
| Analysis quality | 5/5 | Thorough, well-structured, with explicit downstream-use sections |
| Design completeness | 3/5 | CB layout mostly correct but scaler value wrong; alternatives left unresolved; stats storage underspecified |
| Build correctness | 4/5 | Clean infrastructure; 1 include path issue; CB sizes needed expansion later |
| Kernel implementation | 4/5 | All 4 stages passed; only 2 hard attempts total; creative binary mask solution |
| Inter-agent communication | 3/5 | Analysis-to-architect handoff was excellent; architect-to-kernel-writer left too many decisions open; builder handoff notes were ignored by kernel writer |
| Logging/observability | 4/5 | Breadcrumbs present for all agents; timestamps consistent; session restart visible; execution logs complete for architect and builder (none for kernel-writer-tdd) |

### Top 3 Things to Fix

1. **Document REDUCE_SCALAR scaler squaring** -- This undocumented hardware behavior cost 8 minutes and will recur for every future operation using `reduce_tile` for statistics computation. Adding it to the architect's and kernel writer's reference material is high-impact, zero-effort.

2. **Require the architect to resolve all design alternatives** -- Leaving "Approach A or B" in the design doc forces the kernel writer to evaluate and choose, consuming precious context in an already context-heavy phase. The group_mean_subtract stage's 24-minute implementation time was partly driven by the need to invent the binary mask approach from scratch.

3. **Validate test template extra_args covers reference_body variables** -- The affine stage's test template bug is a systematic issue: the orchestrator template system has no cross-validation between `extra_setup` and `extra_args`. A simple parsing check would catch this class of bug before any hard attempt is consumed.

### What Worked Best

The analysis phase was the standout success of this pipeline run. The three parallel analyzers produced comprehensive, role-focused reference documents that directly enabled the architect's single-pass design. The explicit "For Group Norm Reference" sections in each analysis doc, combined with exact API signatures and CB configuration tables, gave the downstream agents a solid foundation. The architect's execution log explicitly credits the analyses: "All three analysis documents were thorough and provided the needed patterns." This quality propagated through the entire pipeline -- the 12 CBs that were correct from the start, the clean data_pipeline stage, and the successful tilize/untilize integration all trace back to the analysis quality.
