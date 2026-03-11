# Self-Reflection: softmax

## Metadata
| Field | Value |
|-------|-------|
| Operation | `softmax` |
| Operation Path | `ttnn/ttnn/operations/softmax` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | 3x ttnn-operation-analyzer, ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd, orchestrator |
| Total Git Commits | 10 (excluding 2 pre-existing unrelated commits) |
| Total Pipeline Duration | ~71 minutes (17:06 - 18:17 UTC) |
| Overall Result | SUCCESS -- all 4 TDD stages passed |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~1m | DONE | Selected 3 references: tt-train softmax, reduce_w, reduce_h |
| 1: Analysis | 3x ttnn-operation-analyzer | ~11m (parallel) | DONE | 3 analysis files totaling ~1600 lines; covered dim=-1 and dim=-2 patterns |
| 2: Design | ttnn-operation-architect | ~4m | DONE | Produced comprehensive op_design.md (322 lines) with 7 CBs, 3-pass architecture, 4 TDD stages |
| 3: Build | ttnn-generic-op-builder | ~12m | DONE | Created all infrastructure; 15/15 integration tests passed; 1 kernel include error fixed |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~35m | DONE | 4/4 TDD stages passed; 3 hard attempts on stage 3 (softmax_dim_w) |
| 5: Report | orchestrator | ~2m | DONE | Generated REPORT.md |

### Agent Duration Breakdown

Duration calculation method: Primary source is breadcrumb `"event":"start"` and `"event":"complete"` timestamps. Git commit timestamps used as cross-check. All agents had both start and complete breadcrumb events.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (x3) | 17:06:15 | 17:17:31 | 11m 16s | 0 | ~11m active (all productive) |
| ttnn-operation-architect | 17:19:57 | 17:23:58 | 4m 1s | 0 | ~4m active (all productive) |
| ttnn-generic-op-builder | 17:26:37 | 17:38:46 | 12m 9s | 1 free | ~10m active, ~2m debugging kernel include path |
| ttnn-kernel-writer-tdd | 17:40:14 | 18:15:08 | 34m 54s | 1 free + 3 hard | ~12m active coding, ~23m debugging stage 3 |

### Duration Visualization

```
Phase 0  |#|                                                          (~1m)
Phase 1  |########---|                                                (~11m) 3 analyzers in parallel
Phase 2              |####|                                           (~4m)
Phase 3                    |############|                             (~12m)
Phase 4                                  |###########################| (~35m) <-- longest
Phase 5                                                              |##| (~2m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70 min

Longest phase: Phase 4 (35m) -- stage 3 (softmax_dim_w) consumed 23 of 35 minutes due to 3 hard debugging cycles
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~12m | 17% | 3 analyzers in parallel; efficient |
| Design (Phase 2) | ~4m | 6% | Fast, comprehensive output |
| Build (Phase 3) | ~12m | 17% | Includes 1 include-path fix |
| Kernel implementation (Phase 4) | ~35m | 49% | 4 TDD stages |
| -- Productive coding | ~12m | 17% | Stages 1+2+4 + initial stage 3 code |
| -- Debugging/retries | ~23m | 32% | Stage 3 debugging: numerical mismatch, hang, CB sizing |
| Reporting (Phase 5) | ~2m | 3% | |
| Inter-phase gaps | ~6m | 8% | Agent startup overhead between phases |
| **Total** | **~71m** | **100%** | |

---

## 2. What Went Well

### 1. TDD stage progression was well-designed

**Phase/Agent**: ttnn-operation-architect (Phase 2)
**Evidence**: The 4-stage plan (data_pipeline -> exp_only -> softmax_dim_w -> softmax_dim_h) provided excellent incremental scaffolding. Stage 1 and 2 both passed on first attempt (0 hard retries). Stage 4 (dim_h) also passed first attempt because the dim_w implementation cleanly generalized. The TDD stages isolated complexity: only stage 3 required debugging, and its issues were clearly within the softmax compute logic, not infrastructure.
**Why it worked**: The architect separated infrastructure validation (stages 1-2) from algorithmic complexity (stages 3-4), and separated the two dimensions (stages 3-4). This meant stage 3 failures were unambiguously compute-kernel bugs, not reader/writer issues.

### 2. dim=-2 (height reduction) passed on first attempt

**Phase/Agent**: ttnn-kernel-writer-tdd (Phase 4, Stage 4)
**Evidence**: Stage 4 (softmax_dim_h) passed in ~3 minutes with 0 retries. The kernel writer needed only to fix 3 items: (a) `add_bcast_rows_init_short` instead of nonexistent `sub_bcast_rows_init_short`, (b) `ReduceInputBlockShape::col()` for REDUCE_COL, (c) `mul_bcast_rows_init_short`. All fixes were applied proactively before the first test run.
**Why it worked**: The reduce_h analysis (reduce_h_analysis.md) gave the kernel writer the exact broadcast directions and tile ordering for REDUCE_COL. The dim=-1 implementation in stage 3 provided a battle-tested template that only needed broadcast direction and shape changes.

### 3. Analysis quality was high and directly useful

**Phase/Agent**: ttnn-operation-analyzer (Phase 1)
**Evidence**: The architect explicitly cited all 3 analyses in the design doc. Key findings that directly shaped the design: (a) the 3-pass streaming pattern from tt-train analysis, (b) the `reduce<>` helper API with ReduceInputPolicy from reduce_w analysis, (c) column-major tile ordering for REDUCE_COL from reduce_h analysis, (d) the `generate_mm_scaler` utility for matmul-based sum. The architect did not need to re-read any reference source code.
**Why it worked**: Three distinct references covered complementary aspects (core algorithm, dim=-1 pattern, dim=-2 pattern), and all three analyzers produced detailed, structured reports with exact API signatures and CB layouts.

### 4. Builder handoff notes prevented kernel writer confusion

**Phase/Agent**: ttnn-generic-op-builder (Phase 3)
**Evidence**: The builder's execution log Section 6 (Handoff Notes) explicitly warned: "The `tensor_accessor.hpp` file referenced in instructions does NOT exist. TensorAccessor is available via `api/dataflow/dataflow_api.h`." It also documented all compile-time arg indices, runtime arg layout, and preprocessor define names. The kernel writer did not encounter any TensorAccessor include issues and did not misinterpret any argument positions.
**Why it worked**: The builder proactively surfaced issues it discovered (wrong include path) so the kernel writer would not repeat the same mistake.

### 5. CB synchronization verification was thorough

**Phase/Agent**: ttnn-kernel-writer-tdd (Phase 4)
**Evidence**: The kernel writer logged `cb_sync_check` breadcrumbs after every kernel implementation (5 total across all stages). Each check explicitly enumerated push/pop counts per CB per phase. No CB-related bugs occurred at runtime despite 7 CBs with complex 3-phase producer/consumer relationships. The one hang (stage 3, attempt 2) was a DST register deadlock, not a CB synchronization issue.
**Why it worked**: The kernel writer treated CB balance verification as a required checkpoint before each test run, catching potential deadlocks at design time rather than runtime.

---

## 3. Issues Found

### Issue 1: matmul-based sum accumulation caused DST register deadlock (hang)

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w (stage 3) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 2 hard attempts (numerical mismatch + hang) |
| Time Cost | ~18 minutes (17:48 - 18:06, from stage start to revised approach) |

**Problem**: The architect's design specified matmul-based sum accumulation (following the tt-train reference pattern) using `matmul_tiles(cb_exp, cb_mm_scaler, ...)` within an `acquire_dst/release_dst` block to keep the DST[0] accumulator alive across the inner_dim loop. The first implementation lost the DST accumulator between tiles because `tile_regs_release/acquire` cycles cleared DST (attempt 1, numerical mismatch: max diff 0.51). The fix using `acquire_dst/release_dst` full-sync mode caused a hang (attempt 2) because `pack_tile` calls within the acquire_dst block required `tile_regs_commit/tile_regs_wait` that conflicted with the full-sync mode.

Breadcrumb evidence:
- `{"event":"hypothesis","id":"H2",...,"description":"DST accumulator (DST[0]) for matmul sum is lost between tiles...","confidence":"HIGH"}` at 17:54:10
- `{"event":"test_run","status":"hang","detail":"Device timeout on first shape 1x1x32x32..."}` at 18:00:50
- `{"event":"hypothesis","detail":"REVISED: acquire_dst deadlock confirmed...NEW APPROACH: Replace matmul-based sum with reduce<SUM>..."}` at 18:06:09

**Root Cause**: The architect recommended matmul-based sum accumulation based on the tt-train reference, which uses a different DST register management strategy (the tt-train kernel manages DST lifetime across the entire compute phase, while the pipeline's kernel mixes reduce helper calls with manual DST management). The incompatibility between `acquire_dst/release_dst` (used by matmul accumulation) and `tile_regs_acquire/release` (used by the reduce helper) was not documented anywhere. The architect's design note said "matmul-based approach via `generate_mm_scaler` tile is mandatory for the sum phase" (op_design.md line 307) -- this turned out to be wrong; `reduce<SUM>` with `recip_tile` post_reduce_op achieves acceptable precision.

**Fix for agents**:
- **ttnn-operation-architect**: When recommending matmul-based sum accumulation, add an explicit warning: "matmul_tiles requires acquire_dst/release_dst full-sync mode, which is INCOMPATIBLE with pack_tile and reduce helper calls in the same phase. If the kernel also uses reduce<> helpers or per-tile pack_tile within the accumulation loop, use reduce<SUM> with a post_reduce_op instead." Also, the word "mandatory" in design docs should be reserved for hardware constraints, not precision preferences.
- **Pipeline instructions (known incompatibility database)**: Add `acquire_dst/release_dst` + `tile_regs_acquire/release` mixing as a documented incompatibility.

### Issue 2: Wrong include path for exp.h (sfpu/ subdirectory hallucination)

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- exp_only (stage 2) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 1 free retry |
| Time Cost | ~1 minute |

**Problem**: The kernel writer used `#include "api/compute/eltwise_unary/sfpu/exp.h"` but the correct path is `api/compute/eltwise_unary/exp.h` (no `sfpu/` subdirectory). The error was caught immediately by compilation failure and fixed in under a minute.

Breadcrumb evidence:
- `{"event":"test_run","status":"fail","stage":"exp_only","failure_type":"compile","details":"Include path api/compute/eltwise_unary/sfpu/exp.h not found"}` at 17:46:33
- `{"event":"hypothesis","id":"H1",...,"description":"Wrong include path...should be api/compute/eltwise_unary/exp.h","confidence":"HIGH"}` at 17:46:44

**Root Cause**: The kernel writer hallucinated an `sfpu/` subdirectory that does not exist. The SFPU (vector unit) does process exp, so the name is semantically plausible, but the include tree does not mirror the hardware architecture this way.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: Add to the kernel writer's include reference table: `exp_tile -> #include "api/compute/eltwise_unary/exp.h"` (NOT `sfpu/exp.h`). Similarly for `recip_tile -> api/compute/eltwise_unary/recip.h`.
- **Pipeline tooling**: A curated include path database (as suggested in REPORT.md) would prevent this class of errors entirely.

### Issue 3: CB sizing mismatch -- architect designed cb_exp_sum with 2 tiles, needed inner_dim

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w (stage 3) |
| Agent | ttnn-operation-architect (design), ttnn-kernel-writer-tdd (discovery) |
| Retries Consumed | 0 additional (discovered during design pivot, not a separate test failure) |
| Time Cost | ~2 minutes (part of the matmul->reduce pivot) |

**Problem**: The architect's CB table specified cb_exp_sum (c_4) with 2 tiles for "double-buffering compute self-production/consumption" (op_design.md line 74). However, when the kernel writer pivoted to the `reduce<SUM>` approach (replacing matmul accumulation), Phase 2 needed to buffer ALL inner_dim exp tiles in c_4 before the reduce helper could consume them. The kernel writer had to upstream-fix `softmax_program_descriptor.py` to change `2 * page_size` to `inner_dim * page_size`.

Breadcrumb evidence:
- `{"event":"upstream_fix","file":"softmax_program_descriptor.py","change":"Increased cb_exp_sum from 2 tiles to inner_dim tiles","reason":"Phase 2 needs to buffer all inner_dim exp tiles before reduce<SUM> consumes them."}` at 18:06:27

**Root Cause**: The architect designed the CB sizing for the matmul-based accumulation strategy (where exp tiles are consumed one at a time), not for the reduce-helper strategy (where all exp tiles must be present before the reduce starts). This is a valid design for the original approach, but the design did not account for the fallback path. More broadly, the architect does not validate CB sizes against data flow -- how many tiles must coexist in a CB at peak occupancy depends on the consumer's consumption pattern.

**Fix for agents**:
- **ttnn-operation-architect**: For each CB, document the "peak occupancy" scenario (max tiles simultaneously live), not just the steady-state double-buffering count. When using reduce helpers as consumers, note that `WaitAndPopPerTile` policy can stream tiles one at a time (2 tiles sufficient), but if tiles must be buffered before reduction starts, the CB needs `inner_dim` tiles.
- **Pipeline validation**: Add a post-design check that CB sizes are consistent with the reduce helper's input policy (WaitAndPopPerTile = streaming, WaitUpfrontNoPop = requires all tiles buffered).

### Issue 4: Builder had to fix 4 auto-generated stage test files for syntax errors

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-generic-op-builder (fixer), ttnn-operation-architect (generator) |
| Retries Consumed | 0 (builder fixed proactively) |
| Time Cost | ~2-3 minutes of builder time |

**Problem**: The architect auto-generated 4 stage test files (test_stage_data_pipeline.py, etc.) that all contained syntax errors: missing `return` in `pytorch_reference()`, missing commas in function signatures (e.g., `input_tensordim=-1` instead of `input_tensor, dim=-1`), using `x` instead of `input_tensor` as the variable name, and relative import paths that fail from the test directory. The builder had to fix all 4 files before they could run.

Evidence from builder execution log:
- Section 1 Upstream Feedback: "Auto-generated stage test files had multiple syntax errors: missing `return` in `pytorch_reference`, missing commas in function signatures (`input_tensordim=-1`), using `x` instead of `input_tensor` in reference bodies, relative import paths..."
- Git diff (commit 51d95a2417) shows 17-23 line modifications per test file.

**Root Cause**: The architect's test template generation does not properly handle `extra_args` injection into function signatures, nor does it validate that generated Python files parse successfully. The template likely concatenates `input_tensor` + `dim=-1` without inserting the comma separator.

**Fix for agents**:
- **ttnn-operation-architect**: After generating stage test files, run `python -c "import ast; ast.parse(open('test_file.py').read())"` to validate syntax. Fix the template to (a) always insert `return` before the reference body, (b) properly separate extra_args with `, `, (c) use the canonical variable name `input_tensor` not `x`.
- **Orchestrator**: Add a syntax validation step after the architect finishes before launching the builder.

### Issue 5: Program descriptor variable ordering caused UnboundLocalError

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w (stage 3) |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 0 (classified as free; Python error, not device issue) |
| Time Cost | ~30 seconds |

**Problem**: After increasing cb_exp_sum to `inner_dim * page_size`, the program descriptor used `inner_dim` in the CB section (section 3) before it was computed in the work distribution section (section 4). This caused a Python `UnboundLocalError` at test time.

Breadcrumb evidence:
- `{"event":"test_run","status":"fail","detail":"Python UnboundLocalError: inner_dim referenced before assignment in program descriptor."}` at 18:09:04

**Root Cause**: The kernel writer added the `inner_dim` dependency to the CB section without checking that `inner_dim` was already in scope. The original program descriptor had a fixed `2 * page_size` that did not depend on work distribution variables. The fix was trivial: move work distribution before CB descriptors.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: When modifying program descriptor Python files, always check that new variable references are defined above the point of use. Run a quick `python -c "import ..."` check before committing changes.

### Issue 6: cb_mm_scaler (c_2) is generated but never consumed

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- softmax_dim_w (stage 3) |
| Agent | ttnn-operation-architect (design), ttnn-kernel-writer-tdd (implementation) |
| Retries Consumed | 0 |
| Time Cost | 0 (no test failures, just wasted L1 and reader cycles) |

**Problem**: The architect designed c_2 (cb_mm_scaler) for the matmul-based sum accumulation path. When the kernel writer pivoted to `reduce<SUM>`, c_2 was no longer needed by the compute kernel. However, the reader still generates it at startup via `generate_mm_scaler(cb_mm_scaler, packed_bf16_1_0)`, and the CB is still allocated (1 tile = 2KB in L1). This is dead code / dead resource allocation.

**Root Cause**: The kernel writer focused on getting the compute kernel working and did not clean up the reader or program descriptor to remove the unused CB. The REPORT.md acknowledges this as "harmless overhead."

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: After pivoting away from a design element, add a cleanup step: identify all CBs/args/defines that are no longer referenced by any kernel and remove them from the program descriptor and reader.

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~4m (17:40:45-17:44:50) | 0 free, 0 hard | PASS | Clean -- needed 2 upstream fixes (reader CT args) but no test failures |
| exp_only | ~2.5m (17:45:20-17:47:53) | 1 free, 0 hard | PASS | 1 free retry for include path; trivial fix |
| softmax_dim_w | ~23m (17:48:13-18:11:30) | 0 free, 3 hard | PASS | Dominated by matmul DST deadlock debugging |
| softmax_dim_h | ~3m (18:12:19-18:15:04) | 0 free, 0 hard | PASS | Clean first-pass from dim_w template |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Stage 3 debugging | ttnn-kernel-writer-tdd | ~18m | 25% | matmul accumulation -> numerical mismatch -> DST deadlock hang -> pivot to reduce<SUM> | 2 hard | Architect recommended incompatible matmul pattern |
| 2 | Stage 3 restructuring | ttnn-kernel-writer-tdd | ~5m | 7% | Rewriting Phase 2 compute + CB sizing fix + program descriptor variable ordering | 1 hard | Fallout from approach pivot; new code still had a Python ordering bug |
| 3 | Builder test cycle | ttnn-generic-op-builder | ~4m | 6% | First test run failed (tensor_accessor.hpp), fix, rerun | 1 free | Incorrect include path in builder instructions |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | Implemented matmul-based sum accumulation with acquire_dst/release_dst full-sync mode (~4 minutes of coding at 17:54-17:58) | Entire approach abandoned due to DST deadlock; replaced with reduce<SUM> helper | Architect should warn about acquire_dst/pack_tile incompatibility; or provide both strategies in the design with fallback guidance |
| ttnn-kernel-writer-tdd | Generated cb_mm_scaler tile + CB allocation | Never used after pivot to reduce<SUM> | Kernel writer should clean up unused resources after pivoting |
| ttnn-operation-architect | Designed matmul-based sum as "mandatory" for precision | Precision proved acceptable with reduce<SUM> (tests pass at rtol=0.05, atol=0.2) | Architect should present matmul-based sum as "preferred for precision" not "mandatory"; provide reduce<SUM> as documented fallback |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: Analyzers -> Architect

| Field | Value |
|-------|-------|
| Artifact Passed | 3 analysis files (~1600 lines total): softmax_tt_train_analysis.md, reduce_w_analysis.md, reduce_h_analysis.md |
| Quality | GOOD |
| Issues | None significant. All 3 analyses were referenced in the design doc. The matmul-based sum recommendation from the tt-train analysis was technically correct for the tt-train context but problematic in the pipeline context (different DST management). |
| Downstream Impact | The architect faithfully adopted the matmul-based sum recommendation, which later caused the stage 3 hang. |
| Suggestion | Analyzers should note when a reference implementation uses DST management patterns that differ from the reduce helper library. Specifically, the tt-train analysis should have noted: "This implementation does NOT use the reduce<> helper library for sum; it manages DST registers manually via acquire_dst. This pattern is incompatible with mixing reduce helper calls." |

### Handoff 2: Architect -> Builder

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md (322 lines), .tdd_state.json (144 lines), 4 stage test files |
| Quality | ADEQUATE |
| Issues | (1) Stage test files had syntax errors (Issue 4 above). (2) op_design.md was comprehensive for kernel implementation but the CB sizing for c_4 was optimistic (2 tiles, later needed inner_dim). |
| Downstream Impact | Builder spent ~2-3 minutes fixing test file syntax. CB sizing issue did not affect builder (stubs dont use CBs) but did affect kernel writer. |
| Suggestion | Architect should validate generated Python test files with AST parsing before handoff. |

### Handoff 3: Builder -> Kernel Writer

| Field | Value |
|-------|-------|
| Artifact Passed | Python infrastructure (softmax.py, softmax_program_descriptor.py, __init__.py), 3 stub kernels, 15 passing integration tests, execution log with handoff notes |
| Quality | GOOD |
| Issues | (1) Kernel stubs were truly empty (just `void kernel_main() {}` with a single include), which meant the kernel writer had to set up all include paths from scratch. (2) The builder's handoff notes (Section 6) were excellent -- explicitly documented the wrong tensor_accessor.hpp path and all arg indices. |
| Downstream Impact | The kernel writer needed 2 upstream fixes to the program descriptor (adding reader CT args num_rows_or_cols and num_tiles) and 1 more later (cb_exp_sum sizing), but these were quick modifications. |
| Suggestion | Builder could pre-populate compile-time arg #define names as comments in stub kernels (e.g., `// CT arg 0: Wt, 1: Ht, 2: num_rows_or_cols, 3: num_tiles`) to reduce kernel writer lookup time. |

### Handoff 4: Architect -> Kernel Writer (op_design.md Part 2)

| Field | Value |
|-------|-------|
| Artifact Passed | op_design.md Part 2: Kernel Implementation (~190 lines) |
| Quality | ADEQUATE |
| Issues | (1) The matmul-based sum recommendation was marked "mandatory" but turned out to be replaceable with reduce<SUM> (Issue 1). (2) The design used the word "mandatory" (line 307) for a precision preference, which caused the kernel writer to spend extra time trying to make matmul work before pivoting. (3) CB sizing for c_4 assumed streaming (2 tiles) but the fallback approach required buffering (inner_dim tiles). |
| Downstream Impact | 18 minutes of debugging and approach pivoting in stage 3. |
| Suggestion | Design docs should distinguish between "required by hardware" constraints and "recommended for precision" preferences. When recommending specific compute patterns, provide an explicit fallback: "If matmul-based sum causes DST conflicts, use reduce<SUM, REDUCE_DIM> with recip_tile post_reduce_op as fallback. This may require increasing cb_exp_sum to inner_dim tiles." |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-kernel-writer-tdd | ttnn-kernel-writer-tdd | Add exp.h include path to reference table (NOT sfpu/exp.h) | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Warn about acquire_dst/release_dst + tile_regs_acquire/release incompatibility | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | CB sizing should account for consumer pattern (streaming vs. buffered) | HIGH | HIGH |
| ttnn-operation-architect | ttnn-generic-op-builder | Stage test template must produce syntactically valid Python (run AST parse) | HIGH | MEDIUM |
| ttnn-generic-op-builder | ttnn-generic-op-builder | Remove tensor_accessor.hpp from include mapping; it does not exist | HIGH | MEDIUM |
| ttnn-generic-op-builder | ttnn-generic-op-builder | Document correct kernel_lib include prefix as `ttnn/kernel_lib/X.hpp` | HIGH | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Design | Architect used "mandatory" for a precision preference, blocking kernel writer from considering alternatives | Introduce terminology standard: "REQUIRED" = hardware constraint, "RECOMMENDED" = best practice with fallback | HIGH |
| Design | CB sizing did not account for reduce helper consumption patterns | Add CB sizing validation step that cross-references helper input policies | MEDIUM |
| Build | Builder instructions reference nonexistent tensor_accessor.hpp | Update builder include mapping table | MEDIUM |
| TDD | Stage 3 consumed 65% of Phase 4 time (23m of 35m) | Consider splitting complex stages: softmax_dim_w could be softmax_dim_w_single_tile + softmax_dim_w_multi_tile to isolate accumulation bugs earlier | LOW |
| Cleanup | Kernel writer left dead CB (c_2) and reader code after approach pivot | Add post-implementation cleanup checklist: "Are all CBs consumed? Are all generated tiles read?" | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns massive context on numerical debugging | YES | Stage 3 consumed 23 minutes with 3 hard attempts. The numerical mismatch (attempt 1) was correctly diagnosed quickly, but the subsequent hang (attempt 2) required a full approach pivot. Total debugging was ~18 minutes -- significant but not the "hour+ grind" described in the known issue. |
| 3 | .tdd_state.json coupling fragility | NO | .tdd_state.json worked correctly throughout. No schema issues. |
| 6 | Builder runs on Sonnet while everything else uses Opus | POSSIBLY | Builder had 1 failed test run due to wrong include path. Not conclusive evidence of model capability issues. |
| 9 | No validation between architect output and builder output | YES | CB c_4 was sized at 2 tiles by architect and built with 2 tiles by builder, but the kernel writer later needed inner_dim tiles. A cross-validation step would not have caught this because the architect's spec also said 2. The issue is upstream in the architect's sizing analysis. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| acquire_dst/tile_regs incompatibility undocumented | Mixing `acquire_dst/release_dst` (matmul accumulation) with `tile_regs_acquire/release` (reduce helpers, per-tile pack) causes DST deadlocks. Neither the architect nor kernel writer had guidance on this. | HIGH |
| Architect stage test generation produces invalid Python | All 4 auto-generated stage test files had syntax errors (missing returns, missing commas, wrong variable names). Builder fixes them every time. | MEDIUM |
| Design "mandatory" vs "recommended" ambiguity | Architect wrote "mandatory" for matmul-based sum (a precision preference), which blocked the kernel writer from considering reduce<SUM> as a first option. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Document DST register management incompatibilities

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt, kernel writer reference docs
- **Change**: Add a "DST Register Mode Compatibility" section documenting: (a) `tile_regs_acquire/release` = half-sync, used by reduce helpers and per-tile compute; (b) `acquire_dst/release_dst` = full-sync, used by matmul accumulation; (c) These modes CANNOT be mixed within the same compute phase -- pack_tile inside acquire_dst requires tile_regs_commit/wait which conflicts with full-sync semantics; (d) If a design uses reduce helpers AND needs sum accumulation, use `reduce<SUM>` with post_reduce_op, not manual matmul accumulation.
- **Expected Benefit**: Eliminates the most expensive debugging cycle in this run (18 minutes, 2 hard attempts).
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add "mandatory" vs "recommended" distinction to architect vocabulary

- **Type**: instruction_change
- **Target**: ttnn-operation-architect agent prompt
- **Change**: Define terminology: "REQUIRED" or "MUST" = hardware constraint or correctness requirement. "RECOMMENDED" or "SHOULD" = best practice with explicit fallback. When the architect writes a compute strategy recommendation, require a fallback: "RECOMMENDED: matmul-based sum for precision. FALLBACK: reduce<SUM> + recip_tile post_reduce_op if DST management conflicts arise. CB implication: cb_exp_sum needs inner_dim tiles instead of 2."
- **Expected Benefit**: Kernel writer can immediately pivot to fallback instead of spending time debugging the primary approach.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 3: Validate auto-generated stage test files with AST parsing

- **Type**: new_validation
- **Target**: ttnn-operation-architect agent (test generation step)
- **Change**: After generating each stage test file, run `python -c "import ast; ast.parse(open(path).read())"` and fix any SyntaxError before committing. The template should also: (a) always prefix the reference body with `return`, (b) separate `extra_args` with `, ` not just concatenation, (c) use `input_tensor` consistently as the reference function parameter name.
- **Expected Benefit**: Eliminates 2-3 minutes of builder time per operation fixing test syntax.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Add curated compute kernel include path table

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd agent prompt
- **Change**: Add a verified include path reference table: `exp_tile -> api/compute/eltwise_unary/exp.h`, `recip_tile -> api/compute/eltwise_unary/recip.h`, `copy_tile -> api/compute/tile_move_copy.h`, `sub_tiles_bcast/mul_tiles_bcast -> api/compute/bcast.h`, `reduce helpers -> ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `binary_op helpers -> ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`.
- **Expected Benefit**: Prevents free retries from include path hallucinations.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: CB sizing validation against reduce helper consumption pattern

- **Type**: new_validation
- **Target**: ttnn-operation-architect agent prompt
- **Change**: For each CB consumed by a reduce helper, the architect should annotate whether the helper streams tiles (WaitAndPopPerTile -- 2 tiles sufficient) or requires all tiles upfront (WaitUpfrontNoPop or when tiles are produced before reduce starts -- needs inner_dim tiles). The CB page count should reflect the worst-case consumption pattern across all design strategies (including fallbacks).
- **Expected Benefit**: Prevents CB sizing mismatches that require upstream fixes during TDD.
- **Priority**: MEDIUM
- **Effort**: MEDIUM

### Recommendation 6: Post-pivot cleanup checklist for kernel writer

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd agent prompt
- **Change**: After pivoting away from a design element (e.g., abandoning matmul-based sum), the kernel writer should run a cleanup pass: (1) identify CBs no longer referenced by any kernel, (2) remove their allocation from program_descriptor.py, (3) remove their generation from the reader kernel, (4) remove unused compile-time defines. Log a `cleanup` breadcrumb event.
- **Expected Benefit**: Cleaner final code; avoids wasted L1 memory and reader cycles.
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 4/5 | Selected 3 complementary references that covered both dimensions; all were useful to the architect |
| Analysis quality | 4/5 | Detailed, structured reports with exact API signatures; minor gap: did not flag DST management differences between tt-train and reduce helper patterns |
| Design completeness | 3/5 | Comprehensive architecture and kernel design, but matmul-based sum recommendation caused the biggest debugging cycle; CB sizing was optimistic; stage test files had syntax errors |
| Build correctness | 4/5 | 15/15 tests passed after 1 include fix; handoff notes were excellent; stage test syntax fixes were handled proactively |
| Kernel implementation | 4/5 | All 4 stages passed within budget; kernel writer showed strong debugging methodology (hypothesis -> evidence -> fix); clean CB sync verification; but 23 minutes on stage 3 is costly |
| Inter-agent communication | 3/5 | Handoff notes were strong, but the "mandatory" matmul recommendation created a costly dead end; stage test syntax errors add friction to every run |
| Logging/observability | 5/5 | All 4 agents produced breadcrumbs with timestamps; kernel writer logged cb_sync_check, hypothesis, upstream_fix events; builder produced a full execution log; the timeline was fully reconstructable |

### Top 3 Things to Fix

1. **Document DST register mode incompatibilities** (acquire_dst vs tile_regs) -- this single gap caused 25% of total pipeline time to be spent debugging.
2. **Distinguish "mandatory" from "recommended" in design docs** with explicit fallback strategies including CB sizing implications -- prevents kernel writers from over-investing in approaches that may not work.
3. **Validate auto-generated stage test files** with AST parsing before handoff to builder -- eliminates a recurring friction point in every pipeline run.

### What Worked Best

The **TDD stage design** was the strongest aspect of this pipeline run. The 4-stage progression (data_pipeline -> exp_only -> softmax_dim_w -> softmax_dim_h) provided clean isolation of concerns: infrastructure was validated before algorithmic complexity was introduced, and the two dimension variants were separated. This meant that when stage 3 hit issues, the kernel writer could confidently focus on the compute kernel's sum accumulation logic without questioning the reader, writer, or CB infrastructure. Stages 1, 2, and 4 all passed on first attempt, demonstrating that the incremental approach works extremely well when the design correctly scaffolds complexity.
