# Self-Reflection: rms_norm

## Metadata
| Field | Value |
|-------|-------|
| Operation | `rms_norm` |
| Operation Path | `ttnn/ttnn/operations/rms_norm` |
| Pipeline Phases Executed | 0 (Discovery), 1 (Analysis), 2 (Design), 3 (Build), 4 (TDD Kernels), 5 (Report) |
| Agents Invoked | ttnn-operation-analyzer (x3), ttnn-operation-architect, ttnn-generic-op-builder, ttnn-kernel-writer-tdd |
| Total Git Commits | 12 |
| Total Pipeline Duration | ~105 minutes (17:06 - 18:51 UTC) |
| Overall Result | SUCCESS |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | ~5m (est.) | Complete | 3 references identified: tilize, untilize, moreh_norm_w |
| 1: Analysis | ttnn-operation-analyzer (x3) | ~12m (17:06 - 17:18) | Complete | 3 parallel analyzers produced tilize, untilize, moreh_norm_w analyses |
| 2: Design | ttnn-operation-architect | ~10m (17:20 - 17:29) | Complete | Hybrid mode, 8-phase compute, 4 TDD stages |
| 3: Build | ttnn-generic-op-builder | ~12m (17:30 - 17:41) | Complete | 14 files created, 3 compilation fixes needed |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | ~64m (17:45 - 18:49) | Complete | 4 stages, 4 hard + 1 free failures total |
| 5: Report | orchestrator | ~2m (18:49 - 18:51) | Complete | REPORT.md generated |

### Agent Duration Breakdown

Duration derived from breadcrumb `"event":"start"` and `"event":"complete"` timestamps. Git commit timestamps used as cross-check.

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| ttnn-operation-analyzer (tilize) | 17:06:22 | 17:18:42 | 12m 20s | 0 | ~12m active (no debugging) |
| ttnn-operation-analyzer (untilize) | 17:06:31 | 17:18:27 | 11m 56s | 0 | ~12m active (no debugging) |
| ttnn-operation-analyzer (moreh_norm_w) | 17:07:21 | 17:18:13 | 10m 52s | 0 | ~11m active (no debugging) |
| ttnn-operation-architect | 17:20:11 | 17:29:00 | 8m 49s | 0 | ~9m active (no debugging) |
| ttnn-generic-op-builder | 17:30:50 | 17:41:38 | 10m 48s | 2 | ~6m active, ~5m fixing compilation errors |
| ttnn-kernel-writer-tdd | 17:45:08 | 18:48:39 | 63m 31s | 5 | ~35m active coding, ~28m debugging hangs/errors |

**Duration calculation method**: Used breadcrumb `start` and `complete` timestamps as primary source. Git commit timestamps align within 1-2 minutes.

### Duration Visualization

```
Phase 0  |###|                                                       (~5m, estimated)
Phase 1  |############|                                              (~12m) 3 analyzers in parallel
Phase 2              |#########|                                     (~9m)
Phase 3                       |###########|                          (~11m)
Phase 4                                   |###############################...############|  (~64m)
Phase 5                                                                                |##| (~2m)
         0    5    10   15   20   25   30   35   40   45   50   55   60   65   70 min

Longest phase: Phase 4 (64m) -- 4 TDD stages with debugging in stage 2
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | ~17m | 16% | 3 analyzers ran in parallel |
| Design (Phase 2) | ~9m | 9% | Clean, no retries |
| Build (Phase 3) | ~11m | 10% | 2 compilation fix cycles |
| Kernel implementation (Phase 4) | ~64m | 61% | 4 TDD stages |
| -- Productive coding | ~35m | 33% | Writing kernel code that passed |
| -- Debugging/retries | ~28m | 27% | Stage 2 CB deadlock debugging dominated |
| Reporting (Phase 5) | ~2m | 2% | |
| **Total** | **~105m** | **100%** | |

---

## 2. What Went Well

### 1. Reference selection was excellent

**Phase/Agent**: Phase 0 (Discovery) / Phase 1 (Analysis)
**Evidence**: Three references (tilize, untilize, moreh_norm_w) with clearly differentiated roles (input_stage, output_stage, compute_core) covered all aspects of the operation. The moreh_norm_w analysis explicitly called out the W-dimension reduction pattern and `reduce<SUM, REDUCE_ROW>` helper usage. The tilize analysis documented the 32-stick batching pattern and TensorAccessor usage. The untilize analysis documented the `untilize<>()` helper signature and writer stick-extraction pattern.
**Why it worked**: The hybrid mode with role-based reference assignment gave the architect three focused, non-overlapping perspectives. Each analysis was written with explicit "relevance to RMS norm" sections that the architect could directly use.

### 2. Stage 1 (data_pipeline) and Stage 4 (rms_norm_with_gamma) passed on first attempt

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: `data_pipeline` had 0 hard attempts and 0 free retries in `.tdd_state.json`. `rms_norm_with_gamma` also had 0 hard attempts and 0 free retries. Both passed all 8 tests (4 shapes x 2 layouts) on the first run.
**Why it worked**: For Stage 1, the kernel writer performed 4 upstream fixes to the program descriptor proactively (tile_size, cb_in/cb_out page_size, stick_size layout-aware) before running any tests. For Stage 4, the kernel writer proactively identified the cb_normed page count deadlock risk (breadcrumb: "mul_col pushes 1 tile at a time...cb_normed with 1 page for gamma path will deadlock") and fixed it before running the test.

### 3. Kernel writer's proactive CB sync checks prevented issues

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: The kernel writer logged `cb_sync_check` breadcrumbs before running tests at every stage. For example, the Stage 4 check at 18:47:18 documented: "cb_normed now has Wt pages for gamma path (sequential compute phases). cb_gamma_rm(Wt) one-shot push/consume by tilize. cb_gamma(Wt) persistent." This pre-test analysis caught the cb_normed sizing issue before it could cause a hang.
**Why it worked**: The kernel writer has an explicit CB sync verification step in its workflow that forces it to reason about push/pop balance before running tests.

### 4. Clean compute kernel implementation

**Phase/Agent**: Phase 4 (ttnn-kernel-writer-tdd)
**Evidence**: The final compute kernel (`rms_norm_compute.cpp`, 164 lines) implements all 8 phases with correct helper calls, compile-time routing via `#if IS_INPUT_RM` / `#if HAS_GAMMA`, and proper persistent CB lifecycle (wait at start, pop at end). The CB routing (`compute_input_cb`, `final_tile_cb`, `mul_col_out_cb`) uses constexpr variables determined at compile time, which is clean and efficient.
**Why it worked**: The architect's design document provided exact helper call signatures with explicit CB routing rules for all 4 paths (RM/TILE x gamma/no-gamma). The kernel writer followed these closely.

### 5. Good analysis depth from all three analyzers

**Phase/Agent**: Phase 1 (ttnn-operation-analyzer)
**Evidence**: Each analysis document was 300-500 lines of structured detail. The moreh_norm_w analysis (395 lines) documented exact helper function signatures, the two-phase accumulation pattern (and why RMSNorm can skip it), and persistent CB lifecycle patterns. The tilize analysis documented the critical insight that CB pages are tile-sized even for RM data. The untilize analysis provided a "Simplified Writer Pattern (Interleaved Output)" section with exact pseudocode.
**Why it worked**: The analyzers had clear role-based focus, preventing overlap and ensuring depth in each area.

---

## 3. Issues Found

### Issue 1: cb_xsq page count designed as 1 but required Wt

| Field | Value |
|-------|-------|
| Severity | HIGH |
| Phase / TDD Stage | Phase 4 -- square_reduce_mean |
| Agent | ttnn-operation-architect (design), ttnn-kernel-writer-tdd (debugging) |
| Retries Consumed | 2 hard attempts (attempts 3-4 in stage 2) |
| Time Cost | ~10 minutes debugging (18:03 - 18:12 breadcrumb timestamps) |

**Problem**: The design doc (`op_design.md` CB table, line 80) specified `cb_xsq` (c_25) with 1 page and lifetime "Streaming". The kernel writer initially implemented this faithfully. However, the `square()` helper writes all Wt tiles to cb_xsq before `reduce()` starts consuming them, because both operations run on the same compute thread sequentially. With Wt > 1 (any shape wider than 32), `cb_reserve_back(cb_xsq, 1)` blocks after the first tile because the CB is full and the consumer (reduce) has not started.

The kernel writer's breadcrumb at 18:11:55 diagnosed: "DEADLOCK ROOT CAUSE: cb_xsq has 1 page but square writes Wt tiles sequentially BEFORE reduce starts reading. With Wt>1, square blocks on cb_reserve_back after 1st tile because reduce hasnt consumed anything."

**Root Cause**: The architect marked cb_xsq as "Streaming" (1 page), which would work if square and reduce were on different threads (reader/compute) where the producer and consumer run concurrently. But both square and reduce are compute-thread operations running sequentially. The `square()` helper has `OutputPolicy::OutputPerTile` -- it pushes each output tile immediately. However, it also continues to the next input tile without waiting for the output to be consumed (since on the same thread, nobody is consuming). This means all Wt output tiles must fit in cb_xsq before reduce begins.

The 1x1x32x32 (Wt=1) test case passed because 1 page is sufficient for Wt=1. The hang only manifested on 1x1x64x128 (Wt=4) and larger shapes.

**Fix for agents**:
- **ttnn-operation-architect**: When two compute helpers run sequentially on the same compute thread and one produces into a CB that the other consumes, the CB must hold all output tiles (Wt pages, not 1). Add a validation rule: "If a CB is both produced and consumed entirely within the compute thread (no reader/writer involvement), its page count must be the full batch size, not streaming (1)." The design document should annotate same-thread sequential CBs differently from cross-thread streaming CBs.

### Issue 2: Design doc specified `final_tile_cb = cb_x` for no-gamma RM path, causing in-place deadlock

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 4 -- rms_norm_no_gamma |
| Agent | ttnn-operation-architect (design), ttnn-kernel-writer-tdd (fixed) |
| Retries Consumed | 0 (caught before test run) |
| Time Cost | ~5 minutes (design deviation, no debugging time) |

**Problem**: The design doc (`op_design.md` lines 92-94) specified: "When no gamma and RM: mul_col writes to c_24 (reuse), untilize(c_24, c_16)". This means Phase 6 (`mul<COL>`) reads from `cb_x` (c_24) and also writes back to `cb_x` (c_24). This is an in-place read-modify-write pattern that deadlocks: mul_col's `WaitAndPopPerTile` policy for input A pops tiles from cb_x one at a time, but if mul_col's output also goes to cb_x, after Wt tiles are popped from input and Wt tiles are pushed to output, the CB state may be correct -- but the helper implementation does not support reading and writing the same CB simultaneously.

The kernel writer deviated from the design and used `cb_normed` instead of `cb_x` for the no-gamma RM path output of mul_col. REPORT.md deviation 1 documents this: "Changed to `final_tile_cb = cb_normed` and `untilize(cb_normed, cb_out)` to avoid in-place deadlock."

**Root Cause**: The architect attempted to reuse cb_x to save L1 allocation, but did not validate that reading from and writing to the same CB in a single helper call is safe. The binary op helpers do not support in-place operation.

**Fix for agents**:
- **ttnn-operation-architect**: Add a validation rule: "A binary op helper's input CB(s) and output CB must be distinct. Never specify the same CB as both input and output for a binary op helper call." Document this in the design doc's critical notes section.

### Issue 3: Architect specified non-existent include paths

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) and Phase 4 -- rms_norm_no_gamma |
| Agent | ttnn-operation-architect (design) |
| Retries Consumed | 1 free retry (compilation error in stage 3) + 2 builder compilation fixes |
| Time Cost | ~5 minutes total across builder and kernel writer |

**Problem**: Two include path issues surfaced:
1. The builder's breadcrumb H1: "tensor_accessor.hpp include path wrong -- should be api/tensor/tensor_accessor.h" (builder breadcrumb at 17:36:39). The design doc or system prompt referenced a host-side path that does not exist in device kernel compilation.
2. The kernel writer's stage 3 test failed with: "sfpu_init.h does not exist" (kernel writer breadcrumb at 18:24:44). The `rsqrt.h` include path was also wrong (`compute_kernel_api/` prefix instead of `api/compute/`).

**Root Cause**: The architect (or its reference materials) uses host-side C++ include paths rather than device-side kernel paths. The `sfpu_init.h` header simply does not exist -- the architect hallucinated it as needed for `rsqrt_tile_init()`, but that function comes from `rsqrt.h` alone.

**Fix for agents**:
- **ttnn-operation-architect**: Maintain a validated list of device-side kernel include paths. Before committing the design, grep the actual codebase for the include paths specified. Specifically: `api/compute/eltwise_unary/rsqrt.h` (not `compute_kernel_api/...`), `api/tensor/tensor_accessor.h` (not `ttnn/cpp/.../tensor_accessor.hpp`).

### Issue 4: Architect specified define names that clash with kernel_lib identifiers

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-operation-architect (design) |
| Retries Consumed | 2 builder compilation attempts |
| Time Cost | ~3 minutes |

**Problem**: The design doc specified `Wt` and `fp32_dest_acc_en` as preprocessor defines. Both names collide with local variables/template parameters in kernel_lib headers:
- `#define Wt 1` causes `const uint32_t Wt = input_block_shape.cols;` in `binary_op_helpers.inl` to become `const uint32_t 1 = ...` (builder breadcrumb H2 at 17:38:42).
- `fp32_dest_acc_en` collides with template parameter names in rsqrt header chain (builder breadcrumb at 17:36:39).

The builder renamed them to `RMS_Wt` and `ENABLE_FP32_DEST_ACC`.

**Root Cause**: The architect used short, generic names without checking for collisions with the kernel_lib library headers that would be included.

**Fix for agents**:
- **ttnn-operation-architect**: Always use operation-prefixed define names (e.g., `RMS_NORM_Wt`, `ENABLE_FP32_DEST_ACC`) to avoid collisions with kernel_lib identifiers. Add to design doc checklist: "Verify define names do not match any identifiers in kernel_lib headers (common collisions: `Wt`, `Ht`, `fp32_dest_acc_en`)."

### Issue 5: Architect-generated TDD stage test files had syntax errors

| Field | Value |
|-------|-------|
| Severity | MEDIUM |
| Phase / TDD Stage | Phase 3 (Build) |
| Agent | ttnn-operation-architect (test generation) |
| Retries Consumed | 0 (builder fixed them) |
| Time Cost | ~3 minutes of builder time |

**Problem**: The builder's execution log documents: "Stage test files had syntax errors: `Markup('"TILE_LAYOUT"')`, missing `return` in pytorch_reference, `bfloat16` without quotes" and "Stage 4 test had `pytorch_reference(input_tensorgamma=gamma_t)` (missing comma)". The builder had to fix all 4 stage test files before they could even parse.

**Root Cause**: The test generation in the architect's TDD state definition does not validate Python syntax. String interpolation in test templates introduced quoting and comma errors.

**Fix for agents**:
- **ttnn-operation-architect**: After generating test file content for `.tdd_state.json`, validate that each test file is syntactically valid Python (at minimum, check that `compile(source, filename, 'exec')` does not raise `SyntaxError`).

### Issue 6: Stage 2 CB deadlock was misdiagnosed initially

| Field | Value |
|-------|-------|
| Severity | LOW |
| Phase / TDD Stage | Phase 4 -- square_reduce_mean |
| Agent | ttnn-kernel-writer-tdd |
| Retries Consumed | 2 hard attempts (attempts 2-3 wasted before correct diagnosis) |
| Time Cost | ~10 minutes (17:58 - 18:09, between first hang and correct hypothesis at 18:09) |

**Problem**: The first hang (attempt 2) was diagnosed with MEDIUM confidence as "compute_kernel_hw_startup needs correct CB args" (hypothesis H2 at 17:58:28). This was partially correct -- hw_startup was wrong -- but the fix did not resolve the core deadlock issue. The second hang (attempt 3-4) continued with the same symptom. Only on the third hypothesis (H3 at 18:03:18) did the kernel writer identify the scaler lifetime issue, and then separately in the hypothesis breadcrumbs (18:11:55) identified the cb_xsq page count as the actual root cause.

The progression: attempt 2 fixed hw_startup (real issue, but not the hang cause) -> attempt 3 removed manual scaler CB ops (partial fix) -> attempt 4 realized cb_xsq page count was the true deadlock cause.

**Root Cause**: The triage log from the hang provided limited information ("TRISC1=W" without specifying which CB or function). The kernel writer had to reason from first principles, which took multiple iterations. The `1x1x32x32 TILE PASSED, 1x1x64x128 TILE HUNG on 2nd row` pattern (breadcrumb at 18:03:10) was the critical clue that led to the correct diagnosis.

**Fix for agents**:
- **ttnn-kernel-writer-tdd**: When a hang occurs and the smallest shape (Wt=1) passes but a larger shape (Wt>1) fails, immediately suspect CB page count issues for compute-internal CBs. Add this as a diagnostic heuristic: "If Wt=1 passes but Wt>1 hangs, check that all compute-internal CBs (produced and consumed on the same thread) have Wt pages, not 1."

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| data_pipeline | ~4m (17:45 - 17:49) | 0 free, 0 hard | PASS | Clean -- 4 upstream fixes applied proactively |
| square_reduce_mean | ~25m (17:50 - 18:14) | 0 free, 4 hard | PASS | CB deadlock debugging (cb_xsq page count) |
| rms_norm_no_gamma | ~27m (18:15 - 18:41) | 1 free, 0 hard | PASS | 1 free retry (bad include), rest clean. Long test runtime (~15s per run x 8 tests) |
| rms_norm_with_gamma | ~8m (18:41 - 18:49) | 0 free, 0 hard | PASS | Clean -- proactive cb_normed fix |

### Time Sinks

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | Stage 2 CB deadlock | ttnn-kernel-writer-tdd | ~25m | 24% | 4 hard attempts on square_reduce_mean, dominated by 2 hangs requiring device reset + re-diagnosis | 4 | Architect specified cb_xsq as 1-page "Streaming" when Wt pages were needed for same-thread sequential ops |
| 2 | Stage 3 long duration | ttnn-kernel-writer-tdd | ~27m | 26% | Only 1 free retry, but wall clock was long; likely includes time implementing 4 new compute phases (add_eps, rsqrt, 2nd tilize, mul_col) plus upstream program descriptor fixes | 1 | Legitimate complexity -- 4 new compute phases added in a single stage |
| 3 | Builder compilation fixes | ttnn-generic-op-builder | ~5m | 5% | 3 compilation errors from bad include paths and define name collisions | 2 | Architect-originated issues (include paths, define names) |

### Wasted Work

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| ttnn-kernel-writer-tdd | Attempt 2 fix: removed copy_tile path for TILE output, changed hw_startup | Fix was partially correct (hw_startup was wrong) but did not address the deadlock root cause (cb_xsq page count). Time spent testing this partial fix was wasted. | Better hang diagnosis heuristics: when Wt=1 passes but Wt>1 fails, skip non-CB-related hypotheses |
| ttnn-kernel-writer-tdd | Attempt 3 fix: removed manual scaler CB ops to let reduce helper manage scaler | This was also a real issue (scaler lifecycle) but was not the hang root cause either. Two separate bugs masked each other. | When multiple bugs exist, the agent should fix the most likely one and re-test, rather than combining fixes |
| ttnn-generic-op-builder | First two test runs failed due to include paths and define clashes | The builder had to discover issues that the architect should have prevented | Architect validation of include paths and define names against kernel_lib |

---

## 5. Inter-Agent Communication Issues

### Handoff 1: ttnn-operation-analyzer (x3) -> ttnn-operation-architect

| Field | Value |
|-------|-------|
| Artifact Passed | `tilize_analysis.md`, `untilize_analysis.md`, `moreh_norm_w_analysis.md` |
| Quality | GOOD |
| Issues | None significant. All three analyses were focused on their assigned roles and provided actionable information. |
| Downstream Impact | The architect used all three analyses effectively. Helper function signatures, CB patterns, and TensorAccessor usage from the analyses appear directly in the design doc. |
| Suggestion | No changes needed. The role-based focus (input_stage, output_stage, compute_core) worked well. |

### Handoff 2: ttnn-operation-architect -> ttnn-generic-op-builder

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md`, `.tdd_state.json` |
| Quality | ADEQUATE |
| Issues | (1) TDD stage test files in `.tdd_state.json` had Python syntax errors. (2) Specified include paths (`tensor_accessor.hpp`, `sfpu_init.h`) that do not exist for device kernels. (3) Define names (`Wt`, `fp32_dest_acc_en`) clashed with kernel_lib identifiers. |
| Downstream Impact | Builder spent ~5 minutes fixing 4 test files and ~5 minutes fixing compilation errors (3 attempts total). These were all straightforward fixes, but they are avoidable. |
| Suggestion | Architect should validate test file syntax and use prefixed define names. Include paths should be verified against the actual codebase. |

### Handoff 3: ttnn-generic-op-builder -> ttnn-kernel-writer-tdd

| Field | Value |
|-------|-------|
| Artifact Passed | Stub kernels, program descriptor, test files, execution log handoff notes |
| Quality | GOOD |
| Issues | The builder's handoff notes were clear and detailed (section 6 of execution log), specifically calling out renamed defines (`RMS_Wt` not `Wt`, `ENABLE_FP32_DEST_ACC`), runtime arg layout, and CT arg offsets. |
| Downstream Impact | The kernel writer successfully used the renamed defines without confusion. The handoff notes prevented what would have been additional compilation errors. |
| Suggestion | The builder's deviation documentation (renamed defines) was excellent. This pattern should be standard. |

### Handoff 4: ttnn-operation-architect -> ttnn-kernel-writer-tdd (design doc)

| Field | Value |
|-------|-------|
| Artifact Passed | `op_design.md` Part 2 (Kernel Implementation) |
| Quality | ADEQUATE |
| Issues | (1) cb_xsq specified as 1 page "Streaming" -- caused 2 hangs. (2) `final_tile_cb = cb_x` for no-gamma RM path was an in-place deadlock pattern. (3) cb_normed specified as 1 page for gamma path -- the kernel writer caught this proactively. |
| Downstream Impact | The cb_xsq issue consumed 4 hard attempts and ~25 minutes. The cb_x in-place issue was caught without wasting attempts. The cb_normed issue was caught proactively by the kernel writer's CB sync check. |
| Suggestion | The design doc needs a validation pass specifically for same-thread sequential operations. Any CB that is both produced and consumed within the compute thread must have full-batch (Wt) page counts, never streaming (1). |

---

## 6. Upstream Feedback Synthesis

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| ttnn-operation-architect | ttnn-generic-op-builder | Use prefixed define names (e.g., `RMS_Wt` not `Wt`) to avoid collisions with kernel_lib identifiers | HIGH | HIGH |
| ttnn-operation-architect | ttnn-generic-op-builder | Use device-side include paths (`api/tensor/tensor_accessor.h` not `ttnn/cpp/.../tensor_accessor.hpp`) | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Validate that same-thread sequential CB pairs have full-batch page counts, not streaming (1) | HIGH | HIGH |
| ttnn-operation-architect | ttnn-kernel-writer-tdd | Never specify the same CB as both input and output for a binary op helper | HIGH | MEDIUM |
| ttnn-operation-architect | ttnn-generic-op-builder | Validate generated test files are syntactically valid Python | HIGH | MEDIUM |
| ttnn-kernel-writer-tdd | self-reflection | Add diagnostic heuristic: if Wt=1 passes but Wt>1 hangs, suspect compute-internal CB page counts | MEDIUM | MEDIUM |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| Design | cb_xsq page count error caused 60% of all failures (3 of 5) | Add architect validation rule: same-thread sequential CBs must have full-batch page counts | HIGH |
| Design | In-place CB usage (read and write same CB in one helper) attempted | Add architect checklist item: binary op helpers require distinct input/output CBs | MEDIUM |
| Build | 3 compilation errors from architect-originated issues | Add a lightweight pre-build validation for include paths and define collisions | MEDIUM |
| TDD | Stage 2 had 4 hard attempts while stages 1, 3, 4 had 0 | Stage 2 was the first to use compute helpers (square, reduce). The first "real compute" stage tends to surface all infrastructure issues. Consider a lighter intermediate stage. | LOW |

---

## 7. Comparison with Known Issues

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| 1 | Kernel writer burns context on numerical debugging | NO | No numerical mismatches in this run. All failures were structural (hangs, compile errors). This is a positive signal. |
| 2 | Long leash (planner/designer gap) | NO (DONE) | Already merged into single Architect agent. This run used the merged design. |
| 3 | `.tdd_state.json` fragility | NO | No schema issues observed. Builder successfully read the architect's output. |
| 4 | No fast path for simple operations | NO | RMS norm is not a simple operation (8 compute phases). Not applicable. |
| 6 | Builder model choice | PARTIALLY | Builder needed 3 attempts to get compilation right, but the errors originated from the architect's design (wrong include paths, bad define names), not from the builder itself. |
| 7 | Discovery keyword matching | NO | References were appropriate. No evidence of missed or wrong references. |
| 9 | No architect/builder cross-validation | YES | The cb_xsq page count (1 vs Wt) was wrong in the design, carried through the builder into the program descriptor, and only caught at runtime in Phase 4. A static cross-check could have caught this. |
| 11 | No incremental re-run | NO | Pipeline completed successfully without needing to resume. |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| Same-thread sequential CB page count validation | The architect specified "Streaming" (1 page) for cb_xsq, but since both producer (square) and consumer (reduce) run on the same compute thread, the CB needs Wt pages. This is a general pattern that affects any operation with sequential compute helpers sharing an intermediate CB. | HIGH |
| In-place CB deadlock pattern not detected | The architect specified read-from and write-to the same CB (cb_x) for a binary op helper. This is always a deadlock risk and should be flagged at design time. | MEDIUM |
| Non-existent include path hallucination | The architect specified `sfpu_init.h` which does not exist in the codebase. Include paths should be validated against the actual file system. | MEDIUM |

---

## 8. Actionable Recommendations

### Recommendation 1: Add same-thread CB page count validation to architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a validation rule in the CB allocation section: "When two compute helpers run sequentially on the same compute thread (no reader/writer boundary between them) and one produces into a CB that the other consumes, that CB MUST have a page count equal to the full batch size (typically Wt for row operations). Mark these CBs as 'Same-Thread Block' in the CB table, not 'Streaming'. Streaming (1-page) CBs are ONLY valid when producer and consumer run on different threads (reader-compute, compute-writer)."
- **Expected Benefit**: Would have prevented 3 of 5 failures in this run (60% of failures). Saves ~25 minutes of debugging time.
- **Priority**: HIGH
- **Effort**: SMALL

### Recommendation 2: Add in-place CB validation to architect

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a validation rule in the binary op verification section: "A binary op helper's output CB must be distinct from both input CBs. If you need to reuse a CB slot to save L1, introduce a dedicated intermediate CB instead. In-place operation (same CB as input and output) is NOT supported by the binary op helpers."
- **Expected Benefit**: Would have prevented the `final_tile_cb = cb_x` design error that the kernel writer had to deviate from.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 3: Validate device-side include paths at design time

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a reference table of validated device-side kernel include paths to the architect's system prompt. At minimum: `api/compute/compute_kernel_hw_startup.h`, `api/compute/eltwise_unary/rsqrt.h`, `api/dataflow/dataflow_api.h`, `api/tensor/tensor_accessor.h`, and all kernel_lib headers (`ttnn/cpp/ttnn/kernel_lib/*.hpp`). Instruct the architect to ONLY use paths from this validated list.
- **Expected Benefit**: Would have prevented 1 free retry and 2 builder compilation attempts. Saves ~5 minutes.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 4: Use operation-prefixed define names

- **Type**: instruction_change
- **Target**: ttnn-operation-architect prompt/instructions
- **Change**: Add a rule: "All preprocessor defines passed to kernels must use an operation-specific prefix to avoid collisions with kernel_lib identifiers. Use `{OP_NAME}_Wt` (not `Wt`), `ENABLE_FP32_DEST_ACC` (not `fp32_dest_acc_en`). Known collision-prone names: `Wt`, `Ht`, `fp32_dest_acc_en`, `NUM_TILES`, `BLOCK_SIZE`."
- **Expected Benefit**: Would have prevented 2 builder compilation attempts. Saves ~3 minutes.
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 5: Add hang diagnosis heuristic for Wt-dependent failures

- **Type**: instruction_change
- **Target**: ttnn-kernel-writer-tdd prompt/instructions
- **Change**: Add a diagnostic heuristic: "When a hang occurs, check which test cases pass and which hang. If the smallest shape (typically Wt=1) passes but larger shapes hang, the issue is almost certainly a CB page count problem for compute-internal CBs. Check all CBs that are both produced and consumed on the compute thread -- they must have Wt pages, not 1."
- **Expected Benefit**: Would have reduced Stage 2 debugging from 4 attempts to 2 attempts (~10 minutes saved).
- **Priority**: MEDIUM
- **Effort**: SMALL

### Recommendation 6: Validate TDD stage test syntax at architect time

- **Type**: new_validation
- **Target**: ttnn-operation-architect test generation code
- **Change**: After generating test file content in `.tdd_state.json`, validate each test body is syntactically valid Python. Check for common errors: unquoted identifiers, missing commas between function arguments, missing return statements. The architect should run `ast.parse(test_source)` mentally before committing.
- **Expected Benefit**: Would have prevented builder from spending ~3 minutes fixing 4 test files.
- **Priority**: LOW
- **Effort**: SMALL

---

## 9. Overall Assessment

### Pipeline Maturity Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | 5/5 | Three references with differentiated roles were ideal for this operation |
| Analysis quality | 5/5 | Thorough, focused analyses with explicit "relevance to RMS norm" sections |
| Design completeness | 3/5 | Good overall structure but 3 CB-related errors (cb_xsq pages, cb_x in-place, cb_normed pages) and bad include paths |
| Build correctness | 4/5 | Builder produced correct infrastructure after fixing architect-originated issues. Handoff notes were excellent. |
| Kernel implementation | 4/5 | All 4 stages passed. Stage 2 debugging was painful but ultimately resolved. Proactive CB sync checks in stages 3-4 were effective. |
| Inter-agent communication | 4/5 | Analyzer-to-architect was excellent. Architect-to-builder had include/define issues. Builder-to-writer handoff notes were strong. |
| Logging/observability | 4/5 | Breadcrumbs were detailed enough to reconstruct the full timeline. CB sync checks were informative. Minor gap: some breadcrumb files (stage_start, hypothesis) were sparse compared to the main breadcrumb JSONL. |

### Top 3 Things to Fix

1. **Same-thread sequential CB page count validation**: The architect must recognize when producer and consumer helpers run on the same compute thread and size intermediate CBs accordingly (Wt, not 1). This single issue caused 60% of all failures.
2. **Device-side include path validation**: The architect should use a validated list of kernel include paths rather than guessing from host-side paths or hallucinating headers. This caused failures in both the builder and kernel writer phases.
3. **In-place CB usage detection**: The architect should never specify the same CB as input and output for a binary op helper. Add this as a hard constraint in the CB routing logic.

### What Worked Best

The **three-reference hybrid analysis** was the single strongest aspect of this pipeline run. Having separate, role-focused analyses for input (tilize), output (untilize), and compute (moreh_norm_w) gave the architect exactly the right information to design an 8-phase compute pipeline with correct helper calls. The architect's design was fundamentally sound -- the issues were all in CB sizing and include paths, not in the algorithmic approach. This shows that the analysis-to-design pipeline is working well when references are well-chosen.
