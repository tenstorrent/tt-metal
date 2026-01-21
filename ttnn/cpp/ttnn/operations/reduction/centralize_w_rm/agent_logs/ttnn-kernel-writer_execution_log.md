# Agent Execution Log: ttnn-kernel-writer

## Metadata
| Field | Value |
|-------|-------|
| Operation | `centralize_w_rm` |
| Agent | `ttnn-kernel-writer` |
| Stages | 7 |
| Input | `kernel_design.md` |
| Predecessor | ttnn-kernel-designer |
| Final Status | SUCCESS |
| Total Attempts | 3 (for Stage 7 tests) |

---

## 1. Input Interpretation

### Spec/Input Fields Extracted

| Field | Value | Confidence | Notes |
|-------|-------|------------|-------|
| Compute phases | Tilize -> Reduce -> BcastSub -> Untilize | HIGH | Explicitly stated in design doc |
| CB IDs | c_0 (input RM), c_1 (tiled), c_2 (scaler), c_3 (mean), c_4 (centralized), c_16 (output RM) | HIGH | Explicitly stated |
| Reduce mode | PERSISTENT (keep tiles for bcast_sub) | HIGH | Explicitly stated |
| BcastSub broadcast dim | COL (REDUCE_ROW produces column-shaped output) | HIGH | Explicitly stated |
| Input B policy for bcast_sub | "Streaming" described as "wait upfront, pop after" | MEDIUM | Design doc says "Streaming" but describes WaitUpfront/PopAtEnd behavior |

**Confidence Levels**:
- **HIGH**: Explicitly stated in input, no interpretation needed
- **MEDIUM**: Required some inference or combining multiple sources

### Interpretation Issues

The design document specified `cb_policies::Streaming` for input B of the bcast_sub phase, describing it as "wait upfront, pop after". However, `cb_policies::Streaming` actually means "wait per tile, pop per tile". This mismatch caused a CB deadlock that required debugging. The correct policy for this use case is `InputPolicy<WaitUpfront, PopAtEnd>`.

### Upstream Feedback

| Target Agent | Issue | Suggestion | Severity |
|--------------|-------|------------|----------|
| ttnn-kernel-designer | Design doc said "Streaming" for input B but described behavior inconsistent with Streaming policy | Clarify that for COL broadcast with a single B tile, use `InputPolicy<WaitUpfront, PopAtEnd>` explicitly, not "Streaming" | MEDIUM |
| ttnn-kernel-designer | Design doc said "Preloaded" for input A but `cb_policies::Preloaded` has pops_caller_managed=true (no pop) | Clarify that to pop tiles at end, need `InputPolicy<WaitCallerManaged, PopAtEnd>`, not predefined Preloaded | MEDIUM |

---

## 2. Execution Timeline

### Phase: Kernel Implementation

#### Attempt 1: Initial implementation following design doc
| Field | Value |
|-------|-------|
| Action | Implemented all 3 kernels following design document |
| Expected | Tests pass |
| Actual | First test passed, second test (64x64) hung |
| Result | FAIL |

- **Error Type**: test_timeout (hang)
- **Error Summary**: Test hung on second test case (multi-tile height)
- **Root Cause Hypothesis**: H1: Preloaded policy has pops_caller_managed=true which means NO pop, causing CB deadlock
- **Evidence**: Looking at cb_policies.hpp, Preloaded = InputPolicy<WaitCallerManaged, PopCallerManaged>, which doesn't pop
- **Recovery Action**: Changed to InputPolicy<WaitCallerManaged, PopAtEnd>

#### Attempt 2: Fixed input A policy
| Field | Value |
|-------|-------|
| Action | Changed input A policy for bcast_sub to InputPolicy<WaitCallerManaged, PopAtEnd> |
| Expected | Tests pass |
| Actual | Still hanging on second test |
| Result | FAIL |

- **Error Type**: test_timeout (hang)
- **Error Summary**: Still hanging after fixing input A policy
- **Root Cause Hypothesis**: H2: Streaming policy on input B for COL broadcast waits/pops per tile but we only have 1 B tile
- **Evidence**: binary_op_helpers.hpp shows per-tile wait/pop for COL broadcast with Streaming policy
- **Recovery Action**: Changed input B policy to InputPolicy<WaitUpfront, PopAtEnd>

#### Attempt 3: Fixed both input policies
| Field | Value |
|-------|-------|
| Action | Changed input B policy to InputPolicy<WaitUpfront, PopAtEnd> |
| Expected | All 7 tests pass |
| Actual | All 7 tests passed |
| Result | PASS |

---

## 2a. Design Document Compliance

### Helper Usage Compliance

| Phase | Design Directive | Your Implementation | Compliant? |
|-------|------------------|---------------------|------------|
| Tilize | USE HELPER | `compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1)` | YES |
| Reduce | USE HELPER with PERSISTENT | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>(...)` | YES |
| BcastSub | USE HELPER with policies | `compute_kernel_lib::sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(...)` | YES |
| Untilize | USE HELPER | `compute_kernel_lib::untilize<Wt, cb_centralized_tiled, cb_out_rm>(1)` | YES |

### Redundant CB Operation Check

| Helper Used | Wrapper CB Ops Present? | Status |
|-------------|-------------------------|--------|
| compute_kernel_lib::tilize() | NO | CLEAN |
| compute_kernel_lib::reduce<...>() | NO | CLEAN |
| compute_kernel_lib::sub<...>() | NO | CLEAN |
| compute_kernel_lib::untilize<...>() | NO | CLEAN |

### Stage 7 Correctness Test Results

| Test Case | Input Shape | Tolerance | Result | Notes |
|-----------|-------------|-----------|--------|-------|
| test_basic_correctness_32x64 | [1,1,32,64] | rtol=1e-2, atol=1e-2 | PASS | 1 tile height, 2 tile widths |
| test_multi_tile_height_64x64 | [1,1,64,64] | rtol=1e-2, atol=1e-2 | PASS | 2 tile heights, 2 tile widths |
| test_larger_width_32x128 | [1,1,32,128] | rtol=1e-2, atol=1e-2 | PASS | 1 tile height, 4 tile widths |
| test_square_64x64 | [1,1,64,64] | rtol=1e-2, atol=1e-2 | PASS | Square tensor |
| test_uniform_values | [1,1,32,64] | rtol=1e-2, atol=1e-2 | PASS | All elements same, output should be zeros |
| test_zeros | [1,1,32,64] | rtol=1e-2, atol=1e-2 | PASS | Input zeros, output zeros |
| test_row_means_are_zero | [1,1,64,128] | rtol=1e-2, atol=1e-2 | PASS | Verifies row means of output are zero |

### Numerical Debugging

No numerical issues encountered. The bugs were CB synchronization issues, not numerical.

### Host Files Modified

| File | Build Required | Build Ran | Build Result |
|------|----------------|-----------|--------------|
| (kernel files only) | NO | NO | N/A |

---

## 3. Recovery Summary

### Error Recovery Table

| # | Stage | Error Type | Root Cause (Hypothesis) | Recovery Action | Resolved? |
|---|-------|------------|-------------------------|-----------------|-----------|
| 1 | 7 | test_timeout | H1: Preloaded policy has pops_caller_managed=true (no pop) | Changed to InputPolicy<WaitCallerManaged, PopAtEnd> | PARTIAL |
| 2 | 7 | test_timeout | H2: Streaming policy on B waits/pops per tile but only 1 B tile | Changed to InputPolicy<WaitUpfront, PopAtEnd> | YES |

### Attempts Per Stage

| Stage | Attempts | Final Result |
|-------|----------|--------------|
| Stage 7 | 3 | PASS |

### Unresolved Issues

All issues were resolved.

---

## 4. Deviations from Instructions

| What | Why | Impact |
|------|-----|--------|
| Used custom InputPolicy instead of predefined Preloaded | Preloaded has pops_caller_managed=true which doesn't pop tiles | Required to fix CB deadlock |
| Used InputPolicy<WaitUpfront, PopAtEnd> instead of Streaming for input B | Streaming waits/pops per tile but COL broadcast with 1 B tile needs upfront wait | Required to fix CB deadlock |

Note: These deviations were necessary to achieve correct CB synchronization. The design document's description of the policies was misleading (said "Streaming" but described WaitUpfront/PopAtEnd behavior).

---

## 5. Artifacts

### Files Created

| Path | Purpose |
|------|---------|
| `test_dev/test_stage7_kernel_correctness.py` | Stage 7 correctness tests (7 test cases) |

### Files Modified

| Path | Changes |
|------|---------|
| `device/kernels/dataflow/reader_centralize_w_rm.cpp` | Implemented scaler generation and tile-row data reading |
| `device/kernels/compute/centralize_w_rm_compute.cpp` | Implemented 4-phase compute pipeline with helpers |
| `device/kernels/dataflow/writer_centralize_w_rm.cpp` | Implemented full-width output writing |

---

## 6. Handoff Notes

### For Next Agent: N/A

N/A - This is the final stage. Operation is complete.

**Key Configuration**:
- All 4 compute phases use kernel helpers (no raw CB operations needed)
- Custom CB policies required for bcast_sub phase due to PERSISTENT mode interaction

**Special Considerations**:
- The PERSISTENT reduce mode keeps tiles in c_1, requiring careful policy selection for bcast_sub
- For COL broadcast with single B tile, do NOT use Streaming policy - use WaitUpfront/PopAtEnd

**Known Limitations**:
- Single-core implementation only (no multi-core support yet)
- No batching support (N=C=1)

---

## 7. Instruction Improvement Recommendations

### Recommendation 1: Document policy combinations for common patterns
- **Observed**: Design doc said "Preloaded" and "Streaming" but these didn't work correctly
- **Frequency**: First time encountering this pattern
- **Current Instruction**: Instructions reference predefined policies (Streaming, Preloaded, etc.)
- **Suggested Change**: Add a table showing which InputPolicy combinations to use for common patterns like "tiles already present, pop at end"
- **Rationale**: Would have avoided 2 debugging iterations
- **Confidence**: HIGH

### Recommendation 2: Add CB sync verification checklist
- **Observed**: CB deadlocks are subtle and hard to debug
- **Frequency**: Every time
- **Current Instruction**: Instructions mention to verify CB sync but don't provide checklist
- **Suggested Change**: Add checklist: "For each CB: Does push count equal pop count? Does helper handle CB ops internally?"
- **Rationale**: Systematic verification would catch issues earlier
- **Confidence**: MEDIUM

---

## 8. Raw Logs

<details>
<summary>Test Output (Final Successful Run)</summary>

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-8.4.2, pluggy-1.6.0
collected 7 items

test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_basic_correctness_32x64 PASSED
test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_multi_tile_height_64x64 PASSED
test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_larger_width_32x128 PASSED
test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_square_64x64 PASSED
test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_uniform_values PASSED
test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_zeros PASSED
test_stage7_kernel_correctness.py::TestCentralizeWRmCorrectness::test_row_means_are_zero PASSED
============================== 7 passed in 5.00s ===============================
```

</details>

<details>
<summary>Git Commit</summary>

```
commit 4b98271be7
[ttnn-kernel-writer] stage 7: implement centralize_w_rm kernels

- Implemented reader kernel with scaler generation (1/W) and tile-row data reading
- Implemented compute kernel with 4-phase pipeline:
  1. Tilize (tilize_helpers.hpp)
  2. Reduce with PERSISTENT mode (reduce_helpers.hpp)
  3. BcastSub with custom policies for CB persistence handling
  4. Untilize (untilize_helpers.hpp)
- Implemented writer kernel with full-width output (no reduction)
- Created Stage 7 correctness tests (7 tests, all passing)
- Fixed CB deadlock by using custom InputPolicy<WaitCallerManaged, PopAtEnd> for input A
  and InputPolicy<WaitUpfront, PopAtEnd> for input B in the bcast_sub phase

operation: centralize_w_rm
build: SKIPPED (kernel-only changes)
tests: stage7 7/7 PASSED
```

</details>
