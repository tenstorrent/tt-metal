# ttnn-kernel-writer Execution Log

**Agent**: ttnn-kernel-writer
**Operation**: row_mean_sub_square_reduce
**Predecessor**: ttnn-kernel-designer
**Input Document**: kernel_design.md

## Summary

Implemented the kernels for the row_mean_sub_square_reduce operation following the design document. All 12 Stage 7 correctness tests pass.

## Artifacts Produced

| Artifact | Path |
|----------|------|
| Compute Kernel | `device/kernels/compute/row_mean_sub_square_reduce_compute.cpp` |
| Reader Kernel | `device/kernels/dataflow/reader_row_mean_sub_square_reduce.cpp` |
| Writer Kernel | `device/kernels/dataflow/writer_row_mean_sub_square_reduce.cpp` |
| Stage 7 Tests | `test_dev/test_stage7_kernel_correctness.py` |
| Breadcrumbs | `agent_logs/ttnn-kernel-writer_breadcrumbs.jsonl` |

## Design Compliance

### Phase 1: Tilize
- **Directive**: USE HELPER: compute_kernel_lib::tilize()
- **Implementation**: `compute_kernel_lib::tilize(cb_rm_in, Wt, cb_tilized, 1)`
- **Compliant**: YES
- **Redundant CB Ops**: None

### Phase 2: Reduce Mean
- **Directive**: USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>()
- **Implementation**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(...)`
- **Compliant**: YES
- **Redundant CB Ops**: None
- **Note**: PERSISTENT mode keeps tiles in cb_tilized for Phase 3

### Phase 3: Subtract + Square
- **Directive**: NO HELPER - raw calls
- **Implementation**: `sub_tiles_bcast_cols()` + `square_tile()`
- **Compliant**: YES (with correction noted below)
- **Deviation**: Design doc specified `sub_tiles_bcast_scalar` but this broadcasts only the [0,0] element. Changed to `sub_tiles_bcast_cols` to correctly broadcast column 0 (per-row means) to all columns.

### Phase 4: Reduce Variance
- **Directive**: USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW>()
- **Implementation**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(...)`
- **Compliant**: YES
- **Redundant CB Ops**: None

### Phase 5: Untilize
- **Directive**: USE HELPER: compute_kernel_lib::untilize<1, cb_out_tiled, cb_rm_out>()
- **Implementation**: `compute_kernel_lib::untilize<1, cb_out_tiled, cb_rm_out>(1)`
- **Compliant**: YES
- **Redundant CB Ops**: None

## Redundant CB Operation Check

All helper calls are clean - no redundant cb_wait_front/cb_pop_front/cb_reserve_back/cb_push_back operations wrapped around helpers.

## Correctness Test Results

| Test | Status |
|------|--------|
| test_basic_correctness_single_tile | PASSED |
| test_basic_correctness_multi_tile_width | PASSED |
| test_basic_correctness_multi_tile_row | PASSED |
| test_uniform_values_zero_variance | PASSED |
| test_known_variance | PASSED |
| test_batched_input | PASSED |
| test_multi_channel_input | PASSED |
| test_full_batch_channel | PASSED |
| test_large_width | PASSED |
| test_large_height | PASSED |
| test_numerical_stability_large_values | PASSED |
| test_numerical_stability_small_values | PASSED |

**Total**: 12/12 PASSED (9.16s)

## Key Implementation Notes

1. **Broadcast Type Fix**: The critical fix was changing from `sub_tiles_bcast_scalar` (broadcasts single value [0,0]) to `sub_tiles_bcast_cols` (broadcasts column 0 to all columns). This is required because the mean tile contains 32 different means (one per row in column 0), and we need each row's elements to be subtracted by that row's mean.

2. **SFPU Square**: Used `square_tile()` from compute_kernel_api.h to square values in DST register in-place. This is more efficient than pack+reload+mul_tiles.

3. **PERSISTENT Mode**: The reduce helper's PERSISTENT mode waits for all tiles but does NOT pop them, allowing Phase 3 to access the tilized data directly via indexed CB access.

4. **Helper Library Usage**: Phases 1, 2, 4, 5 use kernel_lib helpers which encapsulate all CB and DST register management. No manual CB operations are added around these calls.

## Host Files Modified

None - only kernel files were modified which are compiled at runtime.

## Commit

```
c41a0368da [ttnn-kernel-writer] stage 7: implement row_mean_sub_square_reduce kernels
```
