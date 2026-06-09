# Reduce Helpers Library Plan

## Goal

Create a simplified helper library for reduce operations that hides the complexity of:
- `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` DST register management
- `reduce_init` / `reduce_uninit` initialization
- Circular buffer manipulation (`cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`)
- `pack_tile` for writing results to output CB

The API should mirror the pattern established in `tilize_helpers.h` and `untilize_helpers.h`.

## DST Register API

The library uses the **new (non-deprecated)** DST register API:

```cpp
tile_regs_acquire();  // MATH thread acquires DST
// ... math operations ...
tile_regs_commit();   // MATH thread releases DST
tile_regs_wait();     // PACK thread waits for MATH
// ... pack operations ...
tile_regs_release();  // PACK thread releases DST
```

Note: `acquire_dst()` and `release_dst()` are deprecated and will NOT be used.

## Current Reduce Usage Analysis

### reduce_hw.cpp (REDUCE_SCALAR - reduces both H and W)
```cpp
compute_kernel_hw_startup(c_0, c_2, c_3);
reduce_init(c_0, c_2, c_3);
cb_wait_front(c_2, 1);  // scaler tile
for (nc = 0; nc < NC; nc++) {
    tile_regs_acquire();
    for (ht = 0; ht < Ht; ++ht) {
        for (wt = 0; wt < Wt; ++wt) {
            cb_wait_front(c_0, 1);
            reduce_tile(c_0, c_2, 0, 0, reduce_dst_idx);
            cb_pop_front(c_0, 1);
        }
    }
    cb_reserve_back(c_3, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(reduce_dst_idx, c_3);
    tile_regs_release();
    cb_push_back(c_3, 1);
}
```
**Output: 1 tile per batch (NC)**

### reduce_w.cpp (REDUCE_ROW - reduces W dimension)
```cpp
compute_kernel_hw_startup(c_0, c_2, c_3);
reduce_init(c_0, c_2, c_3);
cb_wait_front(c_2, 1);  // scaler tile
for (nc = 0; nc < NC; nc++) {
    for (ht = 0; ht < Ht; ++ht) {
        tile_regs_acquire();
        for (wt = 0; wt < Wt; ++wt) {
            cb_wait_front(c_0, 1);
            reduce_tile(c_0, c_2, 0, 0, reduce_dst_idx);
            cb_pop_front(c_0, 1);
        }
        cb_reserve_back(c_3, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(reduce_dst_idx, c_3);
        tile_regs_release();
        cb_push_back(c_3, 1);
    }
}
```
**Output: Ht tiles per batch (NC)**

### reduce_h.cpp (REDUCE_COL - reduces H dimension)
```cpp
compute_kernel_hw_startup(c_0, c_2, c_3);
reduce_init(c_0, c_2, c_3);
cb_wait_front(c_2, 1);  // scaler tile
for (nc = 0; nc < NC; nc++) {
    for (wt = 0; wt < Wt; wt += row_chunk) {
        tile_regs_acquire();
        for (ht = 0; ht < Ht; ++ht) {
            for (i = wt; i < chunk_end; ++i) {
                cb_wait_front(c_0, 1);
                reduce_tile(c_0, c_2, 0, 0, reduce_dst_idx++);
                cb_pop_front(c_0, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (i = wt; i < chunk_end; ++i) {
            cb_reserve_back(c_3, 1);
            pack_tile(i - wt, c_3);
            cb_push_back(c_3, 1);
        }
        tile_regs_release();
    }
}
```
**Output: Wt tiles per batch (NC)**

## Proposed API Design

### File Location
`ttnn/cpp/ttnn/kernel_lib/reduce_helpers.h`

### Namespace
`compute_kernel_lib`

### Design Principles
1. **Single unified function** - One `reduce` function handles all patterns
2. **Template-based compile-time optimization** - Use templates for reduce_type, reduce_dim, init/uninit flags
3. **Zero runtime overhead** - All functions inlined (ALWI)
4. **Simple parameters** - User provides CBs and dimensions, library handles the rest
5. **Modern DST API** - Uses `tile_regs_*` functions (not deprecated `acquire_dst/release_dst`)

### Proposed API

```cpp
namespace compute_kernel_lib {

/**
 * @brief Unified reduce function handling all reduction patterns
 *
 * This single function handles:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() at the start of your kernel.
 *
 * IMPORTANT - SCALER CB REQUIREMENT:
 * The scaler CB (icb_scaler) must contain the scaling factor tile BEFORE calling
 * this function. The function will wait for it automatically.
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX) - defaults to REDUCE_OP define
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR) - defaults to REDUCE_DIM define
 * @tparam init If true, calls reduce_init before processing (default: true)
 * @tparam uninit If true, calls reduce_uninit after processing (default: true)
 * @tparam enforce_fp32_accumulation Enable FP32 accumulation (default: false)
 *
 * @param icb Input circular buffer containing tiles to reduce
 * @param icb_scaler Circular buffer containing scaler tile
 * @param ocb Output circular buffer for reduced tiles
 * @param Ht Height in tiles (number of tile rows)
 * @param Wt Width in tiles (number of tile columns)
 * @param num_batches Number of batches to process (NC dimension)
 *
 * @example
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Using defines for reduce type/dim
 *   compute_kernel_lib::reduce(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 */
template <
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    bool init = true,
    bool uninit = true,
    bool enforce_fp32_accumulation = false>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_batches);

} // namespace compute_kernel_lib
```

## Implementation Strategy

### Key Implementation Details

1. **DST Register Management**
   - Auto-detect DEST capacity from JIT headers (like untilize_helpers.h)
   - For REDUCE_COL, chunk Wt dimension to fit within DEST limit
   - For REDUCE_ROW and REDUCE_SCALAR, single DST index is sufficient
   - Use modern `tile_regs_acquire/commit/wait/release` API

2. **CB Handling**
   - Wait for scaler tile once at the beginning (if init=true)
   - Handle input tiles with wait/pop per tile
   - Handle output tiles with reserve/pack/push

3. **Compile-time Dispatch**
   - Use `if constexpr` to select the appropriate reduction loop pattern
   - Zero runtime overhead for pattern selection

### Implementation Outline

```cpp
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    bool init,
    bool uninit,
    bool enforce_fp32_accumulation>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_batches) {

    // Initialization
    if constexpr (init) {
        reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, ocb);
        cb_wait_front(icb_scaler, 1);  // Wait for scaler tile
    }

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == REDUCE_SCALAR) {
        // HW reduction: all tiles -> 1 output tile per batch
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            tile_regs_acquire();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    cb_wait_front(icb, 1);
                    reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                        icb, icb_scaler, 0, 0, 0);
                    cb_pop_front(icb, 1);
                }
            }
            cb_reserve_back(ocb, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, ocb);
            tile_regs_release();
            cb_push_back(ocb, 1);
        }
    } else if constexpr (reduce_dim == REDUCE_ROW) {
        // W reduction: each row -> 1 output tile (Ht outputs per batch)
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                tile_regs_acquire();
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    cb_wait_front(icb, 1);
                    reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                        icb, icb_scaler, 0, 0, 0);
                    cb_pop_front(icb, 1);
                }
                cb_reserve_back(ocb, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, ocb);
                tile_regs_release();
                cb_push_back(ocb, 1);
            }
        }
    } else { // REDUCE_COL
        // H reduction: each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += dest_limit) {
                uint32_t chunk_end = (wt + dest_limit < Wt) ? (wt + dest_limit) : Wt;
                uint32_t chunk_size = chunk_end - wt;

                tile_regs_acquire();
                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    uint32_t dst_idx = 0;
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        cb_wait_front(icb, 1);
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, 0, 0, dst_idx++);
                        cb_pop_front(icb, 1);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < chunk_size; ++i) {
                    cb_reserve_back(ocb, 1);
                    pack_tile(i, ocb);
                    cb_push_back(ocb, 1);
                }
                tile_regs_release();
            }
        }
    }

    // Cleanup
    if constexpr (uninit) {
        reduce_uninit<enforce_fp32_accumulation>();
    }
}
```

## Usage Example

### Before (Current)
```cpp
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
    reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    cb_wait_front(tt::CBIndex::c_2, 1);
    for (uint32_t nc = 0; nc < NC; nc++) {
        tile_regs_acquire();
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(tt::CBIndex::c_0, 1);
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0);
                cb_pop_front(tt::CBIndex::c_0, 1);
            }
        }
        cb_reserve_back(tt::CBIndex::c_3, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_3);
        tile_regs_release();
        cb_push_back(tt::CBIndex::c_3, 1);
    }
    reduce_uninit();
}
```

### After (With Helper)
```cpp
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.h"

void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    // Single line replaces the entire reduction loop!
    compute_kernel_lib::reduce(
        tt::CBIndex::c_0,   // input CB
        tt::CBIndex::c_2,   // scaler CB
        tt::CBIndex::c_3,   // output CB
        Ht, Wt, NC);
}
```

## Additional Considerations

### Template Parameters vs Compile-Time Args
- `reduce_type` and `reduce_dim` should use defaults from REDUCE_OP/REDUCE_DIM defines when possible
- This allows kernel to be parameterized via program defines without code changes

### Scaler Tile Management
- The scaler tile should remain in the CB after reduce completes
- User is responsible for ensuring scaler is pushed to CB before calling reduce
- This matches current behavior and allows scaler reuse

### Error Handling
- No runtime error handling (matches LLK philosophy)
- Trust user to provide valid parameters
- Compile-time checks via static_assert where possible

### DEST Register Capacity
- Auto-detect from JIT headers (DST_SYNC_MODE, DST_ACCUM_MODE)
- Reuse `get_dest_limit()` pattern from `untilize_helpers.h`
- REDUCE_COL requires chunking when Wt > dest_limit

## Implementation Steps

1. Create `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.h` with header guards and includes
2. Implement `get_dest_limit()` function (can reuse from untilize_helpers.h)
3. Implement the main `reduce` template function with all three patterns
4. Add comprehensive documentation and usage examples
5. Test with existing reduce kernels by replacing manual loops

## Files to Create

| File | Action | Description |
|------|--------|-------------|
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.h` | Create | Main helper header |

## Codebase Compatibility Analysis

### Kernels That Are a GOOD FIT

These kernels use the exact sequential reduce pattern and would directly benefit from the helper:

| Kernel | Pattern | Current Lines | After Helper |
|--------|---------|---------------|--------------|
| `ttnn/.../reduction/generic/device/kernels/compute/reduce_hw.cpp` | REDUCE_SCALAR | ~25 | ~3 |
| `ttnn/.../reduction/generic/device/kernels/compute/reduce_w.cpp` | REDUCE_ROW | ~25 | ~3 |
| `ttnn/.../reduction/generic/device/kernels/compute/reduce_h.cpp` | REDUCE_COL | ~30 | ~3 |
| `tests/tt_metal/tt_metal/test_kernels/compute/reduce_hw.cpp` | REDUCE_SCALAR | ~25 | ~3 |
| `tests/tt_metal/tt_metal/test_kernels/compute/reduce_w.cpp` | REDUCE_ROW | ~25 | ~3 |
| `tests/tt_metal/tt_metal/test_kernels/compute/reduce_h.cpp` | REDUCE_COL | ~25 | ~3 |

**Total: 6 kernels** that are perfect matches for the helper.

### Kernels That Are NOT a Good Fit

These kernels use `reduce_tile` but have patterns incompatible with the simple helper:

#### 1. Moreh Operations (Complex Multi-Phase with Masking)
| Kernel | Reason Not Compatible |
|--------|----------------------|
| `moreh_sum_h.cpp` | Masking logic, intermediate CBs, multi-phase reduce |
| `moreh_mean_h.cpp` | Same as moreh_sum_h |
| `moreh_norm_h_kernel.cpp` | Multi-phase with intermediate results |
| `moreh_norm_w_kernel.cpp` | Multi-phase with intermediate results |
| `moreh_softmax_*.cpp` | Complex control flow, exp + reduce interleaved |
| `moreh_layer_norm_*.cpp` | Multi-step normalization pipeline |
| `moreh_bias_backward_*.cpp` | Masking + reload from intermediate CB |
| `moreh_dot.cpp` | mul_tiles + reduce_tile interleaved |

#### 2. Normalization Kernels (Interleaved Operations)
| Kernel | Reason Not Compatible |
|--------|----------------------|
| `layernorm_*.cpp` | E[x], E[x²] phases, subtract mean, multiply gamma/beta |
| `rmsnorm_*.cpp` | Similar to layernorm, multi-phase |
| `groupnorm*.cpp` | Pre-loaded indexed tiles, complex control flow |
| `softmax*.cpp` | Max reduce + exp + sum reduce interleaved |

#### 3. Pool Operations (Combined Tilize/Reduce/Untilize)
| Kernel | Reason Not Compatible |
|--------|----------------------|
| `compute_pool_2d.cpp` | Uses `reduce_tile_math`, tilize+reduce+untilize combined |
| `max_pool*.cpp` | Similar combined operations |

#### 4. Other Complex Patterns
| Kernel | Reason Not Compatible |
|--------|----------------------|
| `ssm_1d_sum_reduce.cpp` | Transpose + reduce interleaved (has own local `reduce()` helper) |
| `deepseek_grouped_gate.cpp` | Complex gating logic around reduce |
| `sampling.cpp` | Specialized sampling with reduce |
| `moe.cpp` | Mixture-of-experts routing logic |

### Pattern Analysis Summary

| Pattern | Count | Compatible? |
|---------|-------|-------------|
| **Sequential wait/pop per tile, single DST** | 6 | YES |
| **Pre-loaded tiles with indexed access** | 8+ | NO (different pattern) |
| **Reduce interleaved with other ops** | 20+ | NO (not pure reduce) |
| **Multi-phase reduce (E[x] then E[x²])** | 10+ | NO (init/uninit per phase) |
| **Reduce with masking** | 6+ | NO (mask logic interleaved) |

### Conclusion

The proposed `reduce()` helper is well-suited for its intended purpose:
- **6 kernels** will directly benefit (generic reduce and test kernels)
- The helper correctly targets the "pure reduce" pattern
- More complex patterns intentionally require manual implementation

The scope is appropriate - trying to handle more complex patterns would:
1. Significantly complicate the API
2. Add runtime overhead for conditionals
3. Reduce clarity and maintainability

## Testing Strategy

### Verification Approach

After replacing the manual reduce loops in the generic compute kernels with the `reduce()` helper, we must verify correctness by running the existing Python unit tests. These tests exercise all reduce patterns through the ttnn Python API.

### Python Unit Tests to Execute

The following test files exercise the generic reduce compute kernels (`reduce_hw.cpp`, `reduce_w.cpp`, `reduce_h.cpp`):

#### Core Reduction Tests

```bash
# Run all reduce tests
pytest tests/ttnn/unit_tests/operations/reduce/ -v

# Or run individual test files:

# 1. Sum reduction (REDUCE_OP=SUM)
pytest tests/ttnn/unit_tests/operations/reduce/test_sum.py -v

# 2. Max reduction (REDUCE_OP=MAX)
pytest tests/ttnn/unit_tests/operations/reduce/test_max.py -v

# 3. Min reduction (REDUCE_OP=MAX with sign flip)
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_min.py -v

# 4. Mean reduction (REDUCE_OP=SUM with scaling)
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_mean.py -v

# 5. General reductions (std, var, prod, topk)
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction.py -v
```

#### Test Coverage by Reduce Dimension

| Test | REDUCE_SCALAR (HW) | REDUCE_ROW (W) | REDUCE_COL (H) |
|------|-------------------|----------------|----------------|
| `test_sum.py::test_sum` | `dim=(2,1)` | `dim=-1` | `dim=-2` |
| `test_sum.py::test_sum_global` | `dim=None` | - | - |
| `test_sum.py::test_sum_4d` | `dim=None` | `dim=3` | `dim=2` |
| `test_max.py::test_max` | - | `dim=-1` | `dim=-2` |
| `test_max.py::test_max_global` | `dim=None` | - | - |
| `test_max.py::test_max_dim` | various | various | various |
| `test_reduction_min.py::test_min` | - | `dim=-1` | `dim=-2` |
| `test_reduction_min.py::test_min_global` | `dim=None` | - | - |
| `test_reduction_mean.py::test_mean` | - | `dim=-1` | `dim=-2` |
| `test_reduction.py::test_mean_*` | `dim=None` | `dim=-1` | `dim=-2` |
| `test_reduction.py::test_std` | - | `dim=-1` | `dim=-2` |
| `test_reduction.py::test_var` | `dim=None` | `dim=-1` | `dim=-2` |

#### Key Test Parameters

The tests cover:
- **Batch sizes**: 1, 16, 32
- **Heights (H)**: 32, 37, 41, 64 (both tile-aligned and non-aligned)
- **Widths (W)**: 31, 32, 63, 64 (both tile-aligned and non-aligned)
- **Data types**: `ttnn.bfloat16`, `ttnn.float32`, `ttnn.bfloat8_b`
- **keepdim**: True, False
- **Tensor ranks**: 2D, 3D, 4D, 5D, 6D, 7D, 8D

### Recommended Test Execution Order

1. **Smoke test** - Quick validation with minimal parameters:
   ```bash
   pytest tests/ttnn/unit_tests/operations/reduce/test_sum.py::test_sum -v -k "batch_size=1 and h=32 and w=32"
   pytest tests/ttnn/unit_tests/operations/reduce/test_max.py::test_max -v -k "batch_size=1 and h=32 and w=32"
   ```

2. **Full dimension coverage** - All reduce dimensions:
   ```bash
   pytest tests/ttnn/unit_tests/operations/reduce/test_sum.py -v
   pytest tests/ttnn/unit_tests/operations/reduce/test_max.py -v
   pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_min.py -v
   ```

3. **Full test suite** - Complete validation:
   ```bash
   pytest tests/ttnn/unit_tests/operations/reduce/ -v
   ```

### Expected Results

All tests should pass with the same PCC (Pearson Correlation Coefficient) thresholds as before:
- Most tests: PCC >= 0.99
- Some precision-sensitive tests: PCC >= 0.999

### Additional Validation

After the helper is integrated, also run:

```bash
# Reduction on batch dimension
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_on_batch.py -v

# H-interleaved reduction
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_h_interleaved.py -v
```

### Debugging Failed Tests

If tests fail after integration:
1. Compare the generated kernel code (check if helper expands correctly)
2. Verify template parameters are correctly deduced
3. Check DEST register limits for REDUCE_COL with large Wt
4. Ensure scaler CB has tile ready before `reduce()` call
