# Migration Guide: `TileShape` API for `reduce()`

## Overview

Replace positional `Ht, Wt, num_batches` parameters with a `TileShape` struct using static factory methods. This provides:

- Self-documenting call sites
- Safe defaults for common cases
- Consistent API style with existing `InputMemoryLayout`

---

## New API

### TileShape Struct

```cpp
struct TileShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    // Full grid specification
    static constexpr TileShape grid(uint32_t r, uint32_t c, uint32_t b = 1) {
        return {r, c, b};
    }

    // Single tile (1x1x1) - for scalar reductions on one tile
    static constexpr TileShape single() {
        return {1, 1, 1};
    }

    // Single row of tiles (1 x cols x 1) - common for REDUCE_ROW
    static constexpr TileShape row(uint32_t c, uint32_t b = 1) {
        return {1, c, b};
    }

    // Single column of tiles (rows x 1 x 1) - common for REDUCE_COL
    static constexpr TileShape col(uint32_t r, uint32_t b = 1) {
        return {r, 1, b};
    }
};
```

### Updated Function Signature

```cpp
template <
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    ReduceInputMode input_mode = ReduceInputMode::STREAMING,
    bool init = true,
    bool uninit = true,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    TileShape shape,                    // <-- NEW: replaces Ht, Wt, num_batches
    InputMemoryLayout layout = {},
    PostReduceOp post_reduce_op = {});
```

---

## Migration Table

| File | Before | After |
|------|--------|-------|
| `reduce_h.cpp` | `reduce(..., Ht, Wt, NC)` | `reduce(..., TileShape::grid(Ht, Wt, NC))` |
| `reduce_w.cpp` | `reduce(..., Ht, Wt, NC)` | `reduce(..., TileShape::grid(Ht, Wt, NC))` |
| `reduce_hw.cpp` | `reduce(..., Ht, Wt, NC)` | `reduce(..., TileShape::grid(Ht, Wt, NC))` |
| `groupnorm_sharded_v2.cpp` | `reduce(..., 1, 1, 1)` | `reduce(..., TileShape::single())` |
| `sampling.cpp` | `reduce(..., rows, cols, 1)` | `reduce(..., TileShape::grid(rows, cols))` |
| `compute_common.hpp` | `reduce(..., rows, cols, 1, {}, lambda)` | `reduce(..., TileShape::grid(rows, cols), {}, lambda)` |
| `softmax.cpp` | `reduce(..., 1, Wt, 1)` | `reduce(..., TileShape::row(Wt))` |
| `softmax_sharded.cpp` | `reduce(..., 1, Wt, 1)` | `reduce(..., TileShape::row(Wt))` |
| `softmax_large_tensor.cpp` | `reduce(..., 1, cb_length_t, 1, {}, lambda)` | `reduce(..., TileShape::row(cb_length_t), {}, lambda)` |
| `rmsnorm_post_allgather.cpp` | `reduce(..., 1, stats_tiles_cols, 1)` | `reduce(..., TileShape::row(stats_tiles_cols))` |
| `layernorm_post_allgather.cpp` | `reduce(..., 1, stats_tiles_cols, 1)` | `reduce(..., TileShape::row(stats_tiles_cols))` |

---

## Detailed Migration Examples

### Generic Reduce Kernels

```cpp
// Before (reduce_h.cpp, reduce_w.cpp, reduce_hw.cpp)
compute_kernel_lib::reduce(
    tt::CBIndex::c_0,
    tt::CBIndex::c_2,
    tt::CBIndex::c_3,
    Ht,
    Wt,
    NC);

// After
compute_kernel_lib::reduce(
    tt::CBIndex::c_0,
    tt::CBIndex::c_2,
    tt::CBIndex::c_3,
    TileShape::grid(Ht, Wt, NC));
```

### Single Tile Reduction

```cpp
// Before (groupnorm_sharded_v2.cpp)
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    cb_ex_external,
    cb_scaler_global,
    cb_ex_global,
    1,
    1,
    1);

// After
compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
    cb_ex_external,
    cb_scaler_global,
    cb_ex_global,
    TileShape::single());
```

### Row Reduction Pattern

```cpp
// Before (softmax.cpp)
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ...>(
    cb_in,
    cb_bcast_scaler,
    cb_max,
    1,
    Wt,
    1);

// After
compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ...>(
    cb_in,
    cb_bcast_scaler,
    cb_max,
    TileShape::row(Wt));
```

### With Post-Reduce Lambda

```cpp
// Before (compute_common.hpp)
compute_kernel_lib::reduce<pool_type, reduce_dim, ReduceInputMode::PRELOADED>(
    in0_cb, scale_cb, out_cb, rows, cols, 1, {}, [&]() {
        // lambda body
    });

// After
compute_kernel_lib::reduce<pool_type, reduce_dim, ReduceInputMode::PRELOADED>(
    in0_cb, scale_cb, out_cb, TileShape::grid(rows, cols), {}, [&]() {
        // lambda body
    });
```

---

## Implementation Changes

### In `reduce_helpers.hpp`

```cpp
namespace compute_kernel_lib {

struct TileShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    static constexpr TileShape grid(uint32_t r, uint32_t c, uint32_t b = 1) {
        return {r, c, b};
    }

    static constexpr TileShape single() {
        return {1, 1, 1};
    }

    static constexpr TileShape row(uint32_t c, uint32_t b = 1) {
        return {1, c, b};
    }

    static constexpr TileShape col(uint32_t r, uint32_t b = 1) {
        return {r, 1, b};
    }
};

template <...>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    TileShape shape,
    InputMemoryLayout layout = {},
    PostReduceOp post_reduce_op = {})
{
    // Replace internal usage:
    // Ht         -> shape.rows
    // Wt         -> shape.cols
    // num_batches -> shape.batches

    // ... rest of implementation
}

}  // namespace compute_kernel_lib
```

---

## Internal Variable Mapping

Within `reduce()` implementation, update references:

| Old Variable | New Access |
|--------------|------------|
| `Ht` | `shape.rows` |
| `Wt` | `shape.cols` |
| `num_batches` | `shape.batches` |

---

## Migration Steps

1. **Add `TileShape` struct** to `reduce_helpers.hpp` (above the `reduce` function)

2. **Update `reduce()` signature** to take `TileShape shape` instead of three separate params

3. **Update `reduce()` implementation** to use `shape.rows`, `shape.cols`, `shape.batches`

4. **Update documentation/examples** in the header comment block

5. **Migrate call sites** (see table above):
   - `reduce_h.cpp`
   - `reduce_w.cpp`
   - `reduce_hw.cpp`
   - `groupnorm_sharded_v2.cpp`
   - `sampling.cpp`
   - `compute_common.hpp`
   - `softmax.cpp`
   - `softmax_sharded.cpp`
   - `softmax_large_tensor.cpp`
   - `rmsnorm_post_allgather.cpp`
   - `layernorm_post_allgather.cpp`

6. **Run validation tests** (see Testing section below)

---

## Build Notes

**No explicit build required.** These are kernel-only changes and kernels are compiled with JIT (Just-In-Time) compilation. The tests will trigger kernel compilation automatically.

---

## Testing

### Required Test Suites

All of the following tests must pass after migration:

```bash
# Fused operations (includes softmax, layernorm, groupnorm, etc.)
pytest tests/ttnn/unit_tests/operations/fused/

# Reduce operations
pytest tests/ttnn/unit_tests/operations/reduce/

# SDPA decode (uses reduce in compute_common.hpp)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention_decode.py::test_sdpa_decode
```

### Troubleshooting: Test Hangs

If any test hangs, the device may be in an incorrect state. Run the following to reset:

```bash
tt-smi -r
```

Then re-run the failing test.

---

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| Single tile | `1, 1, 1` | `TileShape::single()` |
| Row pattern | `1, Wt, 1` | `TileShape::row(Wt)` |
| Column pattern | `Ht, 1, 1` | `TileShape::col(Ht)` |
| Full grid | `Ht, Wt, NC` | `TileShape::grid(Ht, Wt, NC)` |
| Readability | Positional, cryptic | Self-documenting |
| Default batches | Must specify `1` | Defaults to `1` |
| API consistency | Different from `InputMemoryLayout` | Matches `InputMemoryLayout` pattern |
