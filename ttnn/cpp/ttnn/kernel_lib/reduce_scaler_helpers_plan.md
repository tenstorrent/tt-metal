# Reduce Scaler Helper Library Migration Plan

## Overview

This document outlines the plan to migrate the `generate_reduce_scaler` function from its deprecated location to the new kernel helper library.

## Analysis Summary

### What is a Reduce Scaler?

A reduce scaler is a scaling factor applied during reduction operations (particularly for AVG pooling) to normalize the result. It's typically:
- A `float` value converted to `bfloat16` and packed into a `uint32_t`
- For SUM/MAX reductions: typically `1.0f`
- For AVG reduction: `1/N` where N is the number of elements being averaged

The scaler is placed in a circular buffer tile with a specific fill pattern that the reduction hardware expects.

---

## Existing Helpers (Scattered Locations)

| Helper | Location | Purpose | Migration |
|--------|----------|---------|-----------|
| `generate_reduce_scaler<half_tile>` | `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp` | Fill first row (8 elements) of each face with scaler | → `ttnn::kernel_lib::dataflow::generate_reduce_scaler<half_tile>` |

### Fill Patterns Explained

#### REDUCE_ROW Pattern (`generate_reduce_scaler`)
Used for row-wise reductions. Fills first row (8 uint32 elements = 16 bf16 values) of each of the 4 faces:
```
Face layout (each face is 16x16 elements):
[S S S S S S S S 0 0 0 0 0 0 0 0]  <- Row 0: first 8 elements filled
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
...
```

---

## Inline Implementations Found (Need Migration)

### Direct Duplicates

1. **`ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_h.cpp:27-52`**
   - Exact duplicate of `generate_reduce_scaler` logic
   - Lines 30-52 can be replaced with single function call

### Kernels Using Deprecated Helpers

Total of **36 files** reference `reduce_scaler` patterns:

**Reduction Operations:**
- `reader_tilize_untilize_interleaved.cpp`
- `reader_unary_reduce_universal_start_id.cpp`
- `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`
- `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`

**Transformer/SDPA:**
- `writer_decode_all.cpp`
- `writer_windowed.cpp`
- `joint_writer.cpp`
- `ring_joint_writer.cpp`
- `writer_interleaved.cpp` (multiple)

**Normalization:**
- `reader_unary_interleaved_sm.cpp`
- `reader_unary_interleaved_sm_large_tensor.cpp`
- `reader_unary_sharded_sm.cpp`
- `reader_unary_sharded_sm_causal_mask_hw_dims.cpp`
- `reader_unary_sharded_sm_rm_mask.cpp`
- `reader_unary_interleaved_ln.cpp`
- `reader_unary_interleaved_ln_rm_gb.cpp`
- `reader_unary_interleaved_ln_large_tensor.cpp`
- `writer_unary_sharded_ln.cpp`
- `writer_unary_sharded_ln_rm_gb.cpp`
- `writer_unary_sharded_ln_pre_all_gather.cpp`
- `welford_writer_unary_gn_rm_gb.cpp`
- `writer_unary_gn_rm_gb.cpp`

**LayerNorm Distributed:**
- `reader_layernorm_preallgather_2d.cpp`
- `reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`
- `reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp`

**Moreh Operations:**
- `reader_moreh_sum_h.cpp`
- `reader_moreh_sum_nc.cpp`
- `reader_moreh_bias_backward_h.cpp`

**Experimental:**
- `rms_post_allgather_reader.cpp`
- `rms_pre_allgather_reader.cpp`
- `reader_reduce_nc.cpp`
- `reader_ssm_1d_sum_reduce.cpp`
- `rms_writer.cpp`

---

## New Helper Library

### Location
`ttnn/cpp/ttnn/operations/kernel_lib/dataflow/reduce_scaler.hpp`

### Proposed API

```cpp
#pragma once
#include "dataflow_api.h"

namespace ttnn::kernel_lib::dataflow {

/**
 * @brief Generate a reduce scaler tile
 *
 * Creates a tile in the specified circular buffer with the scaler value
 * placed in row 0 of each face. The tile is first zeroed, then
 * positions [0-7] of row 0 in each face are filled with the scaler.
 *
 * @tparam half_tile If true, only fill faces 0-1 (half tile mode)
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 *
 * @note The function handles cb_reserve_back and cb_push_back internally
 */
template <bool half_tile = false>
FORCE_INLINE void generate_reduce_scaler(uint32_t cb_id, uint32_t scaler);

} // namespace ttnn::kernel_lib::dataflow
```

### Implementation

```cpp
#pragma once

#include "dataflow_api.h"

namespace ttnn::kernel_lib::dataflow {

// Face size in uint32 (128 u32 = 256 bf16 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32 = 128;
// Row size in uint32 (8 u32 = 16 bf16)
constexpr uint32_t ROW_SIZE_U32 = 8;

namespace detail {

template <bool half_tile>
FORCE_INLINE void zero_faces(uint32_t write_addr) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;
    constexpr uint32_t bytes_to_zero = num_faces * FACE_SIZE_U32 * sizeof(uint32_t);
    constexpr uint32_t num_zeros_reads = bytes_to_zero / MEM_ZEROS_SIZE;

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);

    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

template <bool half_tile>
FORCE_INLINE void fill_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;

    #pragma unroll
    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * FACE_SIZE_U32;
        #pragma unroll
        for (uint32_t col = 0; col < ROW_SIZE_U32; ++col) {
            ptr[face_offset + col] = scaler;
        }
    }
}

}  // namespace detail

template <bool half_tile>
FORCE_INLINE void generate_reduce_scaler(uint32_t cb_id, uint32_t scaler) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    detail::zero_faces<half_tile>(write_addr);

    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    detail::fill_row0<half_tile>(ptr, scaler);

    cb_push_back(cb_id, 1);
}

}  // namespace ttnn::kernel_lib::dataflow
```

### Usage Examples

**Before (using deprecated helper):**
```cpp
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // ...
    generate_reduce_scaler(cb_id_in2, scaler);
}
```

**After (using new helper):**
```cpp
#include "ttnn/cpp/ttnn/operations/kernel_lib/dataflow/reduce_scaler.hpp"

using ttnn::kernel_lib::dataflow::generate_reduce_scaler;

void kernel_main() {
    // ...
    generate_reduce_scaler(cb_id_in2, scaler);  // full tile (default)
    // or
    generate_reduce_scaler<true>(cb_id_in2, scaler);  // half tile
}
```

---

## Migration Plan

### Tier 1: Easy (Direct replacement, no logic changes)

These kernels simply include the deprecated header and call `generate_reduce_scaler`. Migration is mechanical.

| Kernel | File Path | Notes |
|--------|-----------|-------|
| `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `reduction/generic/device/kernels/dataflow/` | Simple include swap |
| `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `reduction/generic/device/kernels/dataflow/` | Simple include swap |
| `reader_unary_reduce_universal_start_id.cpp` | `reduction/generic/device/kernels/dataflow/` | Simple include swap |
| `reader_tilize_untilize_interleaved.cpp` | `reduction/tilize_untilize/device/kernels/dataflow/` | Conditional on `REDUCE_OP` define |
| `reader_reduce_nc.cpp` | `experimental/reduction/fast_reduce_nc/device/kernels/` | Simple include swap |
| `joint_writer.cpp` | `transformer/sdpa/device/kernels/dataflow/` | Simple include swap |
| `ring_joint_writer.cpp` | `transformer/sdpa/device/kernels/dataflow/` | Simple include swap |
| `writer_interleaved.cpp` (SDPA) | `transformer/sdpa/device/kernels/dataflow/` | Simple include swap |
| `reader_moreh_sum_h.cpp` | `moreh/moreh_sum/device/moreh_sum_h_impl_kernels/` | Simple include swap |
| `reader_ssm_1d_sum_reduce.cpp` | `experimental/ssm/hc_sum_reduce/device/kernels/` | Simple include swap |

**Estimated effort:** 10 files, ~30 minutes

### Tier 2: Moderate (Inline code to migrate or multiple helpers)

| Kernel | File Path | Current State | Migration Notes |
|--------|-----------|---------------|-----------------|
| `reader_moreh_mean_h.cpp` | `moreh/moreh_mean/device/kernels/` | **Inline implementation** | Replace lines 27-52 with function call |
| `reader_unary_sharded_sm.cpp` | `normalization/softmax/device/kernels/attention/dataflow/` | Multiple conditional paths | Test all code paths |
| `reader_unary_interleaved_ln_rm_gb.cpp` | `normalization/layernorm/device/kernels/dataflow/` | Uses `generate_reduce_scaler` | Migrate reduce_scaler only |
| All softmax readers | `normalization/softmax/device/kernels/attention/dataflow/` | Various conditional compilation | Careful testing required |

**Estimated effort:** 14 files, ~2-3 hours

---

## Recommended Migration Order

### Phase 1: Create Library & Migrate Easiest Cases (Day 1)

1. **Create the helper library header**
   - `ttnn/cpp/ttnn/operations/kernel_lib/dataflow/reduce_scaler.hpp`
   - Implement based on existing `generate_reduce_scaler.hpp`
   - Add proper documentation

2. **Migrate `reader_moreh_mean_h.cpp`** (First win - inline code)
   - Replace lines 27-52 with single function call
   - Verify test passes: `pytest tests/ttnn/unit_tests/operations/moreh/test_moreh_mean.py -v`

3. **Migrate 5 simple reduction kernels**
   - Update includes, verify no behavior change

### Phase 2: Migrate SDPA & More Reductions (Day 2)

4. **Migrate SDPA kernels** (6 files)
   - `joint_writer.cpp`, `ring_joint_writer.cpp`, `writer_interleaved.cpp`, etc.
   - Run SDPA tests

5. **Migrate remaining reduction kernels**
   - `reader_reduce_nc.cpp`, `reader_ssm_1d_sum_reduce.cpp`

### Phase 3: Migrate Normalization Kernels (Day 3-4)

6. **Migrate softmax kernels** (5 files)
   - Multiple conditional compilation paths - test thoroughly

7. **Migrate LayerNorm kernels** (6 files)
   - Includes both sharded and interleaved variants

8. **Migrate GroupNorm kernels** (2 files)
   - Only kernels using `generate_reduce_scaler`

### Phase 4: Deprecate Old Headers (Day 5)

9. **Add deprecation warnings to old helpers**
    ```cpp
    [[deprecated("Use ttnn::kernel_lib::dataflow::generate_reduce_scaler instead")]]
    ```

10. **Update documentation**
    - Add to CLAUDE.md if appropriate
    - Create usage examples

---

## Testing Strategy

### Unit Tests to Run After Each Migration

```bash
# Reduction operations
pytest tests/ttnn/unit_tests/operations/reduction/ -v

# SDPA
pytest tests/ttnn/unit_tests/operations/transformer/ -v -k sdpa

# Normalization
pytest tests/ttnn/unit_tests/operations/normalization/ -v

# Moreh operations
pytest tests/ttnn/unit_tests/operations/moreh/ -v

# Full sweep (after all migrations)
pytest tests/ttnn/unit_tests/operations/ -v
```

### Specific Tests for Key Operations

```bash
# Moreh mean (first migration target)
pytest tests/ttnn/unit_tests/operations/moreh/test_moreh_mean.py -v

# LayerNorm
pytest tests/ttnn/unit_tests/operations/normalization/test_layernorm.py -v

# Softmax
pytest tests/ttnn/unit_tests/operations/normalization/test_softmax.py -v
```

---

## Start Point Recommendation

**Start with `reader_moreh_mean_h.cpp`** because:

1. Has inline code that's an exact duplicate of the helper (lines 27-52)
2. Self-contained - no conditional compilation affecting the scaler generation
3. Low risk - isolated moreh operation
4. Proves the library works before tackling more complex cases
5. Clear test to validate: `test_moreh_mean.py`

---

## File Changes Summary

### New Files
- `ttnn/cpp/ttnn/operations/kernel_lib/dataflow/reduce_scaler.hpp`

### Files to Modify (36 total)
- See "Kernels Using Deprecated Helpers" section above

### Files to Deprecate (Eventually)
- `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`
