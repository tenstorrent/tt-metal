# Reduce Scaler Helper Library Migration Plan

## Overview

This document outlines the plan to migrate the `generate_reduce_scaler` function from its deprecated location to the new kernel helper library.

## Current Status

**Library Location:** `ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp`

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Create library & migrate first batch |
| Phase 2 | ✅ Complete | Migrate SDPA, reduction, softmax, moreh kernels |
| Phase 3 | ✅ Complete | Migrate remaining 20 files |
| Phase 4 | ✅ Complete | Migrate DIT LayerNorm (✅ 2/2) & UDM test files (✅ 3/3) |
| Phase 5 | 🔲 Pending | Deprecate old header |

## Analysis Summary

### What is a Reduce Scaler?

A reduce scaler is a scaling factor applied during reduction operations (particularly for AVG pooling) to normalize the result. It's typically:
- A `float` value converted to `bfloat16` and packed into a `uint32_t`
- For SUM/MAX reductions: typically `1.0f`
- For AVG reduction: `1/N` where N is the number of elements being averaged

The scaler is placed in a circular buffer tile with a specific fill pattern that the reduction hardware expects.

---

## Helpers

| Helper | Location | Status |
|--------|----------|--------|
| `dataflow_kernel_lib::generate_reduce_scaler<half_tile>` | `ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp` | ✅ New library (use this) |
| `generate_reduce_scaler<half_tile>` | `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp` | ⚠️ Deprecated (migrate away) |

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

## Inline Implementations Found

### Direct Duplicates

1. **`ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_h.cpp`**
   - ✅ **MIGRATED** - Now uses `dataflow_kernel_lib::generate_reduce_scaler`

### Kernels Migration Status

#### ✅ Migrated to New Library (36 files)

**Reduction Operations:**
- ✅ `reader_unary_reduce_universal_start_id.cpp` - `reduction/generic/device/kernels/dataflow/`
- ✅ `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` - `reduction/generic/device/kernels/dataflow/`
- ✅ `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` - `reduction/generic/device/kernels/dataflow/`
- ✅ `writer_unary_interleaved.cpp` - `reduction/moe/device/kernels/dataflow/`
- ✅ `writer_interleaved.cpp` - `reduction/sampling/device/kernels/dataflow/`

**Transformer/SDPA:**
- ✅ `joint_writer.cpp` - `transformer/sdpa/device/kernels/dataflow/`
- ✅ `ring_joint_writer.cpp` - `transformer/sdpa/device/kernels/dataflow/`
- ✅ `writer_interleaved.cpp` - `transformer/sdpa/device/kernels/dataflow/`
- ✅ `writer_windowed.cpp` - `transformer/sdpa_windowed/device/kernels/dataflow/`
- ✅ `writer_decode_all.cpp` - `transformer/sdpa_decode/device/kernels/dataflow/`

**Softmax:**
- ✅ `reader_unary_interleaved_sm.cpp` - `normalization/softmax/device/kernels/attention/dataflow/`
- ✅ `reader_unary_interleaved_sm_large_tensor.cpp` - `normalization/softmax/device/kernels/attention/dataflow/`
- ✅ `reader_unary_sharded_sm.cpp` - `normalization/softmax/device/kernels/attention/dataflow/`
- ✅ `reader_unary_sharded_sm_causal_mask_hw_dims.cpp` - `normalization/softmax/device/kernels/attention/dataflow/`
- ✅ `reader_unary_sharded_sm_rm_mask.cpp` - `normalization/softmax/device/kernels/attention/dataflow/`

**LayerNorm:**
- ✅ `reader_unary_interleaved_ln_rm_gb.cpp` - `normalization/layernorm/device/kernels/dataflow/`
- ✅ `reader_unary_interleaved_ln.cpp` - `normalization/layernorm/device/kernels/dataflow/`
- ✅ `reader_unary_interleaved_ln_large_tensor.cpp` - `normalization/layernorm/device/kernels/dataflow/`
- ✅ `writer_unary_sharded_ln.cpp` - `normalization/layernorm/device/kernels/dataflow/`
- ✅ `writer_unary_sharded_ln_rm_gb.cpp` - `normalization/layernorm/device/kernels/dataflow/`
- ✅ `writer_unary_sharded_ln_pre_all_gather.cpp` - `normalization/layernorm/device/kernels/dataflow/`

**LayerNorm Distributed:**
- ✅ `reader_layernorm_preallgather_2d.cpp` - `normalization/layernorm_distributed/device/kernels/dataflow/`
- ✅ `reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` - `normalization/layernorm_distributed/device/kernels/dataflow/`
- ✅ `reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` - `normalization/layernorm_distributed/device/kernels/dataflow/`

**GroupNorm:**
- ✅ `welford_writer_unary_gn_rm_gb.cpp` - `normalization/groupnorm/device/kernels/dataflow/`
- ✅ `writer_unary_gn_rm_gb.cpp` - `normalization/groupnorm/device/kernels/dataflow/`
- ✅ `writer_unary_sharded_gn_rm_gb_v2.cpp` - `normalization/groupnorm/device/kernels/dataflow/`

**Moreh Operations:**
- ✅ `reader_moreh_mean_h.cpp` - `moreh/moreh_mean/device/kernels/`
- ✅ `reader_moreh_sum_h.cpp` - `moreh/moreh_sum/device/moreh_sum_h_impl_kernels/`
- ✅ `reader_moreh_sum_nc.cpp` - `moreh/moreh_sum/device/moreh_sum_nc_impl_kernels/`
- ✅ `reader_moreh_bias_backward_h.cpp` - `moreh/moreh_linear_backward/device/kernels/`

**Experimental:**
- ✅ `reader_reduce_nc.cpp` - `experimental/reduction/fast_reduce_nc/device/kernels/`
- ✅ `reader_ssm_1d_sum_reduce.cpp` - `experimental/ssm/hc_sum_reduce/device/kernels/`
- ✅ `rms_post_allgather_reader.cpp` - `experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/`
- ✅ `rms_pre_allgather_reader.cpp` - `experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/`
- ✅ `rms_writer.cpp` - `experimental/ccl/rms_allgather/device/kernels/dataflow/`

#### ⚠️ Still Using Deprecated Helper (0 files)

All files have been successfully migrated to the new helper library!

**DIT LayerNorm Operations (2 files) - ✅ MIGRATED:**
- ✅ `reader_layernorm_preallgather_dit.cpp` - `experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/dataflow/`
- ✅ `reader_layernorm_postallgather_dit.cpp` - `experimental/transformer/dit_layernorm_post_all_gather/device/kernels/dataflow/`

**UDM Test Files (3 files) - ✅ MIGRATED:**
- ✅ `dataflow_reduce.cpp` - `tests/ttnn/unit_tests/gtests/udm/reduction/interleaved/kernels/`
- ✅ `reader_receiver_unary_sharded_reduce.cpp` - `tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/`
- ✅ `reader_sender_unary_sharded_reduce.cpp` - `tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/`

---

## New Helper Library

### Location
`ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp`

### Proposed API

```cpp
#pragma once
#include "dataflow_api.h"

namespace dataflow_kernel_lib {

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

} // namespace dataflow_kernel_lib
```

### Implementation

```cpp
#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

// Face size in uint32 (128 u32 = 256 bf16 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32 = 128;
// Row size in uint32 (8 u32 = 16 bf16)
constexpr uint32_t ROW_SIZE_U32 = 8;

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

    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * FACE_SIZE_U32;
        for (uint32_t col = 0; col < ROW_SIZE_U32; ++col) {
            ptr[face_offset + col] = scaler;
        }
    }
}

template <bool half_tile = false>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<half_tile>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
        fill_row0<half_tile>(ptr, scaler);
    }

    cb_push_back(cb_id, 1);
}

}  // namespace dataflow_kernel_lib
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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp"

using dataflow_kernel_lib::generate_reduce_scaler;

void kernel_main() {
    // ...
    generate_reduce_scaler(cb_id_in2, scaler);  // full tile (default)
    // or
    generate_reduce_scaler<true>(cb_id_in2, scaler);  // half tile
}
```

---

## Migration Plan

### ✅ Phase 1: Create Library & Migrate First Batch (COMPLETED)

1. **Created the helper library header**
   - `ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp`
   - Implemented based on existing `generate_reduce_scaler.hpp`
   - Added proper documentation

2. **Migrated `reader_moreh_mean_h.cpp`**
   - Replaced inline implementation with function call

3. **Migrated reduction kernels**
   - `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`
   - `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp`
   - `reader_unary_reduce_universal_start_id.cpp`
   - `reader_reduce_nc.cpp`
   - `reader_ssm_1d_sum_reduce.cpp`

### ✅ Phase 2: Migrate SDPA, Softmax & Moreh (COMPLETED)

4. **Migrated SDPA kernels**
   - `joint_writer.cpp`
   - `ring_joint_writer.cpp`
   - `writer_interleaved.cpp`

5. **Migrated softmax kernels**
   - `reader_unary_interleaved_sm.cpp`
   - `reader_unary_interleaved_sm_large_tensor.cpp`
   - `reader_unary_sharded_sm.cpp`
   - `reader_unary_sharded_sm_causal_mask_hw_dims.cpp`
   - `reader_unary_sharded_sm_rm_mask.cpp`

6. **Migrated additional kernels**
   - `reader_unary_interleaved_ln_rm_gb.cpp`
   - `reader_moreh_sum_h.cpp`

### ✅ Phase 3: Migrate Remaining Kernels (COMPLETED)

All 20 remaining kernel files have been successfully migrated to use the new helper library. The migration involved:

1. **Transformer/SDPA variants (2 files)** - Include swap and using declaration
2. **Reduction operations (2 files)** - Include swap and using declaration
3. **LayerNorm kernels (5 files)** - Include swap and using declaration
4. **LayerNorm Distributed kernels (3 files)** - Include swap and using declaration
5. **GroupNorm kernels (3 files)** - Include swap and using declaration
6. **Moreh operations (2 files)** - Include swap and using declaration
7. **Experimental RMSNorm kernels (3 files)** - Include swap and using declaration

All migrations followed the same pattern:
- Replace `#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"`
- With `#include "ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp"`
- Add `using dataflow_kernel_lib::generate_reduce_scaler;`

Tests have been run and are passing successfully.

### ✅ Phase 4: Migrate DIT LayerNorm & UDM Test Files (COMPLETED)

**DIT LayerNorm Operations (2 files) - ✅ COMPLETED:**

1. **✅ `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/dataflow/reader_layernorm_preallgather_dit.cpp`**
   - ✅ Migrated: Replaced deprecated include with `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`
   - ✅ Updated function call to use full namespace: `dataflow_kernel_lib::generate_reduce_scaler`
   - ✅ All 172 DIT LayerNorm tests passing

2. **✅ `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/dataflow/reader_layernorm_postallgather_dit.cpp`**
   - ✅ Migrated: Replaced deprecated include with `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`
   - ✅ Header updated for consistency (doesn't directly call the function)

**UDM Test Files (3 files) - ✅ COMPLETED:**

3. **✅ `tests/ttnn/unit_tests/gtests/udm/reduction/interleaved/kernels/dataflow_reduce.cpp`**
   - ✅ Migrated: Replaced deprecated include with `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`
   - ✅ Updated function call to use full namespace: `dataflow_kernel_lib::generate_reduce_scaler`

4. **✅ `tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/reader_receiver_unary_sharded_reduce.cpp`**
   - ✅ Migrated: Replaced deprecated include with `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`
   - ✅ Updated function call to use full namespace: `dataflow_kernel_lib::generate_reduce_scaler`

5. **✅ `tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/reader_sender_unary_sharded_reduce.cpp`**
   - ✅ Migrated: Replaced deprecated include with `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`
   - ✅ Updated function call to use full namespace: `dataflow_kernel_lib::generate_reduce_scaler`

**Testing:**
- ✅ DIT LayerNorm: 172 tests passed (test_distributed_dit_layernorm.py)
- Note: UDM tests require special build configuration (unit_tests_ttnn_udm binary)

### 🔲 Phase 5: Deprecate Old Header

1. **Add deprecation warning to old helper**
    ```cpp
    [[deprecated("Use dataflow_kernel_lib::generate_reduce_scaler instead")]]
    ```

2. **Update documentation**
    - Add to CLAUDE.md if appropriate

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

## File Changes Summary

### New Files (Created)
- ✅ `ttnn/cpp/ttnn/kernel_lib/reduce_scaler_helpers.hpp`

### Files Migrated (41 total)
- ✅ 36 kernel files successfully migrated to new helper library (Phase 1-3)
- ✅ 2 DIT LayerNorm kernels successfully migrated (Phase 4)
- ✅ 3 UDM test kernel files successfully migrated (Phase 4)
- See "✅ Migrated to New Library" section above for complete list

### Files Pending Migration (0 total)
All files have been successfully migrated!

### Files to Deprecate (Phase 5)
- `ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp`

### Note: Alternative Implementation
- ⚠️ **Deepseek v3 B1** has a custom implementation at `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/dm_utils.hpp`
  - Different signature: `template <uint32_t num_faces = 4, uint32_t num_cols_per_face = 16> void generate_reduce_scaler(const uint32_t cb_id, const uint16_t scaler)`
  - Takes `uint16_t` instead of `uint32_t` for scaler parameter
  - This is a specialized variant and may not need migration

---

## Next Steps / Related Patterns

### Other Similar Patterns in the Codebase

The following related implementations exist in the codebase that follow similar patterns and could be candidates for future consolidation:

#### 1. Deepseek V3 B1 Custom Implementation
**Location:** `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/dm_utils.hpp`

- Different signature: `uint16_t` scaler instead of `uint32_t`
- Configurable template params for `num_faces` (default: 4) and `num_cols_per_face` (default: 16)
- Used by: `models/demos/deepseek_v3_b1/micro_ops/rmsnorm/kernels/rmsnorm_reader.cpp`
- **Status:** Specialized variant for Deepseek model, may not need migration

#### 2. Partial Reduce Scaler (Groupnorm-specific)
**Location:** `ttnn/cpp/ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h`

- Function: `generate_partial_reduce_scaler()`
- For partial column reductions where `num_cols < 16`
- Fills only up to `num_cols` columns in row 0
- Uses `tt::constants::TILE_HEIGHT/WIDTH/FACE_HEIGHT/WIDTH`
- **Status:** Active, used for custom groupnorm patterns

#### 3. Fill Tile Utilities
**Location:** `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`

Generic tile filling patterns (not reduction-specific but similar approach):
- `fill_with_val_bfloat16()` - Fill entire tile with single value
- `fill_tile_with_first_element_bfloat16()` - Replicate first element
- `fill_tile_with_first_row_bfloat16()` - Replicate first row across faces
- `fill_tile_with_first_column_bfloat16()` - Replicate first column
- **Status:** Active, general-purpose tile filling utilities

#### 4. Simplified uint16_t Scaler Overload
Consider adding an overload signature that accepts `const uint16_t scaler` and performs the packing internally:
```cpp
uint32_t scaler_32 = scaler | (scaler << 16);
```
This would remove the packing burden from the caller and make the API more ergonomic.

### Future Considerations

- Consider whether the partial reduce scaler from `custom_tiles.h` should be added to the main library
- Evaluate if the fill tile utilities could benefit from consolidation with reduce scaler helpers
- Monitor Deepseek implementation for potential unification if model-specific customizations become more common
- Add `uint16_t` scaler overload to simplify caller usage (see point 4 above)
