# Binary Operation Kernel Analysis

## Executive Summary

This document analyzes the usage patterns of `add_tiles`, `sub_tiles`, `mul_tiles`, and their broadcast variants across the TTNN kernel codebase. The analysis identifies common patterns, groups them by complexity, and assesses the feasibility of replacing raw API calls with the unified `binary_op_helpers.hpp` library.

**Key Findings:**
- 60+ files use `add_tiles`, 49 files use `mul_tiles`, 7 files use `sub_tiles`
- 28 files use row/column broadcast variants
- 32 files use scalar broadcast variants
- Most patterns fall into 8 distinct categories
- ~70% of patterns can be directly replaced with `binary_op_helpers.hpp`
- ~20% require minor library extensions
- ~10% are complex patterns that may not benefit from abstraction

---

## 1. Binary Operation APIs Found

### 1.1 Element-wise Operations
| API | Files | Description |
|-----|-------|-------------|
| `add_tiles(cb_a, cb_b, ia, ib, dst)` | 60 | Element-wise A + B |
| `sub_tiles(cb_a, cb_b, ia, ib, dst)` | 7 | Element-wise A - B |
| `mul_tiles(cb_a, cb_b, ia, ib, dst)` | 49 | Element-wise A × B |

### 1.2 Row Broadcast Operations (B has 1 row, broadcasts across all rows)
| API | Files | Description |
|-----|-------|-------------|
| `add_tiles_bcast_rows(cb_a, cb_b, ia, ib, dst)` | 15 | A[h,w] + B[w] → C[h,w] |
| `mul_tiles_bcast_rows(cb_a, cb_b, ia, ib, dst)` | 12 | A[h,w] × B[w] → C[h,w] |
| `sub_tiles_bcast<BroadcastType::ROW>(...)` | 2 | A[h,w] - B[w] → C[h,w] |

### 1.3 Column Broadcast Operations (B has 1 column, broadcasts across all columns)
| API | Files | Description |
|-----|-------|-------------|
| `add_tiles_bcast_cols(cb_a, cb_b, ia, ib, dst)` | 6 | A[h,w] + B[h] → C[h,w] |
| `sub_tiles_bcast_cols(cb_a, cb_b, ia, ib, dst)` | 18 | A[h,w] - B[h] → C[h,w] |
| `mul_tiles_bcast_cols(cb_a, cb_b, ia, ib, dst)` | 24 | A[h,w] × B[h] → C[h,w] |

### 1.4 Scalar Broadcast Operations (B is single value, broadcasts everywhere)
| API | Files | Description |
|-----|-------|-------------|
| `add_tiles_bcast_scalar(cb_a, cb_b, ia, ib, dst)` | 5 | A[h,w] + B[0,0] → C[h,w] |
| `sub_tiles_bcast_scalar(cb_a, cb_b, ia, ib, dst)` | 15 | A[h,w] - B[0,0] → C[h,w] |
| `mul_tiles_bcast_scalar(cb_a, cb_b, ia, ib, dst)` | 18 | A[h,w] × B[0,0] → C[h,w] |

### 1.5 Initialization Functions
| API | Usage Count | Description |
|-----|-------------|-------------|
| `add_tiles_init(cb_a, cb_b)` | 45 | Initialize add |
| `sub_tiles_init(cb_a, cb_b)` | 5 | Initialize sub |
| `mul_tiles_init(cb_a, cb_b)` | 38 | Initialize mul |
| `add_bcast_rows_init_short(cb_a, cb_b)` | 12 | Initialize row broadcast add |
| `sub_bcast_cols_init_short(cb_a, cb_b)` | 15 | Initialize col broadcast sub |
| `mul_bcast_cols_init_short(cb_a, cb_b)` | 18 | Initialize col broadcast mul |
| `mul_tiles_bcast_scalar_init_short(cb_a, cb_b)` | 12 | Initialize scalar broadcast mul |
| `init_bcast<LLKOP, DIM>(cb_a, cb_b, cb_out)` | 8 | Generic macro-based init |
| `binary_op_init_common(cb_a, cb_b, cb_out)` | 40+ | Common init for all binary ops |

---

## 2. Pattern Categories

### Pattern 1: Simple Streaming (One Tile at a Time)
**Difficulty: EASY** | **Files: ~15** | **Helper Support: FULL**

```cpp
// Example: bcast_h.cpp, bcast_w.cpp, bcast_hw.cpp
for (uint32_t i = 0; i < N; i++) {
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_reserve_back(cb_out, 1);
    acquire_dst();

    add_tiles(cb_in0, cb_in1, 0, 0, 0);  // or BCAST_OP<>
    pack_tile(0, cb_out);

    release_dst();
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    cb_push_back(cb_out, 1);
}
```

**Replacement:**
```cpp
compute_kernel_lib::add<
    compute_kernel_lib::BroadcastDim::NONE,
    compute_kernel_lib::BinaryInputMode::STREAMING>(
    cb_in0, cb_in1, cb_out,
    compute_kernel_lib::BinaryTileShape::block(N));
```

**Files affected:**
- `bcast_h.cpp`, `bcast_w.cpp`, `bcast_hw.cpp` (eltwise/binary)
- `bcast_h.cpp`, `bcast_w.cpp`, `bcast_hw.cpp` (data_movement/bcast)
- `line_reduction.cpp`, `ring_reduction.cpp` (reduce_scatter)

---

### Pattern 2: Batched Processing with DST Accumulation
**Difficulty: EASY** | **Files: ~20** | **Helper Support: FULL**

```cpp
// Example: reduce_nc.cpp
cb_wait_front(cb_in0, N);
cb_wait_front(cb_in1, 1);  // Scalar or persistent
tile_regs_acquire();
for (uint32_t i = 0; i < N; i++) {
    add_tiles(cb_in0, cb_in1, i, 0, dst0);  // Accumulate to same dst
}
tile_regs_commit();
tile_regs_wait();
pack_tile(dst0, cb_out);
tile_regs_release();
```

**Replacement:** Already covered by `BinaryInputMode::STREAMING_BATCHED` or `BinaryAccumulate`.

**Files affected:**
- `reduce_nc.cpp`
- `accumulation_compute.cpp`
- Most CCL reduction kernels

---

### Pattern 3: Grid Processing with Index Computation
**Difficulty: MEDIUM** | **Files: ~25** | **Helper Support: PARTIAL**

```cpp
// Example: layernorm.cpp, groupnorm.cpp
for (uint32_t h = 0; h < Ht; ++h) {
    for (uint32_t w = 0; w < Wt; w += subblock_w) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < subblock_w; ++i) {
            uint32_t idx = w + i + h * Wt;
            mul_tiles(cb_a, cb_b, idx, idx_mask, i);
        }
        tile_regs_commit();
        // pack loop...
    }
}
```

**Current helper support:**
- `BinaryTileShape::grid(Ht, Wt)` handles grid dimensions
- `BinaryTileLayout::with_stride_a(stride)` handles custom strides
- Subblock processing needs manual chunking or DEST_AUTO_LIMIT

**Gap:** Current library handles `dest_limit` internally but doesn't expose subblock configuration.

---

### Pattern 4: Broadcast with Persisted Secondary Input
**Difficulty: EASY-MEDIUM** | **Files: ~30** | **Helper Support: FULL**

```cpp
// Example: layernorm.cpp, softmax.cpp
cb_wait_front(cb_scalar, 1);  // Load once
for (uint32_t h = 0; h < Ht; ++h) {
    for (uint32_t w = 0; w < Wt; ++w) {
        cb_wait_front(cb_data, 1);
        // ...
        mul_tiles_bcast_cols(cb_data, cb_scalar, w, 0, dst);
        // ...
    }
}
// Don't pop cb_scalar - reused
```

**Replacement:** Use `BinaryInputMode::PERSISTENT` for secondary input:
```cpp
compute_kernel_lib::mul<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputMode::STREAMING>(  // Primary streams
    cb_data, cb_scalar, cb_out,
    compute_kernel_lib::BinaryTileShape::grid(Ht, Wt));
// Note: ROW broadcast variant auto-persists secondary input
```

**Files affected:**
- All layernorm variants
- All softmax variants
- Most normalization kernels

---

### Pattern 5: Binary Operation with Post-Op (Chained)
**Difficulty: MEDIUM** | **Files: ~15** | **Helper Support: PARTIAL**

```cpp
// Example: layernorm.cpp (variance calculation)
tile_regs_acquire();
add_tiles(cb_var, cb_eps, 0, 0, dst0);
rsqrt_tile_init<true>();
rsqrt_tile<true>(dst0);
tile_regs_commit();
// pack...
```

**Current helper support:**
- `PostOp` callback parameter exists but is limited
- Only applies to non-streaming modes

**Gap:** PostOp callback works per-tile in DST, not per-batch. Complex post-ops may need explicit handling.

**Workaround:**
```cpp
compute_kernel_lib::add<...>(
    cb_var, cb_eps, cb_out,
    compute_kernel_lib::BinaryTileShape::single(),
    {},  // layout
    {},  // accumulation
    [](uint32_t dst) {
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(dst);
    });
```

---

### Pattern 6: Reconfig-Heavy Operations
**Difficulty: MEDIUM** | **Files: ~20** | **Helper Support: PARTIAL**

```cpp
// Example: layernorm.cpp, groupnorm.cpp
reconfig_data_format(cb_a, cb_b);
pack_reconfig_data_format(cb_out);
add_tiles_init(cb_a, cb_b);
// operation...

reconfig_data_format_srcb(cb_b, cb_c);  // Switch only srcb
mul_tiles_init(cb_a, cb_c);
// next operation...
```

**Current helper support:**
- `BinaryDataFormatReconfig::NONE/INPUT/OUTPUT/BOTH` handles full reconfig
- No support for partial reconfig (`_srcb`, `_srca`)

**Gap:** Needs extension for `reconfig_data_format_srcb()` pattern.

---

### Pattern 7: Moreh-Style Single-Tile Operations
**Difficulty: EASY** | **Files: ~25** | **Helper Support: FULL (via moreh_common)**

```cpp
// Example: moreh_adam.cpp, moreh_adamw.cpp
mul_tiles_to_cb(cb_a, cb_b, cb_out, 0, 0, 0, 0);
add_tiles_to_cb(cb_c, cb_out, cb_out, 0, 0, 0, 1);
sub_tiles_to_cb(cb_one, cb_d, cb_tmp, 0, 0, 0, 0);
```

**Note:** `moreh_common.hpp` already provides `*_tiles_to_cb` helpers that handle:
- `cb_wait_front`, `cb_reserve_back`
- `tile_regs_acquire/commit/wait/release`
- `pack_tile`, `cb_pop_front`, `cb_push_back`
- FP32_DEST_ACC handling

**Relationship:** These are similar to `binary_op_helpers.hpp` but operate on single tiles with explicit pop control. Consider unifying or deprecating one approach.

---

### Pattern 8: Complex Multi-CB Operations
**Difficulty: HARD** | **Files: ~10** | **Helper Support: NONE**

```cpp
// Example: groupnorm.cpp, complex layernorm variants
cb_wait_front(cb_reread_out, N);
cb_reserve_back(cb_reread_write_out, N);
for (uint32_t w = 0; w < block_w; ++w) {
    if (copy_or_add) {
        copy_tile_init(cb_xmm);
        // ...
        copy_tile(cb_xmm, idx, dst0);
    } else {
        add_tiles_init(cb_reread_out, cb_xmm);
        // ...
        add_tiles(cb_reread_out, cb_xmm, idx_reread, idx_xmm, dst0);
    }
    // conditional packing...
}
```

**Gap:** Conditional operation dispatch, multi-CB reread patterns, and complex index management are not suited for the current abstraction.

---

## 3. Kernel Difficulty Ranking

### Tier 1: Direct Replacement (EASY) — ~35 files
Simple patterns that map directly to `binary_op_helpers.hpp`:

| File | Pattern | Replacement Strategy |
|------|---------|---------------------|
| `eltwise/binary/bcast_h.cpp` | Pattern 1 | Direct: `add<ROW, STREAMING>` |
| `eltwise/binary/bcast_w.cpp` | Pattern 1 | Direct: `add<COL, STREAMING>` |
| `eltwise/binary/bcast_hw.cpp` | Pattern 1 | Direct: `add<SCALAR, STREAMING>` |
| `data_movement/bcast/*.cpp` | Pattern 1 | Direct: uses `BCAST_OP` macro |
| `reduce_scatter_minimal_async/line_reduction.cpp` | Pattern 2 | Direct: `add<NONE, STREAMING_BATCHED>` |
| `reduce_scatter_minimal_async/ring_reduction.cpp` | Pattern 2 | Same as above |
| `rotary_embedding_llama.cpp` | Pattern 4 | Already using helper! |
| `accumulation_compute.cpp` | Pattern 2 | Direct: `add/mul<NONE, STREAMING>` with accumulator |

### Tier 2: Minor Adaptation (MEDIUM) — ~30 files
Require small code restructuring or use PostOp callbacks:

| File | Pattern | Adaptation Needed |
|------|---------|-------------------|
| `layernorm.cpp` | Pattern 3, 5 | Use grid() + PostOp for rsqrt |
| `softmax.cpp` | Pattern 4, 5 | Already partially uses `reduce_helpers` |
| `rmsnorm_*.cpp` | Pattern 3, 4 | Grid + persisted scalar |
| `moreh_layer_norm_*.cpp` | Pattern 3, 5 | Multi-phase operations |
| `transformer reduction/*.cpp` | Pattern 2, 4 | CCL reductions |

### Tier 3: Significant Refactoring (HARD) — ~15 files
Complex patterns with conditional logic, multi-CB, or custom indexing:

| File | Pattern | Challenge |
|------|---------|-----------|
| `groupnorm.cpp` | Pattern 8 | Conditional copy vs add, multi-CB |
| `welford_groupnorm.cpp` | Pattern 8 | Welford algorithm state management |
| `moreh_adam.cpp` | Pattern 7 (chain) | Long chains of single-tile ops |
| `moreh_adamw.cpp` | Pattern 7 (chain) | Same as adam |
| `conv_bmm_tilize.cpp` | Mixed | Interleaved with matmul |
| `transformer_attn_matmul.cpp` | Mixed | Fused with matmul |

---

## 4. Current `binary_op_helpers.hpp` Coverage Analysis

### What It Provides
| Feature | Status | Notes |
|---------|--------|-------|
| ADD/SUB/MUL operations | ✅ Full | All three ops supported |
| BroadcastDim::NONE/ROW/COL/SCALAR | ✅ Full | All dimensions supported |
| STREAMING mode | ✅ Full | One-at-a-time processing |
| STREAMING_BATCHED mode | ✅ Full | Batch wait/pop with indexed access |
| PRELOADED mode | ✅ Full | Bulk reserve/push output |
| PERSISTENT mode | ✅ Full | No pop, tiles persist for reuse |
| Data format reconfig | ✅ Full | NONE/INPUT/OUTPUT/BOTH |
| BinaryTileShape | ✅ Full | grid, row, col, single, block |
| BinaryTileLayout | ✅ Full | Custom strides for indexed modes |
| BinaryAccumulate | ✅ Full | Iterative accumulation with reload |
| PostOp callback | ⚠️ Partial | Works in non-STREAMING modes |
| DEST limit auto-detection | ✅ Full | Via `DEST_AUTO_LIMIT` from dest_helpers |
| init flag | ✅ Full | Skip init when chaining ops |

### What's Missing
| Feature | Impact | Recommendation |
|---------|--------|----------------|
| Partial reconfig (`_srcb`, `_srca`) | Medium | Add `BinaryDataFormatReconfig::SRCB_ONLY` |
| Explicit subblock control | Low | Current auto-chunking usually sufficient |
| Conditional op dispatch | Low | Keep manual for complex cases |
| Multi-CB patterns | Low | Not suited for abstraction |
| `sub_bcast_rows` init | Low | Missing in underlying LLK, workaround exists |

---

## 5. Recommendations

### 5.1 Library Extensions Needed

#### Extension 1: Partial Data Format Reconfig
Add support for `reconfig_data_format_srcb()` pattern:

```cpp
enum class BinaryDataFormatReconfig {
    NONE = 0,
    INPUT = 1,      // reconfig_data_format(icb_a, icb_b)
    OUTPUT = 2,     // pack_reconfig_data_format(ocb)
    BOTH = 3,       // Both above
    SRCB_ONLY = 4,  // reconfig_data_format_srcb(old_cb, new_cb)  // NEW
    SRCA_ONLY = 5   // reconfig_data_format_srca(old_cb, new_cb)  // NEW
};
```

#### Extension 2: Explicit B-input Pop Control
Some patterns need to persist B input across multiple operations:

```cpp
template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputMode input_mode = BinaryInputMode::STREAMING,
    bool pop_b = true,  // NEW: Control whether to pop B input
    ...>
ALWI void binary_op(...);
```

#### Extension 3: Multi-tile PostOp with Batching
Current PostOp is per-tile, but some operations want batch-level PostOp:

```cpp
// Current: called per tile
for (uint32_t i = 0; i < chunk_size; ++i) {
    post_op(base_dst + i);
}

// Needed: optional batch-level callback
if constexpr (!std::is_same_v<BatchPostOp, NoOp>) {
    batch_post_op(base_dst, chunk_size);  // e.g., for vectorized sfpu ops
}
```

### 5.2 Migration Strategy

**Phase 1: Easy Wins (Tier 1)**
1. Start with `eltwise/binary/bcast_*.cpp` files — simple, isolated
2. Move to `data_movement/bcast/*.cpp` — similar structure
3. Update CCL reduction kernels — already have pattern separation

**Phase 2: Normalization Kernels (Tier 2)**
1. `layernorm.cpp` — after adding PostOp support verification
2. `softmax.cpp` — already partially migrated via `reduce_helpers`
3. `rmsnorm_*.cpp` — similar to layernorm

**Phase 3: Complex Kernels (Tier 3)**
1. Evaluate case-by-case benefit
2. Some may be better left manual for clarity
3. Consider partial migration (binary op only, leave surrounding logic)

### 5.3 Relationship with `moreh_common.hpp`

The `moreh_common.hpp` file provides similar functionality:
- `add_tiles_to_cb`, `mul_tiles_to_cb`, `sub_tiles_to_cb`
- Various `*_bcast_*_to_cb` variants
- Handles FP32_DEST_ACC via `*_with_dt` variants

**Recommendation:**
1. Mark `moreh_common.hpp` as deprecated for new code
2. Add migration guide from moreh_common to binary_op_helpers
3. Keep moreh_common for backward compatibility in moreh kernels
4. New kernels should use `binary_op_helpers.hpp`

---

## 6. File-by-File Analysis

### 6.1 Already Using `binary_op_helpers.hpp`
| File | Pattern Used |
|------|--------------|
| `rotary_embedding_llama.cpp` | `add<NONE, PERSISTENT>` |
| `softmax.cpp` | Uses `reduce_helpers` which integrates with binary_op |
| `groupnorm.cpp` | Uses `reduce_helpers` for scalar reduce |

### 6.2 Priority Migration Candidates

**High Priority (Simple, High Impact):**
1. `eltwise/binary/bcast_h.cpp` - 44 lines → ~5 lines
2. `eltwise/binary/bcast_w.cpp` - 40 lines → ~5 lines
3. `eltwise/binary/bcast_hw.cpp` - 48 lines → ~5 lines
4. `data_movement/bcast/bcast_h.cpp` - Same structure
5. `data_movement/bcast/bcast_w.cpp` - Same structure

**Medium Priority (Good Simplification):**
1. `reduce_scatter_minimal_async/line_reduction.cpp`
2. `reduce_scatter_minimal_async/ring_reduction.cpp`
3. `accumulation_compute.cpp`
4. `fast_reduce_nc/reduce_nc.cpp`

**Low Priority (Complex, Marginal Benefit):**
1. `groupnorm.cpp` - Very complex, many edge cases
2. `moreh_adam.cpp` - Long chain operations, uses moreh_common
3. `conv_bmm_tilize.cpp` - Interleaved with matmul

---

## 7. Summary Statistics

| Category | Count | Migration Effort |
|----------|-------|------------------|
| Files using add_tiles | 60 | — |
| Files using mul_tiles | 49 | — |
| Files using sub_tiles | 7 | — |
| Files using bcast variants | 47 | — |
| **Tier 1: Easy** | ~35 | 1-2 hours each |
| **Tier 2: Medium** | ~30 | 2-4 hours each |
| **Tier 3: Hard** | ~15 | Case-by-case |
| Already migrated | 3 | — |
| Using moreh_common | ~25 | Separate migration path |

---

## 8. Conclusion

The `binary_op_helpers.hpp` library provides comprehensive coverage for most binary operation patterns in the codebase. With the recommended extensions (partial reconfig, B-input pop control), it can cover approximately 85-90% of current usage patterns.

**Key Actions:**
1. ✅ Library is ready for Tier 1 migrations today
2. ⚠️ Add Extension 1 (SRCB_ONLY reconfig) for Tier 2 migrations
3. 📋 Create migration guide for moreh_common users
4. 🔄 Start with bcast kernels as proof-of-concept
5. 📊 Track metrics: lines of code reduced, pattern coverage

The investment in library adoption pays off through:
- Reduced code duplication
- Automatic DEST limit handling
- Consistent CB management
- Easier maintenance and debugging
- Self-documenting intent via template parameters
