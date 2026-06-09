# Feasibility Analysis: Replacing Binary Operations in Layernorm with Helper Library

## Overview

This document analyzes the feasibility of replacing eltwise binary operations (including broadcast variants) in `layernorm.cpp` with the unified functions from `binary_op_helpers.hpp`.

**Files Analyzed:**
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp`
- `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`

---

## Binary Operations in `layernorm.cpp`

| Location | Operation | Pattern | Current Code |
|----------|-----------|---------|--------------|
| Lines 98-116 | `add_tiles` | Element-wise (FUSE_PRE_ADD) | `add_tiles(cb_in, cb_inb, i, i, i)` |
| Lines 139-149 | `sub_tiles_bcast_cols` | COL broadcast | `sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i)` |
| Lines 160-176 | `mul_tiles` | Self-multiply (square) | `mul_tiles(cb_xmm, cb_xmm, global_i, global_i, i)` |
| Lines 190-191 | `add_tiles` | Single tile | `add_tiles(cb_ex2, cb_eps, 0, 0, dst0)` |
| Lines 216-218 | `mul_tiles_bcast_cols` | COL broadcast | `mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, block.to_global(i), 0, i)` |
| Lines 239-245 | `mul_tiles_bcast_rows` | ROW broadcast | `mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i)` |
| Lines 262-268 | `add_tiles_bcast_rows` | ROW broadcast | `add_tiles_bcast_rows(cb_fusion, cb_beta, i, block.to_global(i), i)` |

---

## Helper Library Capabilities

The `binary_op_helpers.hpp` provides:

### Operation Types
- `BinaryOpType::ADD`, `SUB`, `MUL`

### Broadcast Dimensions
- `BroadcastDim::NONE` - Element-wise
- `BroadcastDim::ROW` - Broadcast single row across all rows
- `BroadcastDim::COL` - Broadcast single column across all columns
- `BroadcastDim::SCALAR` - Broadcast single value

### Input Modes
- `STREAMING` - One-at-a-time, wait/pop each tile
- `STREAMING_BATCHED` - Wait for all in batch, indexed access, pop all
- `PRELOADED` - Tiles already in CB, caller manages wait/pop
- `PERSISTENT` - Wait all upfront, indexed access, NO pop

### Key Design Patterns
1. DEST-based chunking using `DEST_AUTO_LIMIT` (4-16 tiles depending on sync/accumulation mode)
2. Automatic tile_regs_acquire/commit/wait/release
3. Automatic CB reserve/push management per chunk
4. Data format reconfiguration options

---

## Detailed Feasibility by Operation

### 1. FUSE_PRE_ADD (lines 98-116)

**Current Implementation:**
```cpp
for (auto block : generic::blocks(Wt, blk)) {
    cb_wait_front(cb_in, block.full_block_size());
    cb_wait_front(cb_inb, block.full_block_size());
    cb_reserve_back(cb_x, block.full_block_size());
    ACQ();
    for (auto i : block.local()) {
        add_tiles(cb_in, cb_inb, i, i, i);
        pack_tile(i, cb_x);
    }
    REL();
    cb_push_back(cb_x, block.full_block_size());
    cb_pop_front(cb_in, block.full_block_size());
    cb_pop_front(cb_inb, block.full_block_size());
}
```

**Potential Helper Replacement:**
```cpp
compute_kernel_lib::add<BroadcastDim::NONE, BinaryInputMode::STREAMING_BATCHED>(
    cb_in, cb_inb, cb_x, BinaryTileShape::block(Wt));
```

**Issues:**
1. **Chunking granularity mismatch**: Helper uses `effective_dest_limit` (DEST capacity), layernorm uses compile-time `blk` parameter
2. **CB sync pattern**: Layernorm uses `full_block_size()` for CB operations even on partial blocks (for synchronization with reader kernel)
3. **Block iteration**: Helper processes entire shape at once; layernorm iterates block-by-block with explicit CB management

**Feasibility: MEDIUM**

Would require either:
- Modifying helper to support custom block sizes
- Accepting different CB granularity (may break reader/compute synchronization)

---

### 2. x - E[x] (lines 139-149)

**Current Implementation:**
```cpp
cb_reserve_back(cb_xmm, total_buffer_size);  // Reserve ALL upfront
sub_bcast_cols_init_short(cb_x, cb_ex);
for (auto block : generic::blocks(Wt, blk)) {
    ACQ();
    for (auto i : block.local()) {
        sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);
        pack_tile(i, cb_xmm);
    }
    cb_push_back(cb_xmm, block.full_block_size());
    cb_pop_front(cb_x, block.full_block_size());
    REL();
}
cb_pop_front(cb_ex, 1);  // Pop E[x] once at the end
```

**Issues:**
1. **Hybrid reserve/push pattern**: Output CB reserved entirely upfront (`total_buffer_size`), but pushed block-by-block
2. **E[x] persistence**: The broadcast tile (`cb_ex`) stays in CB throughout all blocks, popped once at end
3. Helper's `binary_op_col` either:
   - STREAMING: waits/pops B tile per row
   - PERSISTENT: waits all B tiles upfront, never pops

Neither matches the "wait once, use across blocks, pop once at end" pattern.

**Feasibility: LOW**

The hybrid output CB pattern (reserve-all, push-by-block) is not supported by the helper library.

---

### 3. (x - E[x])² (lines 160-176)

**Current Implementation:**
```cpp
mul_tiles_init(cb_xmm, cb_xmm);
for (auto block : generic::blocks(Wt, blk)) {
    cb_wait_front(cb_xmm, block.start() + block.size());  // Cumulative wait
    cb_reserve_back(cb_xmm2, block.full_block_size());
    ACQ();
    for (auto i : block.local()) {
        const auto global_i = block.to_global(i);
        mul_tiles(cb_xmm, cb_xmm, global_i, global_i, i);  // SAME CB, SAME INDEX
        pack_tile(i, cb_xmm2);
    }
    cb_push_back(cb_xmm2, block.full_block_size());
    REL();
}
```

**Issues:**
1. **Self-multiplication**: Both operands are the same CB with the same tile index
2. Helper library assumes distinct input CBs (`icb_a != icb_b`)
3. **Cumulative wait pattern**: `cb_wait_front(cb_xmm, block.start() + block.size())` - waits for progressively more tiles

**Feasibility: NOT FEASIBLE**

The helper library does not support same-CB multiplication. Would require a new `square()` helper function.

---

### 4. Var[x] + eps (lines 190-191)

**Current Implementation:**
```cpp
cb_wait_front(cb_ex2, 1);
ACQ();
add_tiles_init(cb_ex2, cb_eps);
add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
// Result stays in DST for rsqrt
rsqrt_tile_init<LEGACY_RSQRT>();
rsqrt_tile<LEGACY_RSQRT>(dst0);
pack_tile(dst0, cb_ex2pe);
REL();
```

**Potential Helper:**
```cpp
compute_kernel_lib::add<BroadcastDim::NONE, BinaryInputMode::STREAMING>(
    cb_ex2, cb_eps, cb_temp, BinaryTileShape::single());
```

**Issues:**
1. **DST retention**: Current code keeps result in DST for immediate rsqrt; helper would pack to CB
2. **Flow restructure**: Would need to pack after add, then copy_tile back for rsqrt, or create fused helper

**Feasibility: HIGH (with restructure)**

Simple single-tile operation. Could use helper if willing to add extra copy_tile for rsqrt input.

---

### 5. Multiply by rsqrt (COL broadcast, lines 216-218)

**Current Implementation:**
```cpp
for (auto block : generic::blocks(Wt, blk)) {
    cb_reserve_back(cb_im_or_out, block.full_block_size());
    ACQ();
    mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
    for (auto i : block.local()) {
        mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, block.to_global(i), 0, i);
        pack_tile(i, cb_im_or_out);
    }
    cb_push_back(cb_im_or_out, block.full_block_size());
    REL();
}
```

**Issues:**
1. **Global indexing for A**: `block.to_global(i)` used for cb_xmm access
2. **Block-wise output**: Reserve/push by `full_block_size()`
3. cb_ex2pe (the rsqrt result) is a single tile, accessed with index 0

**Potential Helper:**
```cpp
compute_kernel_lib::mul<BroadcastDim::COL, BinaryInputMode::PRELOADED>(
    cb_xmm, cb_ex2pe, cb_im_or_out, BinaryTileShape::row(Wt));
```

**Feasibility: MEDIUM**

Could work with PRELOADED mode if cb_xmm tiles are pre-waited. The single broadcast tile pattern matches helper's COL broadcast.

---

### 6. Gamma Multiplication (ROW broadcast, lines 239-245)

**Current Implementation:**
```cpp
mul_bcast_rows_init_short(cb_fusion, cb_gamma);
cb_reserve_back(cb_outg, block.full_block_size());
cb_wait_front(cb_gamma, block.start() + block.full_block_size());  // CUMULATIVE WAIT
cb_wait_front(cb_fusion, block.full_block_size());
for (auto i : block.local()) {
    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i);
    //                                         ^-- local  ^-- GLOBAL
    pack_tile(i, cb_outg);
}
cb_pop_front(cb_fusion, block.full_block_size());
// gamma is NOT popped - reused for all NCHt iterations
cb_push_back(cb_outg, block.full_block_size());
```

**Critical Issues:**

1. **Asymmetric indexing**:
   - Input A (cb_fusion): indexed locally with `i` (0 to block.size()-1)
   - Input B (cb_gamma): indexed globally with `block.to_global(i)`

   Helper's ROW broadcast logic assumes B indexed from 0:
   ```cpp
   tile_b = wt_base + wt;  // B always indexed from 0
   ```

2. **Cumulative CB wait**: `cb_wait_front(cb_gamma, block.start() + block.full_block_size())`
   - Waits for progressively more gamma tiles as blocks progress
   - Helper doesn't support this pattern

3. **No-pop for reuse**: Gamma tiles are never popped; reused across all NCHt iterations
   - Helper's PERSISTENT mode doesn't pop, but expects all tiles waited upfront
   - This pattern waits cumulatively per block

**Feasibility: NOT FEASIBLE**

Fundamental pattern mismatch. The asymmetric indexing (local A, global B) and cumulative wait without pop are not supported.

---

### 7. Beta Addition (ROW broadcast, lines 262-268)

**Current Implementation:**
```cpp
add_bcast_rows_init_short(cb_fusion, cb_beta);
cb_reserve_back(cb_out, block.full_block_size());
cb_wait_front(cb_beta, block.start() + block.full_block_size());  // CUMULATIVE WAIT
cb_wait_front(cb_fusion, block.full_block_size());
for (auto i : block.local()) {
    add_tiles_bcast_rows(cb_fusion, cb_beta, i, block.to_global(i), i);
    pack_tile(i, cb_out);
}
cb_pop_front(cb_fusion, block.full_block_size());
// beta is NOT popped - reused for all NCHt iterations
cb_push_back(cb_out, block.full_block_size());
```

**Issues:** Same as gamma multiplication.

**Feasibility: NOT FEASIBLE**

---

## Summary Table

| Operation | Feasibility | Key Blocker |
|-----------|-------------|-------------|
| FUSE_PRE_ADD (add) | **Medium** | Different chunking granularity (DEST vs blk) |
| x - E[x] (sub COL bcast) | **Low** | Hybrid reserve-all + push-by-block pattern |
| (x - E[x])² (self-mul) | **Not Feasible** | Same-CB multiplication not supported |
| Var[x] + eps (add) | **High** | Simple single-tile; needs flow restructure for rsqrt |
| Multiply by rsqrt (mul COL bcast) | **Medium** | Possible with PRELOADED mode |
| Gamma (mul ROW bcast) | **Not Feasible** | Asymmetric indexing, cumulative wait, no-pop reuse |
| Beta (add ROW bcast) | **Not Feasible** | Asymmetric indexing, cumulative wait, no-pop reuse |

---

## Root Cause Analysis

The fundamental mismatch stems from different design philosophies:

### Layernorm Kernel Design
1. **Block-based streaming**: Processes tiles in blocks of `blk` (compile-time parameter) synchronized with reader kernel
2. **Cumulative CB waits**: `cb_wait_front(cb, block.start() + block.size())` for gradual tile availability
3. **Asymmetric broadcast patterns**: Data tiles indexed locally, broadcast tiles (gamma/beta) indexed globally
4. **Persistent broadcast tiles**: Gamma/beta never popped, reused across all NCHt iterations
5. **Full-block CB sync**: Uses `full_block_size()` for reserve/push even on partial blocks

### Helper Library Design
1. **DEST-capacity chunking**: Processes in chunks of `DEST_AUTO_LIMIT` (4-16 tiles)
2. **Symmetric input modes**: Both inputs follow same wait/pop pattern
3. **Shape-based processing**: Processes entire `BinaryTileShape` in one call
4. **Standard CB patterns**: Reserve/push per chunk or bulk reserve/push

---

## Recommendations

### Option 1: Do Not Replace (Recommended for Now)

Keep the current layernorm implementation. The specialized patterns are tightly coupled with:
- Reader kernel synchronization (block-based CB management)
- Memory efficiency (gamma/beta reuse across NCHt)
- Compute flow (cumulative waits for streaming input)

### Option 2: Extend Helper Library

Add new capabilities to support normalization patterns:

```cpp
// New input mode for block-based streaming
enum class BinaryInputMode {
    // ... existing modes ...
    BLOCK_STREAMING,  // Process in blocks of specified size
};

// New configuration for asymmetric indexing
struct BinaryTileLayout {
    // ... existing fields ...
    bool global_index_b = false;  // Index B globally instead of locally
};

// New square() helper for self-multiplication
template <BinaryInputMode input_mode = BinaryInputMode::STREAMING>
ALWI void square(uint32_t icb, uint32_t ocb, BinaryTileShape shape);

// New option for persistent broadcast tiles
enum class BroadcastTilePersistence {
    POP_AFTER_USE,      // Pop broadcast tiles after operation
    PERSIST_NO_POP,     // Keep broadcast tiles for reuse
    CUMULATIVE_WAIT,    // Wait cumulatively per block, no pop
};
```

### Option 3: Create Specialized Layernorm Helpers

Create a `layernorm_compute_helpers.hpp` with operations tailored for normalization:

```cpp
namespace layernorm_helpers {

// Subtract mean with COL broadcast, handling cumulative waits
void subtract_mean(uint32_t cb_x, uint32_t cb_mean, uint32_t cb_out,
                   uint32_t Wt, uint32_t blk);

// Square operation (x * x) with cumulative input wait
void square(uint32_t cb_in, uint32_t cb_out, uint32_t Wt, uint32_t blk);

// Scale by rsqrt with COL broadcast
void scale_by_rsqrt(uint32_t cb_x, uint32_t cb_rsqrt, uint32_t cb_out,
                    uint32_t Wt, uint32_t blk);

// Apply gamma/beta with ROW broadcast and persistence
void apply_gamma_beta(uint32_t cb_in, uint32_t cb_gamma, uint32_t cb_beta,
                      uint32_t cb_out, uint32_t Wt, uint32_t blk,
                      bool do_gamma, bool do_beta);

}  // namespace layernorm_helpers
```

---

## Conclusion

**Partial replacement is feasible** for simple element-wise operations (FUSE_PRE_ADD, Var[x]+eps), but the **core normalization operations (gamma/beta application) are not compatible** with the current helper library design.

The layernorm kernel's specialized patterns—particularly the asymmetric indexing for ROW broadcast and the cumulative CB wait patterns—are optimized for the specific data flow of normalization operations and don't align with the helper library's more general-purpose abstractions.

If helper adoption is desired, **Option 3 (specialized layernorm helpers)** is recommended as it would encapsulate the complex patterns while providing a cleaner interface for future normalization kernel development.
