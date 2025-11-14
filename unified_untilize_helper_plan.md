# Unified Untilize Helper Library - Implementation Plan

## Overview

Create a **single unified untilize function** that replaces both `untilize()` and `pack_untilize()`. The function takes tile width as a parameter and automatically dispatches to the appropriate implementation based on whether the width fits in the DEST register.

**Goal:** One function, one API - user provides width, we handle the rest.

---

## Current State: Two Separate Functions

### Standard Untilize (`untilize_helpers.h`)
```cpp
template <bool init = true, bool uninit = true, bool wait_upfront = false>
ALWI void untilize(uint32_t icb, uint32_t block_w, uint32_t ocb,
                   uint32_t num_blocks, uint32_t total_tiles = 0);
```
- Takes `block_w` (width in tiles)
- **Limitation:** `block_w` must fit in DEST (≤8 half-sync, ≤4 32-bit)

### Pack Untilize (`pack_untilize_helpers.h`)
```cpp
template <uint32_t block_ct_dim, uint32_t full_ct_dim = block_ct_dim,
          bool init = true, bool uninit = true>
ALWI void pack_untilize(uint32_t icb, uint32_t ocb, uint32_t num_rows,
                        uint32_t block_rt_dim = 1);
```
- Takes `block_ct_dim` (width in tiles) as template param
- Works for any width by chunking into DEST-sized blocks

**Both already take width as the key parameter!**

---

## Proposed: Single Unified Function

### New API
```cpp
template <uint32_t tile_width, uint32_t dest_limit = 8, bool init = true, bool uninit = true>
ALWI void untilize(uint32_t icb, uint32_t ocb, uint32_t num_rows, uint32_t block_rt_dim = 1);
```

### Dispatch Logic
```
if (tile_width <= dest_limit):
    use standard untilize implementation
else:
    use pack_untilize implementation (with auto-computed block width)
```

---

## Complete Header Design

### File: `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/cb_api.h"

/**
 * @file untilize_helpers.h
 * @brief Single unified untilize function with automatic dispatch
 *
 * Provides ONE function that handles all untilize operations:
 * - Small widths (≤ DEST limit): Uses standard untilize
 * - Large widths (> DEST limit): Uses pack_untilize with optimal blocking
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"
 *
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // Small width - automatically uses standard untilize
 *   compute_kernel_lib::untilize<4>(cb_in, cb_out, num_rows);
 *
 *   // Large width - automatically uses pack_untilize
 *   compute_kernel_lib::untilize<32>(cb_in, cb_out, num_rows);
 */

namespace compute_kernel_lib {

// =============================================================================
// DEST Register Capacity Constants
// =============================================================================

constexpr uint32_t DEST_HALF_SYNC_LIMIT = 8;   // 8 tiles in half-sync mode
constexpr uint32_t DEST_32BIT_MODE_LIMIT = 4;  // 4 tiles in 32-bit mode

// =============================================================================
// Helper: Compute Optimal Block Width for Pack Path
// =============================================================================

/**
 * @brief Compute optimal block width for pack_untilize
 *
 * Returns the largest value that:
 * 1. Divides tile_width evenly
 * 2. Does not exceed dest_limit
 */
template <uint32_t tile_width, uint32_t dest_limit>
constexpr uint32_t compute_block_width() {
    // Find largest divisor of tile_width that fits in DEST
    for (uint32_t w = dest_limit; w >= 1; --w) {
        if (tile_width % w == 0) {
            return w;
        }
    }
    return 1;
}

// =============================================================================
// Single Unified Untilize Function
// =============================================================================

/**
 * @brief Unified untilize function - automatically dispatches based on width
 *
 * This is the ONLY untilize function you need. Provide the tile width,
 * and the optimal implementation is selected at compile time:
 *
 * - tile_width <= dest_limit: Standard untilize (direct processing)
 * - tile_width > dest_limit: Pack untilize (block-based assembly)
 *
 * @tparam tile_width Width in tiles (number of tiles per row)
 * @tparam dest_limit DEST register capacity (default: 8 for half-sync mode)
 * @tparam init Call init before processing (default: true)
 * @tparam uninit Call uninit after processing (default: true)
 *
 * @param icb Input circular buffer ID (tiled data)
 * @param ocb Output circular buffer ID (row-major data)
 * @param num_rows Number of rows to process
 * @param block_rt_dim Row height per block in tiles (default: 1)
 *                     Used for multi-row blocks in pack path
 *
 * @example
 *   // Width 4 - uses standard untilize (4 <= 8)
 *   untilize<4>(cb_in, cb_out, 10);
 *
 * @example
 *   // Width 32 - uses pack_untilize with block_w=8 (32 > 8)
 *   untilize<32>(cb_in, cb_out, 10);
 *
 * @example
 *   // 32-bit mode - lower DEST limit
 *   untilize<6, DEST_32BIT_MODE_LIMIT>(cb_in, cb_out, 10);
 *
 * @example
 *   // Skip init for reinit scenarios
 *   untilize<8, DEST_HALF_SYNC_LIMIT, false>(cb_in, cb_out, 10);
 */
template <
    uint32_t tile_width,
    uint32_t dest_limit = DEST_HALF_SYNC_LIMIT,
    bool init = true,
    bool uninit = true>
ALWI void untilize(
    uint32_t icb,
    uint32_t ocb,
    uint32_t num_rows,
    uint32_t block_rt_dim = 1)
{
    if constexpr (tile_width <= dest_limit) {
        // =================================================================
        // STANDARD UNTILIZE PATH
        // Width fits in DEST - process directly
        // =================================================================

        if constexpr (init) {
            untilize_init(icb);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_wait_front(icb, tile_width);
            cb_reserve_back(ocb, tile_width);
            untilize_block(icb, tile_width, ocb);
            cb_push_back(ocb, tile_width);
            cb_pop_front(icb, tile_width);
        }

        if constexpr (uninit) {
            untilize_uninit(icb);
        }

    } else {
        // =================================================================
        // PACK UNTILIZE PATH
        // Width exceeds DEST - use block assembly
        // =================================================================

        constexpr uint32_t block_w = compute_block_width<tile_width, dest_limit>();
        constexpr uint32_t num_blocks = tile_width / block_w;

        if constexpr (init) {
            pack_untilize_init<block_w, tile_width>(icb, ocb);
        }

        const uint32_t tiles_per_block = block_w * block_rt_dim;
        const uint32_t tiles_per_row = tile_width * block_rt_dim;

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_reserve_back(ocb, tiles_per_row);

            for (uint32_t b = 0; b < num_blocks; ++b) {
                cb_wait_front(icb, tiles_per_block);
                pack_untilize_block<block_w, tile_width>(icb, block_rt_dim, ocb, b);
                cb_pop_front(icb, tiles_per_block);
            }

            cb_push_back(ocb, tiles_per_row);
        }

        if constexpr (uninit) {
            pack_untilize_uninit(ocb);
        }
    }
}

// =============================================================================
// Variant: Wait-Upfront Pattern (for GroupNorm, etc.)
// =============================================================================

/**
 * @brief Untilize with wait-upfront pattern
 *
 * Waits for all input tiles before processing. Used by GroupNorm and similar
 * operations where all data must be available upfront.
 *
 * @tparam tile_width Width in tiles
 * @tparam dest_limit DEST capacity (default: 8)
 * @tparam init Call init (default: true)
 * @tparam uninit Call uninit (default: true)
 *
 * @param icb Input CB
 * @param ocb Output CB
 * @param num_rows Number of rows to process
 * @param total_tiles Total tiles to wait for (if 0, computes as tile_width * num_rows)
 */
template <
    uint32_t tile_width,
    uint32_t dest_limit = DEST_HALF_SYNC_LIMIT,
    bool init = true,
    bool uninit = true>
ALWI void untilize_wait_upfront(
    uint32_t icb,
    uint32_t ocb,
    uint32_t num_rows,
    uint32_t total_tiles = 0)
{
    static_assert(tile_width <= dest_limit,
        "wait_upfront pattern only supported for widths that fit in DEST");

    if constexpr (init) {
        untilize_init(icb);
    }

    uint32_t wait_amount = (total_tiles > 0) ? total_tiles : (tile_width * num_rows);
    cb_wait_front(icb, wait_amount);

    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_reserve_back(ocb, tile_width);
        untilize_block(icb, tile_width, ocb);
        cb_push_back(ocb, tile_width);
        cb_pop_front(icb, tile_width);
    }

    if constexpr (uninit) {
        untilize_uninit(icb);
    }
}

}  // namespace compute_kernel_lib
```

---

## API Comparison: Before vs After

### Before (Two Functions)
```cpp
// Standard untilize - must know width fits in DEST
compute_kernel_lib::untilize(cb_in, block_w, cb_out, num_blocks);

// Pack untilize - must know width exceeds DEST
compute_kernel_lib::pack_untilize<block_ct_dim, full_ct_dim>(cb_in, cb_out, num_rows);

// SDPA: Manual conditional
if constexpr (use_pack_untilize) {
    compute_kernel_lib::pack_untilize<Wt>(cb_in, cb_out, 1);
} else {
    compute_kernel_lib::untilize(cb_in, Wt, cb_out, Wt);
}
```

### After (Single Function)
```cpp
// Just provide width - dispatch is automatic
compute_kernel_lib::untilize<tile_width>(cb_in, cb_out, num_rows);

// SDPA: No more conditional!
compute_kernel_lib::untilize<Wt>(cb_in, cb_out, 1);
```

---

## Dispatch Examples

| `tile_width` | `dest_limit` | Implementation | Block Config |
|--------------|--------------|----------------|--------------|
| 4 | 8 | Standard | N/A |
| 8 | 8 | Standard | N/A |
| 9 | 8 | Pack | block_w=3, 3 blocks |
| 16 | 8 | Pack | block_w=8, 2 blocks |
| 32 | 8 | Pack | block_w=8, 4 blocks |
| 15 | 8 | Pack | block_w=5, 3 blocks |
| 6 | 4 (32-bit) | Pack | block_w=3, 2 blocks |

---

## Migration Guide

### Kernels Using Old `untilize()`

```cpp
// Before:
compute_kernel_lib::untilize(cb_in, block_w, cb_out, num_blocks);

// After (if block_w is compile-time constant):
compute_kernel_lib::untilize<block_w>(cb_in, cb_out, num_blocks);

// After (wait_upfront pattern):
compute_kernel_lib::untilize_wait_upfront<tile_width>(cb_in, cb_out, num_rows, total_tiles);
```

### Kernels Using Old `pack_untilize()`

```cpp
// Before:
compute_kernel_lib::pack_untilize<block_ct_dim, full_ct_dim>(cb_in, cb_out, num_rows);

// After (use full_ct_dim as tile_width):
compute_kernel_lib::untilize<full_ct_dim>(cb_in, cb_out, num_rows);

// Note: block_ct_dim is now auto-computed, no need to specify
```

### SDPA Kernel (Conditional Removal)

```cpp
// Before:
if constexpr (use_pack_untilize) {
    compute_kernel_lib::pack_untilize<out_chunk_tiles>(cb_in, cb_out, 1);
} else {
    compute_kernel_lib::untilize<true, true>(cb_in, out_chunk_tiles, cb_out, out_chunk_tiles);
}

// After:
compute_kernel_lib::untilize<out_chunk_tiles>(cb_in, cb_out, 1);
// Can remove use_pack_untilize compile-time flag entirely!
```

---

## Implementation Plan

### Phase 1: Update Header
- [ ] Replace contents of `untilize_helpers.h` with unified implementation
- [ ] Add `#include "compute_kernel_api/pack_untilize.h"`
- [ ] Add DEST limit constants
- [ ] Add `compute_block_width()` helper
- [ ] Implement unified `untilize<tile_width>()` function
- [ ] Implement `untilize_wait_upfront<tile_width>()` for GroupNorm pattern
- [ ] Verify compilation

### Phase 2: Deprecate Pack Helper
- [ ] Update `pack_untilize_helpers.h`:
```cpp
#pragma once
#warning "pack_untilize_helpers.h is deprecated. Use untilize_helpers.h with untilize<tile_width>() instead."
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"

namespace compute_kernel_lib {
// Backward compatibility alias
template <uint32_t block_ct_dim, uint32_t full_ct_dim = block_ct_dim,
          bool init = true, bool uninit = true>
ALWI void pack_untilize(uint32_t icb, uint32_t ocb, uint32_t num_rows,
                        uint32_t block_rt_dim = 1) {
    untilize<full_ct_dim, DEST_HALF_SYNC_LIMIT, init, uninit>(icb, ocb, num_rows, block_rt_dim);
}
}
```

### Phase 3: Migrate Standard Untilize Kernels
Kernels with compile-time width:
- [ ] `untilize.cpp` - if width is compile-time, migrate
- [ ] `untilize_w.cpp` - if width is compile-time, migrate
- [ ] `untilize_wh.cpp` - if width is compile-time, migrate
- [ ] `rotary_embedding.cpp`

GroupNorm kernels (use `untilize_wait_upfront`):
- [ ] `groupnorm.cpp`
- [ ] `groupnorm_sharded_v2.cpp`
- [ ] `welford_groupnorm.cpp`
- [ ] `welford_groupnorm_sharded_v2.cpp`

Other:
- [ ] `conv_bmm_tilize.cpp`
- [ ] `conv3d compute.cpp`
- [ ] `transpose_wh_rm.cpp`
- [ ] `ssm_prefix_scan.cpp`

### Phase 4: Migrate Pack Untilize Kernels
- [ ] `pack_untilize.cpp`
- [ ] `pack_untilize_wh.cpp`
- [ ] `pack_untilize_variable_num_blocks.cpp`
- [ ] `kv_cache/update_cache.cpp`
- [ ] `paged_fused_update_cache.cpp`
- [ ] `paged_row_major_fused_update_cache.cpp`
- [ ] `experimental/paged_cache/update_cache.cpp`

### Phase 5: Simplify SDPA
- [ ] `sdpa_flash_decode.cpp` - replace conditional with single `untilize<>()` call
- [ ] Consider removing `use_pack_untilize` flag if no longer needed

### Incompatible (Skip)
- `sliding_window/halo/pack_untilize.cpp` - alternating output CBs

---

## Final API Summary

| Function | Purpose |
|----------|---------|
| `untilize<tile_width>(icb, ocb, num_rows)` | **Primary API** - auto dispatch |
| `untilize<tile_width, dest_limit>(...)` | Custom DEST limit (32-bit mode) |
| `untilize<tile_width, dest_limit, init, uninit>(...)` | Control init/uninit |
| `untilize_wait_upfront<tile_width>(icb, ocb, num_rows, total_tiles)` | GroupNorm pattern |

---

## Template Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tile_width` | Width in tiles (tiles per row) | **Required** |
| `dest_limit` | DEST register capacity | 8 (half-sync) |
| `init` | Call init function | true |
| `uninit` | Call uninit function | true |

---

## Success Criteria

- [ ] Single `untilize<tile_width>()` function replaces both old APIs
- [ ] Automatic dispatch based on width vs DEST limit
- [ ] All existing kernels migrated to new API
- [ ] SDPA conditional eliminated
- [ ] `pack_untilize_helpers.h` deprecated with backward compat
- [ ] All tests pass
- [ ] No performance regression
- [ ] Zero runtime overhead (compile-time dispatch)

---

## Benefits

1. **One function to learn** - no more choosing between untilize/pack_untilize
2. **Automatic optimization** - compiler picks best implementation
3. **Simpler kernel code** - remove conditionals like SDPA
4. **Width is the only input** - user doesn't need to understand DEST limits
5. **Zero overhead** - all dispatch at compile time

---

## References

- Current helpers: `untilize_helpers.h`, `pack_untilize_helpers.h`
- API headers: `compute_kernel_api/untilize.h`, `compute_kernel_api/pack_untilize.h`
- Uninit deprecation: tt-metal#22904
