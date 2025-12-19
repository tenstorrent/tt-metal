// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file dest_helpers.hpp
 * @brief DEST register capacity detection utilities
 *
 * Provides automatic detection of DEST register capacity based on JIT-generated headers.
 * This is shared by reduce_helpers.hpp, untilize_helpers.hpp, and other kernel libraries
 * that need to know the DEST register limits for proper chunking/batching.
 *
 * DEST register capacity depends on:
 * 1. Sync mode (Half vs Full) - determined by DST_SYNC_MODE
 * 2. Accumulation mode (16-bit vs 32-bit) - determined by DST_ACCUM_MODE
 *
 * Capacity table:
 * - SyncFull + 16-bit (DST_ACCUM_MODE=false): 16 tiles
 * - SyncFull + 32-bit (DST_ACCUM_MODE=true):  8 tiles
 * - SyncHalf + 16-bit (DST_ACCUM_MODE=false): 8 tiles
 * - SyncHalf + 32-bit (DST_ACCUM_MODE=true):  4 tiles
 */

namespace compute_kernel_lib {

// =============================================================================
// DEST Register Capacity - Automatic Detection
// =============================================================================

// DST_SYNC_MODE is defined in JIT-generated chlkc_dst_sync_mode.h
// DST_ACCUM_MODE is defined in JIT-generated chlkc_dst_accum_mode.h
// Both are included via chlkc_list.h -> common_globals.h

/**
 * @brief Get the DEST register capacity based on current sync and accumulation modes
 *
 * @return Number of tiles that can be held in DEST registers
 */
constexpr uint32_t get_dest_limit() {
#if defined(DST_SYNC_MODE) && defined(DST_ACCUM_MODE)
    // Automatically detect from JIT-generated header files
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        // Full-sync mode
        if constexpr (DST_ACCUM_MODE) {
            return 8;  // 32-bit accumulation
        } else {
            return 16;  // 16-bit accumulation
        }
    } else {
        // Half-sync mode
        if constexpr (DST_ACCUM_MODE) {
            return 4;  // 32-bit accumulation
        } else {
            return 8;  // 16-bit accumulation
        }
    }
#else
    // Fallback if JIT headers not defined (shouldn't happen in real kernels)
    // Use conservative half-sync 16-bit value
    return 8;
#endif
}

// Auto-detected default dest limit based on current sync and accumulation modes
constexpr uint32_t DEST_AUTO_LIMIT = get_dest_limit();

}  // namespace compute_kernel_lib
