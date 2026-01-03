// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file dest_helpers.hpp
 * @brief DEST register capacity and accumulation mode detection utilities
 *
 * Provides automatic detection of DEST-related configurations based on JIT-generated headers.
 * This is shared by reduce_helpers.hpp, untilize_helpers.hpp, and other kernel libraries
 * that need to know DEST register limits and accumulation modes.
 *
 * Features:
 * 1. DEST register capacity detection (get_dest_limit(), DEST_AUTO_LIMIT)
 * 2. FP32 accumulation mode detection (get_fp32_dest_acc_enabled())
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
 * @brief Detect if FP32 destination accumulation is enabled
 *
 * @return true if ENABLE_FP32_DEST_ACC is defined and set to 1, false otherwise
 *
 * This is auto-detected from the ENABLE_FP32_DEST_ACC compile-time define
 * that is typically set by program factories based on the fp32_dest_acc_en flag.
 */
constexpr bool get_fp32_dest_acc_enabled() {
#ifdef ENABLE_FP32_DEST_ACC
    return (ENABLE_FP32_DEST_ACC == 1);
#else
    return false;
#endif
}

/**
 * @brief Get the DEST register capacity based on current sync and accumulation modes
 *
 * @return Number of tiles that can be held in DEST registers
 *
 * Considers both DST_ACCUM_MODE (from JIT headers) and ENABLE_FP32_DEST_ACC (from defines).
 * If either indicates FP32 accumulation, uses the reduced capacity for 32-bit mode.
 */
constexpr uint32_t get_dest_limit() {
    // Check if FP32 accumulation is enabled via either method
    constexpr bool is_fp32_accum = get_fp32_dest_acc_enabled()
#if defined(DST_ACCUM_MODE)
                                   || DST_ACCUM_MODE
#endif
        ;

#if defined(DST_SYNC_MODE)
    // Automatically detect sync mode from JIT-generated header files
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        // Full-sync mode
        return is_fp32_accum ? 8 : 16;  // 32-bit: 8 tiles, 16-bit: 16 tiles
    } else {
        // Half-sync mode
        return is_fp32_accum ? 4 : 8;  // 32-bit: 4 tiles, 16-bit: 8 tiles
    }
#else
    // Fallback if JIT headers not defined (shouldn't happen in real kernels)
    // Use conservative half-sync value
    return is_fp32_accum ? 4 : 8;
#endif
}

// Auto-detected default dest limit based on current sync and accumulation modes
constexpr uint32_t DEST_AUTO_LIMIT = get_dest_limit();

}  // namespace compute_kernel_lib
