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
 * @return true if FP32 accumulation is enabled via any source
 *
 * For compute kernels (TRISC0/1/2): Uses DST_ACCUM_MODE constexpr bool from JIT-generated
 * chlkc_dst_accum_mode.h (included via chlkc_list.h before user kernel).
 *
 * For data movement kernels: Uses ENABLE_FP32_DEST_ACC macro define.
 */
constexpr bool get_fp32_dest_acc_enabled() {
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK)
    // Compute kernel (TRISC) - DST_ACCUM_MODE is a constexpr bool from JIT header
    return DST_ACCUM_MODE;
#elif defined(ENABLE_FP32_DEST_ACC)
    // Data movement kernel - use macro define
    return (ENABLE_FP32_DEST_ACC == 1);
#else
    static_assert(false, "ENABLE_FP32_DEST_ACC must be defined for data movement kernels");
#endif
}

/**
 * @brief Detect if full destination sync mode is enabled
 *
 * @return true if full-sync mode, false if half-sync mode
 *
 * Checks multiple sources in order of precedence:
 * 1. DST_SYNC_MODE (JIT-generated macro from ComputeConfig.dst_full_sync_en)
 * 2. DST_SYNC_FULL (manual define for dataflow kernels)
 */
constexpr bool get_dst_full_sync_enabled() {
#if defined(DST_SYNC_MODE)
    return (DST_SYNC_MODE == DstSync::SyncFull);  // JIT-generated (preferred)
#elif defined(DST_SYNC_FULL)
    return (DST_SYNC_FULL == 1);  // Manual define fallback
#else
    static_assert(false, "DST_SYNC_MODE or DST_SYNC_FULL must be defined");
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
    // Both functions encapsulate JIT vs manual define logic
    constexpr bool is_fp32_accum = get_fp32_dest_acc_enabled();
    constexpr bool is_full_sync = get_dst_full_sync_enabled();

    // Return DEST capacity based on sync and accumulation modes
    if constexpr (is_full_sync) {
        return is_fp32_accum ? 8 : 16;  // Full-sync: 8 (fp32) or 16 (fp16) tiles
    } else {
        return is_fp32_accum ? 4 : 8;  // Half-sync: 4 (fp32) or 8 (fp16) tiles
    }
}

// Auto-detected default dest limit based on current sync and accumulation modes
constexpr uint32_t DEST_AUTO_LIMIT = get_dest_limit();

}  // namespace compute_kernel_lib
