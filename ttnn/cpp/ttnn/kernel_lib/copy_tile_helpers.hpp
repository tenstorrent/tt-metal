// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

/**
 * @file copy_tile_helpers.hpp
 * @brief CB-to-CB tile copy helper for compute kernels
 *
 * Wraps the common pattern of copying tiles from one circular buffer to another
 * via DST registers (unpack → datacopy → pack). Handles DEST register
 * management, circular buffer synchronization, and data format reconfiguration.
 *
 * PREREQUISITE: Call compute_kernel_hw_startup(input_cb, output_cb) at the
 * start of your kernel before using this function.
 *
 * ## What this replaces
 *
 * Without this helper, copying N tiles from cb_in to cb_out requires:
 *
 *   copy_tile_to_dst_init_short(cb_in);
 *   for (uint32_t i = 0; i < num_tiles; ++i) {
 *       tile_regs_acquire();
 *       cb_wait_front(cb_in, 1);
 *       copy_tile(cb_in, 0, 0);
 *       cb_pop_front(cb_in, 1);
 *       tile_regs_commit();
 *       tile_regs_wait();
 *       cb_reserve_back(cb_out, 1);
 *       pack_tile(0, cb_out);
 *       cb_push_back(cb_out, 1);
 *       tile_regs_release();
 *   }
 *
 * With this helper:
 *
 *   compute_kernel_lib::copy_tiles(cb_in, cb_out, num_tiles);
 *
 * ── Template Parameters (compile-time) ──────────────────────────────────────
 *
 *   input_policy   — Whether to wait/pop input tiles or assume caller manages them.
 *                     CopyInputPolicy::WaitAndPop (default): wait and pop per tile.
 *                     CopyInputPolicy::NoWaitNoPop: caller manages CB synchronization.
 *   reconfig_mode  — Data format reconfiguration mode (default: INPUT_AND_OUTPUT).
 *   PostOp         — Post-copy callback type (default: NoOp).
 *
 * ── Runtime Parameters ──────────────────────────────────────────────────────
 *
 *   input_cb    — Input circular buffer index (0–31).
 *   output_cb   — Output circular buffer index (0–31, must differ from input_cb).
 *   num_tiles   — Number of tiles to copy.
 *   post_op     — Optional callback invoked on each tile while it sits in DST
 *                 (receives dst_idx). Use for applying SFPU ops before packing.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   // Hardware init — must come first
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // 1. Basic copy (most common — stream N tiles from cb_in to cb_out)
 *   copy_tiles(cb_in, cb_out, num_tiles);
 *
 *   // 2. Copy with post-op — apply exp to each tile before packing
 *   copy_tiles(cb_in, cb_out, num_tiles,
 *       [](uint32_t dst_idx) {
 *           exp_tile_init();
 *           exp_tile(dst_idx);
 *       });
 *
 *   // 3. NoWaitNoPop — caller manages input CB synchronization
 *   cb_wait_front(cb_in, total_tiles);
 *   copy_tiles<CopyInputPolicy::NoWaitNoPop>(cb_in, cb_out, total_tiles);
 *   cb_pop_front(cb_in, total_tiles);
 *
 *   // 4. Skip data format reconfiguration (first op or formats already match)
 *   copy_tiles<CopyInputPolicy::WaitAndPop, CopyDataFormatReconfig::NONE>(
 *       cb_in, cb_out, num_tiles);
 */

namespace compute_kernel_lib {

// =============================================================================
// Enums
// =============================================================================

/**
 * @brief Input synchronization and consumption policy for copy operations
 *
 * Controls when to wait for input tiles and whether to pop them after copying:
 * - WaitAndPop: Wait for and pop one tile at a time (streaming, safe for any CB size)
 * - NoWaitNoPop: Caller manages wait/pop externally (tiles already in CB)
 *
 * WARNING - NoWaitNoPop:
 * Only use when paired with explicit cb_wait_front() before the copy, or when
 * tiles are guaranteed to be available (e.g., sharded tensors pre-loaded in CB).
 */
enum class CopyInputPolicy {
    WaitAndPop,  // Wait for and pop one tile at a time (default, streaming)
    NoWaitNoPop  // Caller manages wait/pop externally
};

/**
 * @brief Data format reconfiguration mode for copy operations
 *
 * Controls whether unpacker (input) and/or packer (output) are reconfigured:
 * - NONE: Skip all reconfiguration (copy is first op or formats already match)
 * - INPUT: Reconfigure unpacker only (input format changed)
 * - OUTPUT: Reconfigure packer only (output format changed)
 * - INPUT_AND_OUTPUT: Reconfigure both unpacker and packer (default, safest option)
 */
enum class CopyDataFormatReconfig { NONE = 0, INPUT = 1, OUTPUT = 2, INPUT_AND_OUTPUT = 3 };

// =============================================================================
// Main API
// =============================================================================

/**
 * @brief Copy tiles from input CB to output CB via DST registers
 *
 * Processes tiles one at a time: unpack → optional post_op in DST → pack.
 * Automatically handles DEST register acquire/commit/wait/release and
 * CB reserve/push for each tile.
 *
 * @tparam input_policy   Input handling policy (default: WaitAndPop)
 * @tparam reconfig_mode  Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam PostOp         Post-copy callback type (default: NoOp)
 *
 * @param input_cb   Input circular buffer
 * @param output_cb  Output circular buffer
 * @param num_tiles  Number of tiles to copy
 * @param post_op    Callback invoked per tile in DST before packing (receives dst_idx)
 */
template <
    CopyInputPolicy input_policy = CopyInputPolicy::WaitAndPop,
    CopyDataFormatReconfig reconfig_mode = CopyDataFormatReconfig::INPUT_AND_OUTPUT,
    typename PostOp = NoOp>
ALWI void copy_tiles(uint32_t input_cb, uint32_t output_cb, uint32_t num_tiles, PostOp post_op = {});

}  // namespace compute_kernel_lib

#include "copy_tile_helpers.inl"
