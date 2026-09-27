// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace compute_kernel_lib {

/**
 * Convert a subblock-ordered tiled block to untilized row-major output.
 * Tile-row-major input should use the standard untilize helper instead.
 *
 * Consumes interm one subblock-height band at a time and publishes out one tile row at a time.
 * OutBlockW must be divisible by OutSubblockW;
 * in0_num_subblocks counts row-groups along M. All dimensions are in tiles.
 *
 * Requires compute_kernel_hw_startup. Initializes/uninitializes each call;
 * Reconfigure=false requires the caller to configure both formats.
 */
template <uint32_t OutSubblockW, uint32_t OutBlockW, bool Reconfigure = true>
inline void reblock_and_untilize(
    uint32_t in0_num_subblocks, uint32_t out_subblock_h, uint32_t interm_cb_id, uint32_t out_cb_id);

}  // namespace compute_kernel_lib

#include "reblock_untilize_helpers.inl"
