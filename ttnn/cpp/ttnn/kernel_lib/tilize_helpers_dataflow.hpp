// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace dataflow_kernel_lib {

/**
 * @brief Controls CB page granularity for tilize dataflow
 *
 * Determines how the reader pushes data into the input CB for the
 * compute_kernel_lib::tilize helper (ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp).
 *
 * TILE: CB page_size = tile_size. Reader pushes width_in_tiles pages per
 *       tile-height block. Matches compute_kernel_lib::tilize symmetric mode:
 *         compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks);
 *
 * ROW:  CB page_size = padded_row_bytes (one stick = one page). Reader pushes
 *       1 page per row. Matches compute_kernel_lib::tilize asymmetric mode:
 *         compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks, total_num_rows);
 *       Finer granularity — compute can start as soon as 32 rows arrive.
 *       When total_num_rows < 32, this also reduces L1 usage for the CB
 *       since only total_num_rows row-pages need to be buffered instead
 *       of width_in_tiles tile-pages (which always assume 32 rows of data).
 *
 * Example — TILE granularity (reader kernel):
 *   dataflow_kernel_lib::read_sticks_for_tilize<cb_in>(accessor, num_rows, row_bytes);
 *
 * Example — ROW granularity (reader kernel):
 *   dataflow_kernel_lib::read_sticks_for_tilize<cb_in, dataflow_kernel_lib::TilizeGranularity::ROW>(
 *       accessor, num_rows, row_bytes);
 *
 * Corresponding compute kernel for TILE:
 *   compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks);
 *
 * Corresponding compute kernel for ROW:
 *   compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks, total_num_rows);
 */
enum class TilizeGranularity : uint8_t {
    TILE,
    ROW,
};

/**
 * @brief Read row-major sticks from DRAM into a CB for tilization
 *
 * Reads total_num_rows sticks, grouping them into tile-height blocks.
 * Handles non-tile-aligned widths by padding the L1 stride.
 * Handles non-tile-aligned heights by pushing full tile pages for the
 * last partial block (untouched rows contain stale data).
 *
 * With TILE granularity (default):
 *   - CB must be configured with page_size = tile_size
 *   - Pushes width_in_tiles pages per block
 *   - Compute side: compute_kernel_lib::tilize<W, cb_in, cb_out>(num_blocks)
 *   - CB sizing: double_buffer * width_in_tiles * tile_size
 *
 * With ROW granularity:
 *   - CB must be configured with page_size = padded_row_bytes
 *   - Pushes 1 page per row
 *   - Compute side: compute_kernel_lib::tilize<W, cb_in, cb_out>(num_blocks, total_num_rows)
 *   - CB sizing: double_buffer * min(tile_h, total_num_rows) * padded_row_bytes
 *     (can be smaller than TILE mode when total_num_rows < tile_h)
 *
 * @tparam cb_id Circular buffer to write into (must be constexpr)
 * @tparam granularity TILE (default) or ROW
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the source tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to read
 * @param row_bytes Actual bytes per stick (may be non-tile-aligned)
 * @param start_page Starting page/stick index in the accessor (default 0).
 *        For multi-core work distribution, pass the per-core start_row offset
 *        so the accessor reads from the correct tensor slice.
 */
template <uint32_t cb_id, TilizeGranularity granularity = TilizeGranularity::TILE, typename Accessor>
FORCE_INLINE void read_sticks_for_tilize(
    const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes, uint32_t start_page = 0);

/**
 * @brief Write untilized sticks from a CB to DRAM
 *
 * Reads total_num_rows worth of untilized data from the CB (produced by
 * the compute_kernel_lib::untilize helper from untilize_helpers.hpp) and
 * writes the valid sticks to DRAM.
 *
 * Handles non-tile-aligned widths by skipping L1 padding between rows.
 * Handles non-tile-aligned heights by popping full tile pages for the
 * last partial block but only writing the valid rows.
 *
 * Always operates at TILE granularity — the compute_kernel_lib::untilize
 * helper always produces tile-sized pages on its output CB.
 *
 * Corresponding compute kernel:
 *   compute_kernel_lib::untilize<width_tiles, cb_in, cb_out>(num_blocks);
 *
 * @tparam cb_id Circular buffer to read from (must be constexpr)
 * @tparam Accessor TensorAccessor type (deduced)
 * @param accessor TensorAccessor for the destination tensor (stick-indexed)
 * @param total_num_rows Total number of sticks to write
 * @param row_bytes Actual bytes per stick to write (may be non-tile-aligned)
 * @param start_page Starting page/stick index in the accessor (default 0).
 *        For multi-core work distribution, pass the per-core start_row offset
 *        so the accessor writes to the correct tensor slice.
 */
template <uint32_t cb_id, typename Accessor>
FORCE_INLINE void write_sticks_after_untilize(
    const Accessor& accessor, uint32_t total_num_rows, uint32_t row_bytes, uint32_t start_page = 0);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl"
