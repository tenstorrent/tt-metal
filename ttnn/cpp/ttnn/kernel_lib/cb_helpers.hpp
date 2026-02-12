// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file cb_helpers.hpp
 * @brief Tile size, circular buffer query and validation helpers
 *
 * Reusable utilities for kernel library helpers (reduce, tilize, untilize, etc.)
 */

namespace compute_kernel_lib {

// =============================================================================
// Tile Size Helpers
// =============================================================================

ALWI constexpr uint32_t get_full_tile_size_impl(DataFormat format);

/**
 * @brief Get tile size in bytes for a data format (compile-time version)
 *
 * Device-side equivalent of tt::tile_size() from tt_backend_api_types.hpp.
 * Uses the same datum_shift / exp_shift arithmetic as MUL_WITH_TILE_SIZE.
 * Assumes 32x32 tiles (1024 elements).
 *
 * @tparam format Data format (compile-time constant)
 * @return Tile size in bytes
 */
template <DataFormat format>
ALWI constexpr uint32_t get_full_tile_size();

/**
 * @brief Get tile size in bytes for a data format (runtime version)
 *
 * Same logic as the template version, for use when the data format
 * is not known at compile time (e.g. indexed from unpack_src_format
 * with a runtime CB id).
 *
 * @param format Data format
 * @return Tile size in bytes
 */
ALWI uint32_t get_full_tile_size(DataFormat format);

// =============================================================================
// CB Query Helpers
// =============================================================================

/**
 * @brief Get the number of pages in a circular buffer
 *
 * Derives page count from fifo_size / fifo_page_size, which works on both
 * read and write CB interfaces. (fifo_num_pages is only populated for write interfaces.)
 */
ALWI uint32_t get_cb_num_pages(uint32_t cb_id);

/**
 * @brief Validate that a CB's page size equals the tile size for its data format (compile-time version)
 *
 * @tparam format Data format (compile-time constant)
 * @param cb_id Circular buffer ID
 */
template <DataFormat format>
ALWI bool is_valid_cb_tile_page_size(uint32_t cb_id);

/**
 * @brief Validate that a CB's page size equals the tile size for its data format (runtime version)
 *
 * On compute kernels (TRISC), fifo_page_size is stored in 16-byte units, so we shift left
 * by CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT to convert back to bytes.
 */
ALWI bool is_valid_cb_tile_page_size(uint32_t cb_id, DataFormat format);

}  // namespace compute_kernel_lib

// Include implementation
#include "cb_helpers.inl"
