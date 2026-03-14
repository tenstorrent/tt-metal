// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file cb_helpers_compute.hpp
 * @brief Compute-kernel circular buffer query and validation helpers
 */

namespace compute_kernel_lib {

ALWI constexpr uint32_t get_full_tile_size_impl(DataFormat format);

template <DataFormat format>
ALWI constexpr uint32_t get_full_tile_size();

ALWI uint32_t get_full_tile_size(DataFormat format);

ALWI uint32_t get_cb_num_pages(uint32_t cb_id);

ALWI constexpr bool is_block_float_format(uint32_t format);

template <DataFormat format>
ALWI bool is_valid_cb_tile_page_size(uint32_t cb_id);

ALWI bool is_valid_cb_tile_page_size(uint32_t cb_id, DataFormat format);

}  // namespace compute_kernel_lib

#include "cb_helpers_compute.inl"
