// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file dfb_helpers_compute.hpp
 * @brief Compute-kernel DataflowBuffer query and validation helpers
 */

namespace compute_kernel_lib {

ALWI constexpr uint32_t get_full_tile_size_impl(DataFormat format);

template <DataFormat format>
ALWI constexpr uint32_t get_full_tile_size();

ALWI uint32_t get_full_tile_size(DataFormat format);

ALWI constexpr bool is_block_float_format(uint32_t format);

#ifndef ARCH_QUASAR
ALWI uint32_t get_dfb_num_pages(uint32_t dfb_id);

template <DataFormat format>
ALWI bool is_valid_dfb_tile_page_size(uint32_t dfb_id);

ALWI bool is_valid_dfb_tile_page_size(uint32_t dfb_id, DataFormat format);
#endif  // !ARCH_QUASAR

template <uint32_t dfb_id>
constexpr uint32_t dfb_l1_format();

template <uint32_t dfb_id>
constexpr bool dfb_has_32x32_tiles();

}  // namespace compute_kernel_lib

#include "dfb_helpers_compute.inl"
