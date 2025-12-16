// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file memory.h
 * @brief Memory management functions for compute kernels.
 */

#pragma once

#include "tt_metal/include/compute_kernel_api/cb_api.h"

namespace norm::kernel_util::compute::memory {

/**
 * @brief Get a pointer to the underlying CB tile data, reinterpreted
 * as a pointer to given type.
 *
 * @details Uses get_tile_address() with mailbox-based synchronization to ensure
 * all threads receive the same address. This avoids race conditions due to
 * Tensix semaphore limitations. See GitHub issue #27979 for details.
 *
 * @tparam To The type to reinterpret the CB pointer as a pointer to
 * @param cb_id The CB containing the tile data
 * @param tile_index The index of the tile to get a pointer to
 *
 */
template <typename To>
ALWI auto get_pointer_to_cb_data(uint32_t cb_id, uint32_t tile_index) -> To* {
    return reinterpret_cast<To*>(get_tile_address(cb_id, tile_index));
}
}  // namespace norm::kernel_util::compute::memory
