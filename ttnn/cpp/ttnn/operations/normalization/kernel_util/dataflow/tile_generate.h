// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tile_generate.h
 * @brief Utility functions for generating special-purpose tiles.
          Invoked from dataflow kernels.
 */

#pragma once

#include "tt_metal/hw/inc/dataflow_api.h"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/policies.h"

namespace norm::kernel_util::dataflow {

using norm::kernel_util::generic::policies::CBWaitPolicy;

/**
 * @brief Generate a tile with increasing valuesfrom 1 to TILE_HW inclusive
 * @tparam T The data type of the tile to generate
 * @param cb_id The CB to generate the tile in
 * @param wait_policy The policy to use for waiting on the CB
 */
template <typename T = uint32_t, CBWaitPolicy WaitPolicy = CBWaitPolicy::Wait>
FORCE_INLINE void generate_incremented_tile(const uint32_t cb_id) {
    using L1Ptr = volatile tt_l1_ptr T*;
    cb_reserve_back(cb_id, 1);
    auto ptr = reinterpret_cast<L1Ptr>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
        ptr[i] = static_cast<T>(i + 1);
    }
    cb_push_back(cb_id, 1);
    if constexpr (WaitPolicy == CBWaitPolicy::Wait) {
        cb_wait_front(cb_id, 1);
    }
}

}  // namespace norm::kernel_util::dataflow
