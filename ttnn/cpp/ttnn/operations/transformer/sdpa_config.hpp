// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/core_coord.hpp>
#include <tuple>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::optional<CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "sub_core_grids",
        "q_chunk_size",
        "k_chunk_size",
        "exp_approx_mode",
        "max_cores_per_head_batch");
    auto attribute_values() const {
        return std::forward_as_tuple(
            compute_with_storage_grid_size,
            sub_core_grids,
            q_chunk_size,
            k_chunk_size,
            exp_approx_mode,
            max_cores_per_head_batch);
    }
};

}  // namespace ttnn::operations::transformer
