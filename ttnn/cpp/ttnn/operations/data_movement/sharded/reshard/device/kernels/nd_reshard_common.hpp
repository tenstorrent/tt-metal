// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "accessor/tensor_accessor.h"

// Function to iterate over all page IDs in a shard
template <typename DSpec, typename Func>
void iterate_pages_in_shard(const TensorAccessor<DSpec>& accessor, uint32_t shard_id, Func&& process_page) {
    const auto& dspec = accessor.dspec();

    // Assume static rank
    constexpr uint32_t rank = DSpec::rank_ct;

    // Convert shard_id to shard coordinates using shard_grid
    std::array<uint32_t, rank> shard_coord;
    uint32_t remaining_shard_id = shard_id;
    for (int i = rank - 1; i >= 0; --i) {
        shard_coord[i] = remaining_shard_id % dspec.shard_grid()[i];
        remaining_shard_id /= dspec.shard_grid()[i];
    }

    // Function to convert coordinates to page_id
    auto coords_to_page_id = [&](const std::array<uint32_t, rank>& coords) -> uint32_t {
        uint32_t page_id = 0;
        for (uint32_t i = 0; i < rank; ++i) {
            page_id = page_id * dspec.tensor_shape()[i] + coords[i];
        }
        return page_id;
    };

    // Recursively iterate through all combinations of page coordinates within the shard
    std::array<uint32_t, rank> page_coord_within_shard{};
    std::array<uint32_t, rank> global_page_coord;

    auto iterate_dimension = [&](auto&& self, uint32_t dim) -> void {
        if (dim == rank) {
            // Convert shard-relative coordinates to global coordinates
            for (uint32_t i = 0; i < rank; ++i) {
                global_page_coord[i] = shard_coord[i] * dspec.shard_shape()[i] + page_coord_within_shard[i];

                // Check bounds - some shards at edges might have fewer pages
                if (global_page_coord[i] >= dspec.tensor_shape()[i]) {
                    return;  // Skip this page as it's outside tensor bounds
                }
            }

            // Convert to page_id and process
            uint32_t page_id = coords_to_page_id(global_page_coord);
            process_page(page_id);
            return;
        }

        // Iterate through all positions in current dimension
        for (uint32_t i = 0; i < dspec.shard_shape()[dim]; ++i) {
            page_coord_within_shard[dim] = i;
            self(self, dim + 1);
        }
    };

    // Start the recursive iteration
    iterate_dimension(iterate_dimension, 0);
}
