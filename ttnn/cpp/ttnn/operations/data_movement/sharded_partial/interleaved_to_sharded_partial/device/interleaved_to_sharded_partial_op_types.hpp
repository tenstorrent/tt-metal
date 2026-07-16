// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

struct InterleavedToShardedPartialParams {
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec(tt::tt_metal::CoreRangeSet(), {0, 0});
    uint32_t num_slices{};
    uint32_t slice_index{};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "grid_size", "shard_spec", "num_slices", "slice_index", "output_mem_config", "output_dtype");
    auto attribute_values() const {
        return std::forward_as_tuple(grid_size, shard_spec, num_slices, slice_index, output_mem_config, output_dtype);
    }
};

}  // namespace ttnn::prim
