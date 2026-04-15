// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_grid_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::normalization {

namespace {

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (divisor > 0 && num % divisor != 0) {
        divisor -= 1;
    }
    return divisor;
}

}  // namespace

GroupNormShardedConfigAndGridSize determine_expected_group_norm_sharded_config_and_grid_size(
    tt::tt_metal::CoreCoord device_compute_grid,
    uint32_t num_channels,
    int num_groups,
    uint32_t input_nhw,
    bool is_height_sharded,
    bool is_row_major) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::CoreCoord;
    using tt::tt_metal::CoreRange;
    using tt::tt_metal::CoreRangeSet;
    using tt::tt_metal::MemoryConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpec;
    using tt::tt_metal::TensorMemoryLayout;

    TT_FATAL(num_groups > 0, "num_groups needs to be greater than 0");
    TT_FATAL(num_channels > 0, "num_channels needs to be greater than 0");
    TT_FATAL(
        num_channels % static_cast<uint32_t>(num_groups) == 0,
        "group_norm: num_channels ({}) must be divisible by num_groups ({}).",
        num_channels,
        num_groups);
    TT_FATAL(
        num_channels % ttnn::types::TILE_SIZE == 0,
        "group_norm: num_channels ({}) must be divisible by tile width ({}).",
        num_channels,
        ttnn::types::TILE_SIZE);

    const uint32_t group_size = num_channels / static_cast<uint32_t>(num_groups);
    uint32_t dg0 = static_cast<uint32_t>(device_compute_grid.x);
    uint32_t dg1 = static_cast<uint32_t>(device_compute_grid.y);
    if (is_row_major) {
        std::swap(dg0, dg1);
    }
    const uint32_t max_num_cores = dg0 * dg1;

    const uint32_t input_nhw_padded_to32 = tt::div_up(input_nhw, ttnn::types::TILE_SIZE) * ttnn::types::TILE_SIZE;
    const uint32_t ht_padded = input_nhw_padded_to32 / ttnn::types::TILE_SIZE;
    const uint32_t start_divisor = is_height_sharded ? max_num_cores : dg0;
    uint32_t num_cores_nhw = find_closest_largest_divisor(ht_padded, start_divisor);
    TT_FATAL(
        num_cores_nhw > 0,
        "group_norm: Could not find num_cores_nhw for sharded config (Ht_padded={}, start_divisor={}).",
        ht_padded,
        start_divisor);

    uint32_t num_cores_channels = 1;
    if (!is_height_sharded) {
        num_cores_channels = dg1;
        const uint32_t num_channels_tiles = num_channels / 8;
        while (num_cores_channels > 0 &&
               ((num_channels_tiles % num_cores_channels != 0) ||
                ((num_channels / num_cores_channels) % group_size != 0) || (num_channels / num_cores_channels < 32))) {
            num_cores_channels -= 1;
        }
        TT_FATAL(
            num_cores_channels > 0,
            "group_norm: Could not find num_cores_channels for block-sharded group norm (C={}, G={}).",
            num_channels,
            num_groups);
    }

    const uint32_t input_nhw_padded_to_ncores =
        tt::div_up(input_nhw, num_cores_nhw * ttnn::types::TILE_SIZE) * (num_cores_nhw * ttnn::types::TILE_SIZE);
    const uint32_t gn_in_channels_per_core = num_channels / num_cores_channels;
    TT_FATAL(
        gn_in_channels_per_core % 8 == 0,
        "group_norm: gn_in_channels_per_core ({}) must be divisible by 8.",
        gn_in_channels_per_core);
    const uint32_t gn_nhw_per_core = input_nhw_padded_to_ncores / num_cores_nhw;

    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    if (is_height_sharded) {
        grid_x = (num_cores_nhw >= dg0) ? dg0 : num_cores_nhw;
        grid_y = tt::div_up(num_cores_nhw, dg0);
        TT_FATAL(
            num_cores_nhw <= grid_x * grid_y,
            "group_norm: num_cores_nhw ({}) exceeds grid capacity ({}x{}={}).",
            num_cores_nhw,
            grid_x,
            grid_y,
            grid_x * grid_y);
    } else {
        grid_x = is_row_major ? num_cores_channels : num_cores_nhw;
        grid_y = is_row_major ? num_cores_nhw : num_cores_channels;
    }

    const ShardOrientation shard_orientation =
        (is_height_sharded || is_row_major) ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR;

    std::array<uint32_t, 2> shard_shape{};
    if (shard_orientation == ShardOrientation::ROW_MAJOR) {
        shard_shape = {gn_nhw_per_core, gn_in_channels_per_core};
    } else {
        shard_shape = {gn_in_channels_per_core, gn_nhw_per_core};
    }

    const CoreCoord grid_end{static_cast<int>(grid_x) - 1, static_cast<int>(grid_y) - 1};
    const CoreRangeSet shard_grid(CoreRange(CoreCoord{0, 0}, grid_end));

    const TensorMemoryLayout tensor_memory_layout =
        is_height_sharded ? TensorMemoryLayout::HEIGHT_SHARDED : TensorMemoryLayout::BLOCK_SHARDED;
    const ShardSpec shard_spec(shard_grid, shard_shape, shard_orientation);
    MemoryConfig memory_config(tensor_memory_layout, BufferType::L1, shard_spec);

    return GroupNormShardedConfigAndGridSize{
        .memory_config = std::move(memory_config),
        .core_grid = ttnn::CoreGrid(grid_x, grid_y),
    };
}

uint32_t compute_num_virtual_cols(uint32_t grid_x, int num_groups, uint32_t num_channels) {
    uint32_t nvc = std::min<uint32_t>(grid_x, num_groups);
    while (nvc > 0 && ((num_channels / nvc) % ttnn::types::TILE_SIZE != 0 || (num_groups % nvc) != 0)) {
        nvc -= 1;
    }
    return nvc;
}

std::optional<ttnn::CoreGrid> find_expected_dram_grid(
    uint32_t max_x, uint32_t max_y, uint32_t num_channels, int num_groups, uint32_t input_nhw) {
    uint32_t Ht = static_cast<uint32_t>(std::ceil(static_cast<double>(input_nhw) / ttnn::types::TILE_SIZE));

    for (uint32_t gx = max_x; gx >= 1; --gx) {
        uint32_t nvc = compute_num_virtual_cols(gx, num_groups, num_channels);
        if (nvc == 0) {
            continue;
        }
        uint32_t rows_per_y = gx / nvc;
        if (rows_per_y == 0) {
            continue;
        }
        uint32_t max_gy = std::min<uint32_t>(Ht / rows_per_y, max_y);
        for (uint32_t gy = max_gy; gy >= 1; --gy) {
            uint32_t num_virtual_rows = rows_per_y * gy;
            if (Ht % num_virtual_rows == 0) {
                return ttnn::CoreGrid(gx, gy);
            }
        }
    }
    return std::nullopt;
}

}  // namespace ttnn::operations::normalization
