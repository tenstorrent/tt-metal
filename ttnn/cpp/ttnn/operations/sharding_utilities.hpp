// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning shards work between multiple cores.
//

#pragma once

#include "tt_metal/common/math.hpp"
#include "tt_metal/common/core_coord.h"

#include "tt_metal/host_api.hpp"

namespace tt::tt_metal {

struct PoolConfig {
    uint32_t in_w;
    uint32_t in_h;
    uint32_t out_w;
    uint32_t out_h;
    uint32_t stride_w;
    uint32_t stride_h;
    uint32_t pad_w;
    uint32_t pad_h;
    uint32_t window_w;
    uint32_t window_h;
    uint32_t dilation_w;
    uint32_t dilation_h;
};

struct NewShardingConfig {
    int32_t first_partial_right_aligned_row_width;
    int32_t first_partial_image_num_rows;
    int32_t num_full_images;
    int32_t last_partial_image_num_rows;
    int32_t last_partial_left_aligned_row_width;
    int32_t skip_after_partial_right_aligned_row;
    int32_t skip_after_first_partial_image_row;
    int32_t skip_after_full_image;
    int32_t initial_skip;
    int32_t start_stick;
};

struct InOutShardingConfig {
    int32_t start_stick;
    int32_t first_partial_right_aligned_row_width;
    int32_t first_partial_image_num_rows;
    int32_t num_full_images;
    int32_t last_partial_image_num_rows;
    int32_t last_partial_left_aligned_row_width;
    int32_t initial_skip;
    int32_t skip_after_stick;
    int32_t skip_after_partial_right_aligned_row;
    int32_t skip_after_first_partial_image_row;
    int32_t skip_after_full_image;
    int32_t skip_after_each_full_row;
    int32_t skip_after_each_stick;
};

struct ShardingConfig {
    uint32_t first_partial_right_aligned_row_width;
    uint32_t first_partial_image_num_rows;
    uint32_t num_full_images;
    uint32_t last_partial_image_num_rows;
    uint32_t last_partial_left_aligned_row_width;
    uint32_t skip_after_partial_right_aligned_row;
    uint32_t skip_after_first_partial_image_row;
    uint32_t skip_after_full_image;
};

// Calculate the sharding specs for input sticks with padding (no halo)
NewShardingConfig get_shard_specs(int32_t start_stick, int32_t end_stick, const PoolConfig& pc, bool to_print = false);

// Calculate the sharding config for input sticks with padding and halo data included
inline NewShardingConfig get_shard_specs_with_halo(int32_t start_stick,
                                                   int32_t end_stick,
                                                   const PoolConfig& pc,
                                                   bool to_print = false);

// For given pool config output, calculate the out sticks sharding config and the corresponding input shard config
std::tuple<InOutShardingConfig, InOutShardingConfig> get_inout_shard_specs(int32_t start_stick,
                                                                           int32_t end_stick,
                                                                           const PoolConfig& pc,
                                                                           bool to_print = false);

ShardingConfig get_specs_for_sharding_partition(uint32_t start_stick,
                                                uint32_t end_stick,
                                                uint32_t in_h,
                                                uint32_t in_w,
                                                uint32_t window_w,
                                                uint32_t pad_h,
                                                uint32_t pad_w);

}  // namespace tt::tt_metal
