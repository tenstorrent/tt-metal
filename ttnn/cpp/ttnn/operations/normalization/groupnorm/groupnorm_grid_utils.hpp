// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/types.hpp"

namespace ttnn::operations::normalization {

struct GroupNormShardedConfigAndGridSize {
    ttnn::MemoryConfig memory_config;
    ttnn::CoreGrid core_grid;
};

// Port of Python ``determine_expected_group_norm_sharded_config_and_grid_size`` (L1 shard spec + CoreGrid).
GroupNormShardedConfigAndGridSize determine_expected_group_norm_sharded_config_and_grid_size(
    tt::tt_metal::CoreCoord device_compute_grid,
    uint32_t num_channels,
    int num_groups,
    uint32_t input_nhw,
    bool is_height_sharded,
    bool is_row_major);

// Compute the number of virtual columns for DRAM group-norm.
// Finds the largest nvc <= min(grid_x, num_groups) such that:
//   (num_channels / nvc) % TILE_SIZE == 0  &&  num_groups % nvc == 0
// Returns 0 if no valid value exists for the given grid_x.
uint32_t compute_num_virtual_cols(uint32_t grid_x, int num_groups, uint32_t num_channels);

// Find the largest valid CoreGrid within (max_x, max_y) bounds for DRAM group-norm.
// The grid must satisfy:
//   num_virtual_rows = (grid_x / num_virtual_cols) * grid_y  <=  Ht
//   Ht % num_virtual_rows == 0
//   num_virtual_rows % num_batches == 0  (when num_virtual_rows >= num_batches)
// where Ht = ceil(input_nhw / TILE_SIZE).
// The num_batches constraint ensures that multicast groups have uniform size,
// which is required for correct semaphore synchronization in the kernels.
// Among valid grids, fully-utilized grids (grid_x % num_virtual_cols == 0, i.e. no
// wasted columns) are preferred: if any exists, the one with the largest grid_x
// (ties broken by largest grid_y) is returned. Otherwise the search falls back to
// the largest valid grid with partial column utilization (grid_x % num_virtual_cols
// != 0), again ordered by largest grid_x then largest grid_y.
// Returns std::nullopt if no valid grid exists.
std::optional<ttnn::CoreGrid> find_expected_dram_grid(
    uint32_t max_x,
    uint32_t max_y,
    uint32_t num_channels,
    int num_groups,
    uint32_t input_nhw,
    uint32_t num_batches = 1);

}  // namespace ttnn::operations::normalization
