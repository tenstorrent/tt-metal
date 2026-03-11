// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/types.hpp"

namespace ttnn::operations::normalization {

// Compute the number of virtual columns for DRAM group-norm.
// Finds the largest nvc <= min(grid_x, num_groups) such that:
//   (num_channels / nvc) % TILE_SIZE == 0  &&  num_groups % nvc == 0
// Returns 0 if no valid value exists for the given grid_x.
uint32_t compute_num_virtual_cols(uint32_t grid_x, int num_groups, uint32_t num_channels);

// Find the largest valid CoreGrid within (max_x, max_y) bounds for DRAM group-norm.
// The grid must satisfy:
//   num_virtual_rows = (grid_x / num_virtual_cols) * grid_y  <=  Ht
//   Ht % num_virtual_rows == 0
// where Ht = ceil(input_nhw / TILE_SIZE).
// Returns std::nullopt if no valid grid exists.
std::optional<ttnn::CoreGrid> find_expected_dram_grid(
    uint32_t max_x, uint32_t max_y, uint32_t num_channels, int num_groups, uint32_t input_nhw);

}  // namespace ttnn::operations::normalization
