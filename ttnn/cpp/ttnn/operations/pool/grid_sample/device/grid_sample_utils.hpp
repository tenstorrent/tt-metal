// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Constants
constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
constexpr uint32_t BUFFERING_FACTOR = 2;
constexpr uint32_t REDUCTION_SIZE = 4;
constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;  // Height of one face (always 16)
constexpr bool ONE_SCALAR_PER_CORE = false;
constexpr uint32_t DUMMY_CB_ID = 32;

// Function declarations
bool should_use_split_reader(
    const Tensor& input_tensor, const Tensor& grid_tensor, bool use_precomputed_grid, const std::string& mode);

// A rank-5 grid (N, H_out, W_out, K, 2) is the batched grid before its last two dims are merged; on device that
// merge is a page-by-page gather, so the nearest kernel takes the rank-5 tensor as it is and gathers it itself.
bool is_rank5_grid(const Tensor& grid_tensor);
// The grid's last dim as the kernels see it: 2K for a rank-5 grid, the real last dim otherwise.
uint32_t grid_last_dim_equiv(const Tensor& grid_tensor);
uint32_t get_grid_batching_factor(
    const Tensor& grid_tensor, bool use_precomputed_grid, const std::string& mode = "bilinear");

uint32_t get_aligned_stick_size(const ttnn::Shape& shape, const Tensor& tensor);

}  // namespace ttnn::prim
