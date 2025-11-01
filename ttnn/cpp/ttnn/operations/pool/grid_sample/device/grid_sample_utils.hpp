// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "grid_sample_op.hpp"

namespace ttnn::operations::grid_sample {

// Common constants used across all grid sample program factories
constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
constexpr uint32_t BUFFERING_FACTOR = 2;
constexpr uint32_t REDUCTION_SIZE = 4;
constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;  // Height of one face (always 16)
constexpr bool ONE_SCALAR_PER_CORE = false;
constexpr uint32_t DUMMY_CB_ID = 32;

namespace utils {

// Function to determine if split reader should be used
bool should_use_split_reader(const Tensor& input_tensor, const Tensor& grid_tensor, bool use_precomputed_grid);

// Function to get grid batching factor
uint32_t get_grid_batching_factor(
    const Tensor& grid_tensor, bool use_precomputed_grid, const std::string& mode = "bilinear");

// Function to get aligned stick size for memory alignment
uint32_t get_aligned_stick_size(const ttnn::Shape& shape, const Tensor& tensor);

}  // namespace utils
}  // namespace ttnn::operations::grid_sample
