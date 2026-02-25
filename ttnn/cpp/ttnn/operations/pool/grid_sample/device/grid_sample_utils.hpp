// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

uint32_t get_grid_batching_factor(
    const Tensor& grid_tensor, bool use_precomputed_grid, const std::string& mode = "bilinear");

uint32_t get_aligned_stick_size(const ttnn::Shape& shape, const Tensor& tensor);

}  // namespace ttnn::prim
