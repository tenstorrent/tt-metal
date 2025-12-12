// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul::utilities {

// Define the buffering depth for input CBs (0 and 1) for mcast variants.
// 2 = double buffer, 3 = triple buffer, etc.
// Allows easily changing buffering strategy in one place for relevant factories.
constexpr uint32_t MCAST_INPUT_BUFFERING_DEPTH = 2;

uint32_t get_estimated_size_of_cbs(
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    uint32_t interm_single_tile_size,
    uint32_t bias_single_tile_size);

uint32_t estimate_interm_tile_size(
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype);

uint32_t get_max_l1_space(const tt::tt_metal::Tensor& input_tensor_a);

bool is_input_batched(const ttnn::Shape& shape);

ttnn::Shape compute_matmul_output_shape(const Tensor& input_tensor_a, const Tensor& input_tensor_b);

using Activation = std::variant<std::string, ttnn::operations::unary::UnaryWithParam>;
std::optional<ttnn::operations::unary::UnaryWithParam> get_fused_activation(
    const std::optional<const Activation>& activation);

tt::tt_metal::Tile get_output_tile(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const tt::tt_metal::Tile>& optional_output_tensor_tile);

}  // namespace ttnn::operations::matmul::utilities
