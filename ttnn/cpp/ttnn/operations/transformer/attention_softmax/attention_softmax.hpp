// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp"

namespace ttnn {
namespace operations::transformer {}  // namespace operations::transformer

namespace transformer {

/**
 * @brief Divides input_tensor by the square root of head_size, adds attention_mask (optionally) and computes softmax.
 *
 * This operation is commonly used in attention mechanisms to scale attention scores before applying softmax.
 */
ttnn::Tensor attention_softmax(
    ttnn::Tensor& input_tensor,
    const std::optional<int>& head_size_arg = std::nullopt,
    const std::optional<const ttnn::Tensor>& attention_mask = std::nullopt,
    const ttnn::SoftmaxProgramConfig& program_config = ttnn::SoftmaxDefaultProgramConfig{},
    std::optional<bool> causal_mask = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

/**
 * @brief In-place version: Divides input_tensor by the square root of head_size, adds attention_mask (optionally) and computes softmax.
 *
 * This operation is commonly used in attention mechanisms. The operation is performed in-place to save memory.
 */
ttnn::Tensor attention_softmax_(
    ttnn::Tensor& input_tensor,
    const std::optional<int>& head_size_arg = std::nullopt,
    const std::optional<const ttnn::Tensor>& attention_mask = std::nullopt,
    const ttnn::SoftmaxProgramConfig& program_config = ttnn::SoftmaxDefaultProgramConfig{},
    std::optional<bool> causal_mask = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace transformer

}  // namespace ttnn
