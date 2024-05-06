// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"

namespace ttnn {
namespace operations {
namespace normalization {


inline ttnn::Tensor layer_norm(
    const ttnn::Tensor& input_tensor,
    float epsilon,
    std::optional<const ttnn::Tensor> & weight,
    std::optional<const ttnn::Tensor>& bias,
    std::optional<const ttnn::Tensor>& residual_input_tensor,
    const MemoryConfig& memory_config,
    std::optional<const LayerNormProgramConfig>& program_config
) {

    const LayerNormProgramConfig&  actual_program_config = program_config.value_or(LayerNormDefaultProgramConfig{});

    if (residual_input_tensor.has_value()) {
        return tt::operations::primary::add_layernorm(input_tensor, residual_input_tensor.value(), epsilon, weight, bias, memory_config, actual_program_config);
    }
    else {
        return tt::operations::primary::layernorm(input_tensor, epsilon, weight, bias, memory_config, actual_program_config);
    }
}

inline ttnn::Tensor rms_norm(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight,
    float epsilon = 1e-6
) {
    const MemoryConfig & dram_memory_config = tt::tt_metal::MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED,.buffer_type=tt::tt_metal::BufferType::DRAM};
    return tt::operations::primary::rmsnorm(input_tensor, epsilon, std::optional<const ttnn::Tensor>(weight), std::nullopt, dram_memory_config);
}

} // namespace normalization
} // namespace operations
} // namespace ttnn
