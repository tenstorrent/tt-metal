// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"

namespace ttnn {
namespace operations {
namespace normalization {

struct LayerNorm {

    static inline ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<const LayerNormProgramConfig>& program_config_arg = std::nullopt) {
        const LayerNormProgramConfig& program_config = program_config_arg.value_or(LayerNormDefaultProgramConfig{});

        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        if (residual_input_tensor.has_value()) {
            return tt::operations::primary::add_layernorm(
                input_tensor, residual_input_tensor.value(), epsilon, weight, bias, memory_config, program_config);
        } else {
            return tt::operations::primary::layernorm(
                input_tensor, epsilon, weight, bias, memory_config, program_config);
        }
    }
};

}  // namespace normalization
}  // namespace operations

constexpr auto layer_norm = ttnn::register_operation<ttnn::operations::normalization::LayerNorm>("ttnn::layer_norm");

}  // namespace ttnn
