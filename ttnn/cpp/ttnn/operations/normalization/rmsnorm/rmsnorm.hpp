// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"

namespace ttnn {
namespace operations {
namespace normalization {

struct RMSNorm {

    static inline ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight,
        float epsilon = 1e-12,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        return tt::operations::primary::rmsnorm(input_tensor, epsilon, weight, std::nullopt, memory_config);
    }
};

}  // namespace normalization
}  // namespace operations

constexpr auto rms_norm = ttnn::register_operation<ttnn::operations::normalization::RMSNorm>("ttnn::rms_norm");

}  // namespace ttnn
