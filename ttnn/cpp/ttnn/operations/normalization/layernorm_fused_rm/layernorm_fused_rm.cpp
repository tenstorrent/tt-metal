// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fused_rm.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::layernorm_fused_rm {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteLayernormFusedRm::invoke(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& beta,
    float epsilon,
    const std::optional<MemoryConfig>& memory_config) {
    // Call the primitive device operation
    // Unwrap optional memory_config, defaulting to input tensor's memory config
    return ttnn::prim::layernorm_fused_rm(input, gamma, beta, epsilon, memory_config.value_or(input.memory_config()));
}

}  // namespace ttnn::operations::layernorm_fused_rm
