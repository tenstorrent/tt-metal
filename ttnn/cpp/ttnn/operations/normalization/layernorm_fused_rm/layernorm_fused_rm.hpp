// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/layernorm_fused_rm_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace layernorm_fused_rm {

struct ExecuteLayernormFusedRm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        const ttnn::Tensor& gamma,
        const ttnn::Tensor& beta,
        float epsilon = 1e-5f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace layernorm_fused_rm
}  // namespace operations

// Register the operation
constexpr auto layernorm_fused_rm = ttnn::
    register_operation<"ttnn::layernorm_fused_rm", ttnn::operations::layernorm_fused_rm::ExecuteLayernormFusedRm>();

}  // namespace ttnn
