// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_rmsnorm.hpp"

#include "device/sharded_rmsnorm_device_operation.hpp"

namespace ttnn::experimental::deepseek {

Tensor sharded_rmsnorm(
    const Tensor& input_tensor,
    const Tensor& weight,
    float epsilon,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Layout> output_layout) {
    return ttnn::prim::sharded_rmsnorm(input_tensor, weight, epsilon, memory_config, output_layout);
}

}  // namespace ttnn::experimental::deepseek
