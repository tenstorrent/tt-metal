// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_b1::gate {

struct GateOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& a,
        const ttnn::Tensor& b,
        const ttnn::Tensor& expert_bias,
        const ttnn::CoreGrid& core_grid,
        const std::optional<const ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::DataType>& dtype = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek_b1::gate

namespace ttnn::experimental::deepseek_b1 {

constexpr auto gate = ttnn::register_operation<
    "ttnn::experimental::deepseek_b1::gate",
    ttnn::operations::experimental::deepseek_b1::gate::GateOperation>();

}  // namespace ttnn::experimental::deepseek_b1
