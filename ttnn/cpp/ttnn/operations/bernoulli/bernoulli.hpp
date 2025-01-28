// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::bernoulli {
struct Bernoulli {
    static Tensor invoke(
        const Tensor& input,
        const uint32_t seed,
        const std::optional<Tensor>& output,
        const std::optional<DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::bernoulli

namespace ttnn {
constexpr auto bernoulli = ttnn::register_operation<"ttnn::bernoulli", ttnn::operations::bernoulli::Bernoulli>();
}  // namespace ttnn
