// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::bernoulli_new {
struct BernoulliNew {
    static Tensor invoke(
        const Tensor& input,
        uint32_t seed,
        const std::optional<Tensor>& output,
        const std::optional<DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::bernoulli_new

namespace ttnn {
constexpr auto bernoulli_new =
    ttnn::register_operation<"ttnn::bernoulli_new", ttnn::operations::bernoulli_new::BernoulliNew>();
}  // namespace ttnn
