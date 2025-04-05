// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

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
constexpr auto bernoulli =
    ttnn::register_operation_with_auto_launch_op<"ttnn::bernoulli", ttnn::operations::bernoulli::Bernoulli>();
}  // namespace ttnn
