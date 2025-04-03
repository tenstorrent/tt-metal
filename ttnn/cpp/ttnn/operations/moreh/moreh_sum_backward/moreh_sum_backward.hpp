// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <optional>
#include <variant>

#include <tt-metalium/small_vector.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_sum_backward {
struct MorehSumBackward {
    static Tensor invoke(
        const Tensor& output_grad,
        const std::optional<Tensor>& input,
        const std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>>& dim,
        bool keepdim,
        const std::optional<Tensor>& input_grad,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_sum_backward

namespace ttnn {
constexpr auto moreh_sum_backward = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_sum_backward",
    ttnn::operations::moreh::moreh_sum_backward::MorehSumBackward>();
}  // namespace ttnn
