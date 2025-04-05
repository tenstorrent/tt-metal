// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_bmm_backward {
struct MorehBMMBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mat2,
        const std::vector<bool>& are_required_outputs,
        const std::optional<Tensor>& input_grad,
        const std::optional<Tensor>& mat2_grad,
        const std::optional<MemoryConfig>& input_grad_memory_config,
        const std::optional<MemoryConfig>& mat2_grad_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_bmm_backward

namespace ttnn {
constexpr auto moreh_bmm_backward = ttnn::
    register_operation<"ttnn::moreh_bmm_backward", ttnn::operations::moreh::moreh_bmm_backward::MorehBMMBackward>();
}  // namespace ttnn
