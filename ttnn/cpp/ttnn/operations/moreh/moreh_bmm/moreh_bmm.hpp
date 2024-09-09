// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_bmm {
struct MorehBmm {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& mat2,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& output_memory_config,
        const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_bmm

namespace ttnn {
constexpr auto moreh_bmm = ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_bmm", ttnn::operations::moreh::moreh_bmm::MorehBmm>();
}
