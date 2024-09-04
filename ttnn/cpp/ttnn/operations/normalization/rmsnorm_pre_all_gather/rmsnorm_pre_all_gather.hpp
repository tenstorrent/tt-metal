// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteRMSNormPreAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<const LayerNormProgramConfig>& program_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto rmsnorm_pre_all_gather = ttnn::register_operation_with_auto_launch_op<"ttnn::rmsnorm_pre_all_gather", ttnn::operations::normalization::ExecuteRMSNormPreAllGather>();

}  // namespace ttnn
