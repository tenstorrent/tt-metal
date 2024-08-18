// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/layernorm_types.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteLayerNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        float epsilon,
        const std::optional<const ttnn::Tensor>& weight,
        const std::optional<const ttnn::Tensor>& bias,
        const std::optional<const ttnn::Tensor>& residual_input_tensor,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<const LayerNormProgramConfig>& program_config,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace operations::normalization

constexpr auto layer_norm =
    ttnn::register_operation_with_auto_launch_op<"ttnn::layer_norm", ttnn::operations::normalization::ExecuteLayerNorm>();

}  // namespace ttnn
