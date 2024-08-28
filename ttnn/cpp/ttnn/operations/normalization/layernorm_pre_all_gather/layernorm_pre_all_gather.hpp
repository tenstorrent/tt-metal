// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteLayerNormPreAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<const LayerNormProgramConfig>& program_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto layernorm_pre_all_gather = ttnn::register_operation_with_auto_launch_op<"ttnn::layernorm_pre_all_gather", ttnn::operations::normalization::ExecuteLayerNormPreAllGather>();

}  // namespace ttnn
