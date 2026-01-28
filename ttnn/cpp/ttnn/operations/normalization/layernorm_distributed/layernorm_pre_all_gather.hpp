// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/layernorm_distributed_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteLayerNormPreAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        DataType dtype = DataType::BFLOAT16,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto layer_norm_pre_all_gather = ttnn::register_operation<
    "ttnn::layer_norm_pre_all_gather",
    ttnn::operations::normalization::ExecuteLayerNormPreAllGather>();

}  // namespace ttnn
