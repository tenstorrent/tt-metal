// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::normalization {

struct ExecuteRMSNormPostAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& stats,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<bool>& use_2d_core_grid = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto rms_norm_post_all_gather = ttnn::register_operation<
    "ttnn::rms_norm_post_all_gather",
    ttnn::operations::normalization::ExecuteRMSNormPostAllGather>();

}  // namespace ttnn
