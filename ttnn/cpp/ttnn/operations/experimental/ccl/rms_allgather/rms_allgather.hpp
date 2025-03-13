// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::fused::normalization {

struct ExecuteFusedRMSNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
        const ttnn::operations::normalization::LayerNormProgramConfig& program_config,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        bool is_pre = true);
};

}  // namespace operations::fused::normalization

constexpr auto fused_rms_1_1_32_8192 = ttnn::register_operation_with_auto_launch_op<
    "ttnn::fused_rms_1_1_32_8192",
    ttnn::operations::fused::normalization::ExecuteFusedRMSNorm>();

}  // namespace ttnn
