// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_op.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::fused::normalization {

struct ExecuteFusedRMSNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::operations::normalization::LayerNormProgramConfig& program_config,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
        const std::optional<size_t> num_preferred_links = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& stats = std::nullopt);
};

}  // namespace operations::fused::normalization

constexpr auto fused_rms_1_1_32_8192 = ttnn::
    register_operation<"ttnn::fused_rms_1_1_32_8192", ttnn::operations::fused::normalization::ExecuteFusedRMSNorm>();

}  // namespace ttnn
