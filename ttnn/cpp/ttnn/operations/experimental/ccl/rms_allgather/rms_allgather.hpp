// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn {
namespace operations::fused::normalization {

struct ExecuteFusedRMSNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::prim::LayerNormProgramConfig& program_config,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const GlobalSemaphore& semaphore,
        const std::optional<ttnn::Tensor>& persistent_output_tensor = std::nullopt,
        std::optional<size_t> num_preferred_links = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& stats = std::nullopt,
        bool use_noc1_only = false);
};

}  // namespace operations::fused::normalization

constexpr auto fused_rms_minimal =
    ttnn::register_operation<"ttnn::fused_rms_minimal", ttnn::operations::fused::normalization::ExecuteFusedRMSNorm>();

}  // namespace ttnn
