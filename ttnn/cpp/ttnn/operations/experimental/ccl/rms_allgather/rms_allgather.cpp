// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_allgather.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_device_operation.hpp"

#include <ttnn/device.hpp>

namespace ttnn::operations::fused::normalization {

ttnn::Tensor ExecuteFusedRMSNorm::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::prim::LayerNormProgramConfig& program_config,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<size_t> num_preferred_links,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<const DataType> dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& stats,
    bool use_noc1_only) {
    return ttnn::prim::rms_allgather(
        input_tensor,
        program_config,
        cluster_axis,
        mesh_device,
        semaphore,
        persistent_output_tensor,
        num_preferred_links,
        topology,
        subdevice_id,
        dtype,
        compute_kernel_config,
        memory_config,
        residual_input_tensor,
        epsilon,
        weight,
        stats,
        use_noc1_only);
}

}  // namespace ttnn::operations::fused::normalization
