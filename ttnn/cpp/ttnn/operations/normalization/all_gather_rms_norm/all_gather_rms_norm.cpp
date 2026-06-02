// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm.hpp"

#include "ttnn/operations/normalization/all_gather_rms_norm/device/all_gather_rms_norm_device_operation.hpp"

namespace ttnn {

ttnn::Tensor all_gather_rms_norm(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    float epsilon,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    ttnn::ccl::Topology topology,
    std::optional<size_t> num_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_stats_tensor,
    uint32_t num_heads) {
    return ttnn::prim::all_gather_rms_norm(
        input_tensor,
        cluster_axis,
        mesh_device,
        global_semaphore,
        weight,
        bias,
        epsilon,
        residual_input_tensor,
        topology,
        num_links,
        subdevice_id,
        memory_config,
        compute_kernel_config,
        dtype,
        persistent_stats_tensor,
        num_heads);
}

}  // namespace ttnn
