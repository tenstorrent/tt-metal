// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sparse_ring_sdpa.hpp"

#include <utility>

#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_device_operation.hpp"

namespace ttnn::transformer {

ttnn::Tensor sparse_ring_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& block_mask,
    const ttnn::Tensor& pooled_k,
    const ttnn::Tensor& pooled_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    std::size_t logical_n,
    operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy) {
    auto output_tensors = ttnn::prim::sparse_ring_sdpa(
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        block_mask,
        pooled_k,
        pooled_v,
        persistent_output_buffer_k,  // AllGather output / attention input
        persistent_output_buffer_v,
        logical_n,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        ttnn::ccl::convert_2d_to_1d_topology(topology),
        ccl_core_grid_offset,
        subdevice_id,
        scale,
        compute_kernel_config,
        core_allocation_strategy);
    return output_tensors[prim::SPARSE_RING_SDPA_OUTPUT_IDX];
}

}  // namespace ttnn::transformer
