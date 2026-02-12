// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/minimal_matmul_strided_reduce_scatter_async.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteMinimalMatmulStridedReduceScatterAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord reduce_scatter_core_grid_offset,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    const std::optional<ttnn::MemoryConfig>& rs_output_mem_config,
    const std::optional<ttnn::MemoryConfig>& rs_intermediate_mem_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const Tensor>& bias,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    const std::optional<Tensor>& optional_rs_intermediate_tensor,
    const std::optional<Tensor>& optional_rs_output_tensor) {
    return ttnn::prim::minimal_matmul_strided_reduce_scatter_async(
        input_tensor,
        weight_tensor,
        dim,
        multi_device_global_semaphore,
        reduce_scatter_core_grid_offset,
        num_links,
        memory_config_mm,
        rs_output_mem_config.value_or(input_tensor.memory_config()),
        rs_intermediate_mem_config,
        topology,
        cluster_axis,
        bias,
        fused_activation,
        config,
        compute_kernel_config,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        chunks_per_sync,
        num_workers_per_link.value_or(1),
        num_buffers_per_channel,
        chunk_width_in_mm_blocks,
        optional_rs_intermediate_tensor,
        optional_rs_output_tensor);
}

}  // namespace ttnn::operations::experimental::ccl
