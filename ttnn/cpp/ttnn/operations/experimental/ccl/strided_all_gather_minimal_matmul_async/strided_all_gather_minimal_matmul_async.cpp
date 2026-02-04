// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteStridedAllGatherMinimalMatmulAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord strided_all_gather_core_grid_offset,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const Tensor>& bias,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<bool> read_local_slice_from_input) {
    return ttnn::prim::strided_all_gather_minimal_matmul_async(
        input_tensor,
        weight_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        strided_all_gather_core_grid_offset,
        num_links,
        memory_config_ag,
        topology,
        cluster_axis,
        bias,
        memory_config_mm,
        fused_activation,
        config,
        compute_kernel_config,
        num_workers_per_link.value_or(
            1),  // Conservatively 1 right now since the all gather core grid is hardcoded from the outside
        num_buffers_per_channel,
        read_local_slice_from_input);
}

}  // namespace ttnn::operations::experimental::ccl
