// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "all_gather_minimal_matmul_async.hpp"
#include "device/all_gather_minimal_matmul_async_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::all_gather_minimal_matmul_async {

ttnn::Tensor ExecuteAllGatherMinimalMatmulAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<unary::UnaryWithParam> fused_activation,
    const std::optional<const ::ttnn::experimental::prim::AllGatherMinimalMatmulAsyncConfig>& config,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool force_transpose,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel) {
    return ttnn::prim::all_gather_minimal_matmul_async(
        input_tensor,
        weight_tensor,
        bias_tensor,
        fused_activation,
        config,
        multi_device_global_semaphore,
        topology,
        memory_config,
        dtype,
        compute_kernel_config,
        persistent_output_buffer,
        num_links,
        cluster_axis,
        barrier_semaphore,
        force_transpose,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace ttnn::operations::experimental::all_gather_minimal_matmul_async
