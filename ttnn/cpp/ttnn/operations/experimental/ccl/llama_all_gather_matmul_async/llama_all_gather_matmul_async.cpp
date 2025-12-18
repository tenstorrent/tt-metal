// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async.hpp"

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherMatmulAsync::invoke(
    const ttnn::Tensor& input0,
    const ttnn::Tensor& input1,
    const ttnn::Tensor& intermediate_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& ag_memory_config,
    const std::optional<MemoryConfig>& mm_memory_config,
    const std::optional<size_t> num_preferred_links,
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    auto output_tensors = ttnn::prim::llama_all_gather_matmul_async(
        input0,
        input1,
        intermediate_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        ag_memory_config,
        mm_memory_config,
        num_preferred_links,
        sub_device_id,
        program_config,
        compute_kernel_config,
        dtype,
        global_cb);

    output_tensors.aggregated.deallocate(true);
    return output_tensors.mm;
}

}  // namespace ttnn::operations::experimental::ccl
