// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async.hpp"

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor llama_all_gather_matmul_async(
    const ttnn::Tensor& input_tensor0,
    const ttnn::Tensor& input_tensor1,
    const ttnn::Tensor& intermediate_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& ag_memory_config,
    const std::optional<MemoryConfig>& mm_memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype,
    const std::optional<const GlobalCircularBuffer>& global_cb) {
    auto output_tensors = ttnn::prim::llama_all_gather_matmul_async(
        input_tensor0,
        input_tensor1,
        intermediate_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        ag_memory_config,
        mm_memory_config,
        num_preferred_links,
        subdevice_id,
        program_config,
        compute_kernel_config,
        dtype,
        global_cb);

    output_tensors.aggregated.deallocate(true);
    return output_tensors.mm;
}

}  // namespace ttnn::experimental
