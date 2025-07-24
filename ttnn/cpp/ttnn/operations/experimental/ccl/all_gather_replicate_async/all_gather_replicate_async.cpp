// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_replicate_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_replicate_async/device/all_gather_replicate_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherMatmulAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& input_tensor_b,
    const ttnn::Tensor& intermediate_tensor,
    const ttnn::Tensor& aggregated_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType> dtype) {
    return ttnn::operations::experimental::ccl::llama_all_gather_matmul_async(
        input_tensor,
        input_tensor_b,
        intermediate_tensor,
        aggregated_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        memory_config,
        num_preferred_links,
        subdevice_id,
        program_config,
        compute_kernel_config,
        dtype);
}

}  // namespace ttnn::operations::experimental::ccl
