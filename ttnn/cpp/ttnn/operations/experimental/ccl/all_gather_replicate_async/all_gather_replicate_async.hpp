// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_replicate_async/device/all_gather_replicate_async_op.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherReplicateAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& input_tensor_b,
        const ttnn::Tensor& intermediate_tensor,
        const ttnn::Tensor& aggregated_tensor,
        const int32_t dim,
        const uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        const ttnn::ccl::Topology topology,
        const GlobalSemaphore& multi_device_global_semaphore,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_replicate_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_replicate_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherReplicateAsync>();

}  // namespace experimental
}  // namespace ttnn
