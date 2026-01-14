// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherMatmulAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input0,
        const ttnn::Tensor& input1,
        const ttnn::Tensor& intermediate_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        const GlobalSemaphore& multi_device_global_semaphore,
        const std::optional<MemoryConfig>& ag_memory_config = std::nullopt,
        const std::optional<MemoryConfig>& mm_memory_config = std::nullopt,
        std::optional<size_t> num_preferred_links = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto llama_all_gather_matmul_async = ttnn::register_operation<
    "ttnn::experimental::llama_all_gather_matmul_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherMatmulAsync>();

}  // namespace experimental

}  // namespace ttnn
