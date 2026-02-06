// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
namespace ttnn::operations::experimental::ccl {

struct ExecuteMinimalMatmulReduceScatterAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        ttnn::Tensor& persistent_intermediate_buffer,
        std::optional<ttnn::Tensor>& persistent_output_buffer,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        CoreCoord reduce_scatter_core_grid_offset,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        const std::optional<const Tensor>& bias = std::nullopt,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_rs = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& intermediate_memory_config_rs = std::nullopt,
        tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const ::ttnn::experimental::prim::MinimalMatmulConfig>& program_config = std::nullopt,
        const std::optional<const unary::UnaryWithParam>& activation = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::experimental {

constexpr auto minimal_matmul_reduce_scatter_async = ttnn::register_operation<
    "ttnn::experimental::minimal_matmul_reduce_scatter_async",
    ttnn::operations::experimental::ccl::ExecuteMinimalMatmulReduceScatterAsync>();

}  // namespace ttnn::experimental
