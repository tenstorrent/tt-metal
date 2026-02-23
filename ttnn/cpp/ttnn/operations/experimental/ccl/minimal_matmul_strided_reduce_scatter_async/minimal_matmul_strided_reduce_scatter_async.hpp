// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteMinimalMatmulStridedReduceScatterAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        CoreCoord reduce_scatter_core_grid_offset,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& rs_output_mem_config = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& rs_intermediate_mem_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<const Tensor>& bias = std::nullopt,
        const std::optional<operations::unary::UnaryWithParam>& fused_activation = std::nullopt,
        const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        bool using_persistent_buffers = false,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        std::optional<uint32_t> chunk_width_in_mm_blocks = std::nullopt,
        const std::optional<Tensor>& optional_rs_intermediate_tensor = std::nullopt,
        const std::optional<Tensor>& optional_rs_output_tensor = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto minimal_matmul_strided_reduce_scatter_async = ttnn::register_operation<
    "ttnn::experimental::minimal_matmul_strided_reduce_scatter_async",
    ttnn::operations::experimental::ccl::ExecuteMinimalMatmulStridedReduceScatterAsync>();

}  // namespace experimental
}  // namespace ttnn
