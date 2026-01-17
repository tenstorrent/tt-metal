// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteStridedAllGatherMinimalMatmulAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& persistent_output_buffer,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        CoreCoord strided_all_gather_core_grid_offset,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_ag = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<const Tensor>& bias = std::nullopt,
        const std::optional<operations::unary::UnaryWithParam>& fused_activation = std::nullopt,
        const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        std::optional<bool> read_local_slice_from_input = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto strided_all_gather_minimal_matmul_async = ttnn::register_operation<
    "ttnn::experimental::strided_all_gather_minimal_matmul_async",
    ttnn::operations::experimental::ccl::ExecuteStridedAllGatherMinimalMatmulAsync>();

}  // namespace experimental
}  // namespace ttnn
