// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
// #include <tt-metalium/operation.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "device/all_gather_minimal_matmul_async_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::experimental::all_gather_minimal_matmul_async {

struct ExecuteAllGatherMinimalMatmulAsync {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& bias_tensor,
        std::optional<unary::UnaryWithParam> fused_activation,
        const std::optional<const AllGatherMinimalMatmulAsyncConfig>& config,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const ttnn::ccl::Topology topology,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
        uint32_t num_links = 1,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        uint32_t chunks_per_sync = 1,
        uint32_t num_workers_per_link = 1,
        uint32_t num_buffers_per_channel = 1);
};

}  // namespace ttnn::operations::experimental::all_gather_minimal_matmul_async

namespace ttnn::experimental {
constexpr auto all_gather_minimal_matmul_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_minimal_matmul_async",
    ttnn::operations::experimental::all_gather_minimal_matmul_async::ExecuteAllGatherMinimalMatmulAsync>();
}
