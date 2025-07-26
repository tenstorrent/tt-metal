// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteMatmulReduceScatterAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        ttnn::Tensor& persistent_intermediate_buffer,
        ttnn::Tensor& persistent_output_buffer,
        uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        CoreCoord reduce_scatter_core_grid_offset,
        const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
        const std::optional<const Tensor>& bias = std::nullopt,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_rs = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& intermediate_memory_config_rs = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        bool transpose_a = false,
        bool transpose_b = false,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto matmul_reduce_scatter_async = ttnn::register_operation<
    "ttnn::experimental::matmul_reduce_scatter_async",
    ttnn::operations::experimental::ccl::ExecuteMatmulReduceScatterAsync>();

}  // namespace experimental
}  // namespace ttnn
