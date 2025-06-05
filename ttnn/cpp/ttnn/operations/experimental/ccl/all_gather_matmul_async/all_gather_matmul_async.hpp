// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_op.hpp"
#include "ttnn/distributed/api.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherMatmulAsync {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        ttnn::Tensor& persistent_intermediate_buffer,
        ttnn::Tensor& persistent_output_buffer,
        const uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const CoreCoord all_gather_core_grid_offset,
        const std::optional<const Tensor>& bias = std::nullopt,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_ag = std::nullopt,
        const ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto all_gather_matmul_async = ttnn::register_operation<
    "ttnn::experimental::all_gather_matmul_async",
    ttnn::operations::experimental::ccl::ExecuteAllGatherMatmulAsync>();

}  // namespace experimental
}  // namespace ttnn
