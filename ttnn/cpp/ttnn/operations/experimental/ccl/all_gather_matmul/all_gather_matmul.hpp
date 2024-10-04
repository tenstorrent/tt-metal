// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "common/core_coord.h"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/cpp/ttnn/distributed/mesh_device.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherMatmul {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const uint32_t dim,
        const CoreCoord all_gather_core_grid_offset,
        const uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config_ag = std::nullopt,
        const std::optional<size_t> num_workers = std::nullopt,
        const std::optional<size_t> num_buffers_per_channel = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig> program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);
};

}  // namespace opereations::experimental::ccl

namespace experimental {

constexpr auto all_gather_matmul = ttnn::register_operation<
    "ttnn::experimental::all_gather_matmul",
    ttnn::operations::experimental::ccl::ExecuteAllGatherMatmul>();

}  // namespace experimental
}  // namespace ttnn
