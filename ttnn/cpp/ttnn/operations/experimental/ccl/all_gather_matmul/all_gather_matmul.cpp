// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul/all_gather_matmul.hpp"

namespace ttnn {
namespace operations::experimental::ccl {


std::vector<ttnn::Tensor> ExecuteAllGatherMatmul::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const CoreCoord all_gather_core_grid_offset,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config_ag,
    const std::optional<size_t> num_workers,
    const std::optional<size_t> num_buffers_per_channel,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig> program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid
) {
    return ttnn::operations::experimental::ccl::all_gather_matmul(input_tensor, weight_tensor, dim, all_gather_core_grid_offset, num_links, memory_config_ag, num_workers, num_buffers_per_channel, memory_config_mm, transpose_a, transpose_b, dtype, program_config, activation, compute_kernel_config, core_grid);
}


}  // namespace opereations::experimental::ccl
} // namespace ttnn
