// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteMatmulReduceScatterAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord reduce_scatter_core_grid_offset,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config_rs,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config_rs,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid) {
    return ttnn::operations::experimental::ccl::matmul_reduce_scatter_async(
        input_tensor,
        weight_tensor,
        persistent_intermediate_buffer,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        reduce_scatter_core_grid_offset,
        barrier_semaphore,
        bias,
        num_links,
        memory_config_rs,
        intermediate_memory_config_rs,
        topology,
        subdevice_id,
        memory_config_mm,
        transpose_a,
        transpose_b,
        dtype,
        program_config,
        activation,
        compute_kernel_config,
        core_grid);
}

}  // namespace operations::experimental::ccl
}  // namespace ttnn
