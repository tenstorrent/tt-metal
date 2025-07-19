// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteAllGatherMatmulAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Tensor& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord all_gather_core_grid_offset,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return ttnn::operations::experimental::ccl::all_gather_matmul_async(
        input_tensor,
        weight_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        all_gather_core_grid_offset,
        bias,
        num_links,
        memory_config_ag,
        topology,
        subdevice_id,
        memory_config_mm,
        transpose_a,
        transpose_b,
        dtype,
        program_config,
        activation,
        compute_kernel_config,
        core_grid,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace operations::experimental::ccl
}  // namespace ttnn
