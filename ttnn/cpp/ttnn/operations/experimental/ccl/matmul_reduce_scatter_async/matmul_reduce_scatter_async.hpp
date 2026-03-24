// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> matmul_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    tt::tt_metal::CoreCoord reduce_scatter_core_grid_offset,
    const std::optional<GlobalSemaphore>& barrier_semaphore = std::nullopt,
    const std::optional<const Tensor>& bias = std::nullopt,
    uint32_t num_links = 1,
    const std::optional<ttnn::MemoryConfig>& memory_config_rs = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config_rs = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config_mm = std::nullopt,
    bool transpose_a = false,
    bool transpose_b = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const ttnn::CoreGrid> core_grid = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::experimental
