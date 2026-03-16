// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor llama_all_gather_matmul_async(
    const ttnn::Tensor& input_tensor0,
    const ttnn::Tensor& input_tensor1,
    const ttnn::Tensor& intermediate_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    std::optional<size_t> num_preferred_links = std::nullopt,
    const std::optional<MemoryConfig>& ag_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& mm_memory_config = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt);

}  // namespace ttnn::experimental
