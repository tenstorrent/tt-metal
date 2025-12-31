// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/global_semaphore.hpp"

using tt::tt_metal::MemoryConfig;

namespace ttnn {

Tensor fused_rms_1_1_32_8192(
    const Tensor& input_tensor,
    const operations::normalization::LayerNormProgramConfig& program_config,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& semaphore,
    const std::optional<Tensor>& persistent_output_tensor = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    ccl::Topology topology = ccl::Topology::Linear,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const Tensor>& residual_input_tensor = std::nullopt,
    float epsilon = 1e-12,
    const std::optional<const Tensor>& weight = std::nullopt,
    const std::optional<const Tensor>& stats = std::nullopt,
    bool use_noc1_only = false);

}  // namespace ttnn
