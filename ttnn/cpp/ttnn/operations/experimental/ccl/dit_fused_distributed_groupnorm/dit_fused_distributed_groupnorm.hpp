// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Fused distributed GroupNorm: same contract as ``ttnn.group_norm``, plus fabric
// all-gather of per-group stats on ``cluster_axis`` (PRE → AG → POST).
// Width-1 on ``cluster_axis`` runs local PRE+POST with no fabric.
ttnn::Tensor dit_fused_distributed_groupnorm(
    const ttnn::Tensor& input_tensor,
    int num_groups,
    float epsilon,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    const std::optional<Tensor>& input_mask = std::nullopt,
    const std::optional<Tensor>& weight = std::nullopt,
    const std::optional<Tensor>& bias = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool use_welford = false,
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    std::optional<size_t> num_preferred_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);

// Persistent DRAM scratch for the stats all-gather. Returns nullopt when
// cluster width is 1 (no AG).
std::optional<ttnn::Tensor> dit_fused_distributed_groupnorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    uint32_t num_groups,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    uint32_t num_links = 1);

}  // namespace ttnn::experimental
