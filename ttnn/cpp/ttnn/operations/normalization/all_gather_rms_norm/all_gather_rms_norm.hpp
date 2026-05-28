// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

// Generic fused all-gather RMSNorm.
//
// Fuses, into a single multi-device op: per-device partial stats (E[x^2] over the local shard of the
// reduction dim) -> cross-device all-gather of the stats over `cluster_axis` -> post-normalization
// (rsqrt(mean(E[x^2]) + epsilon) * x), with optional weight (gamma), optional bias (beta) and optional
// fused residual add.
//
// Unlike `ttnn::fused_rms_minimal` (the LLaMA-decode "minimal" variant), this op is generic: it supports
// arbitrary M (long sequences), TILE layout, INTERLEAVED memory. It does NOT fuse RoPE or a head-split;
// those are expected to run as separate ops afterward.
ttnn::Tensor all_gather_rms_norm(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,  // gamma
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,    // beta
    float epsilon = 1e-12,
    const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    std::optional<size_t> num_links = std::nullopt,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_stats_tensor = std::nullopt);

}  // namespace ttnn
