// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

// Block-sparse ring attention with the K/V ring all-gather fused (as in ring_joint_scaled_dot_product_attention),
// non-causal, no joint tokens, batch 1. Per (head, local Q block), block_mask names the global K blocks computed
// exactly. With pooled_k / pooled_v (Sol-Attn) every other block enters the softmax as one pooled entry weighted by
// its token count; without them the softmax covers the kept blocks only. See the Python docstring for details.
// block_size: only 64; q_chunk_size == block_size; k_chunk_size a multiple of it; dim must be 2 (sequence).
// nb_pad = round_up(ceil(N/ring_size / block_size), 32).
//   q, k, v       [1, H, N/ring_size, D]  TILE, bf16 / bf8 / bf4, one dtype
//   block_mask    [1, H, nqb_pad, ring_size * nb_pad]  uint32 ROW_MAJOR
//   pooled_k / v  optional [1, H, ring_size * nb_pad, D]  TILE, q's dtype
//   block_counts  optional [1, 1, 1, ring_size * nb_pad]  uint32 ROW_MAJOR, real tokens per block
// Returns [1, H, N/ring_size, D], same layout as q.
ttnn::Tensor sparse_ring_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& block_mask,
    const std::optional<ttnn::Tensor>& pooled_k,
    const std::optional<ttnn::Tensor>& pooled_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    std::size_t logical_n,
    operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
    uint32_t block_size = 64,
    std::optional<uint32_t> q_blocks_per_pass = std::nullopt,
    const std::optional<ttnn::Tensor>& block_counts = std::nullopt);

}  // namespace ttnn::transformer
