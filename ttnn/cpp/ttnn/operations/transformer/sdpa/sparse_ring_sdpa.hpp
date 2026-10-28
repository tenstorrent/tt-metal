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

// Block-sparse ring attention (Sol-Attn), non-causal, no joint tokens. The ring all-gather of K/V is fused, as in
// ring_joint_scaled_dot_product_attention. Blocks are square, `block_size` tokens (B): per (head, local B-row Q
// block), `block_mask` names the global B-token K blocks computed exactly; every other block enters the same softmax
// through its pooled entry (score q * mean(k), value mean(v), weight = the block's token count). block_size is a
// compile-time parameter of the kernels; only 64 is supported for now. q_chunk_size must equal block_size and
// k_chunk_size (the K/V piece) must be a multiple of it. nb_pad = round_up(ceil(N/ring_size / B), 32).
//   q, k, v    [1, H, N/ring_size, D]  bf16 TILE (sequence-sharded along the ring axis)
//   block_mask [1, H, nqb_pad, ring_size * nb_pad]  uint32 ROW_MAJOR; != 0 keeps global block s * nb_pad + b
//   pooled_k   [1, H, ring_size * nb_pad, D]  bf16 TILE; per-block mean K over the valid tokens, zero padding
//   pooled_v   [1, H, ring_size * nb_pad, D]  bf16 TILE; per-block mean V
// Returns the attention output, [1, H, N/ring_size, D], same layout as q.
ttnn::Tensor sparse_ring_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& block_mask,
    const ttnn::Tensor& pooled_k,
    const ttnn::Tensor& pooled_v,
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
    uint32_t block_size = 64);

}  // namespace ttnn::transformer
