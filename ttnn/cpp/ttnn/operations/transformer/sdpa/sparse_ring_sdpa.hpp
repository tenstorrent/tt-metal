// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    const std::optional<ttnn::Tensor>& block_counts = std::nullopt,
    const std::optional<ttnn::Tensor>& head_weights = std::nullopt,
    const std::vector<uint32_t>& dense_q_blocks = {},
    // Token-major output (MINIMAX_H3_MK_S1_TOKEN_MAJOR_O): write O as [1, 1, N/ring_size, H * D] (the
    // concatenate_heads layout), optionally in another dtype (bfloat8_b for a BFP8 consumer). Defaults = today's
    // [1, H, N/ring_size, D] bf16.
    bool output_token_major = false,
    std::optional<DataType> output_dtype = std::nullopt,
    // Runtime logical length ([1, 1, 1, 1] uint32 DRAM, rewritten in place between calls / trace replays): one
    // program for every length of the padded shape; logical_n is then a placeholder. Defaults to None.
    const std::optional<ttnn::Tensor>& logical_n_tensor = std::nullopt,
    // Dense Q block table ([1, 1, 1, K] uint32 DRAM, K <= 64 global Q-block indices or 0xFFFFFFFF): the
    // dense_q_blocks list read on device, so the host may rewrite it in place; the program is sized for
    // dense_q_blocks_max_per_shard. Defaults to None.
    const std::optional<ttnn::Tensor>& dense_q_blocks_tensor = std::nullopt,
    // With dense_q_blocks_tensor: the most dense Q blocks on one shard the program is sized for (part of the program
    // hash; the list path's value is its own most per shard). Defaults to 8, the worst case.
    uint32_t dense_q_blocks_max_per_shard = 8,
    // Block residual ([1, NH, nqb_pad, 32] float32 row-major DRAM): pooled_k / pooled_v hold C entries per Q block
    // (entry qb * C + c), each Q block attending its own C with these column biases (words 0..C-1 of its row).
    // Defaults to None: one pooled entry per key block.
    const std::optional<ttnn::Tensor>& residual_bias = std::nullopt);
}
