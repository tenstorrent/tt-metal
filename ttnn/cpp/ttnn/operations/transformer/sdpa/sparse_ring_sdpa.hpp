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
    std::optional<DataType> output_dtype = std::nullopt);

}
