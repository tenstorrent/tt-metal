// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Multiplies each 32-row block of an expert-sorted matrix by its own expert's w2, the expert of each block derived on
// device from routing_metadata's expert_counts, into row-major bfloat16 rows. See the binding for the contract.
Tensor grouped_matmul_w2(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& expert_counts,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t first_slots = 2,
    uint32_t second_slots = 2);

// grouped_matmul_w2's weight from the (1, 1, N, num_experts * K) TILE form read transposed, laid out for this device in
// chunks of chunk_tiles tiles; the tiles are copied whole.
Tensor prepare_grouped_matmul_w2_weight(
    const Tensor& weight, uint32_t num_experts, uint32_t chunk_tiles, ttnn::MeshDevice* device);

// The memory config grouped_matmul_w2 reads its (1, 1, rows, k) input in.
tt::tt_metal::MemoryConfig grouped_matmul_w2_input_memory_config(uint32_t rows, uint32_t k, ttnn::MeshDevice* device);

// Circular-buffer bytes a core of a grouped_matmul_w2 call takes, for L1 planning.
uint32_t grouped_matmul_w2_circular_buffer_bytes(
    uint32_t k,
    uint32_t n,
    uint32_t first_slots,
    uint32_t second_slots,
    uint32_t num_experts,
    ttnn::MeshDevice* device);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
