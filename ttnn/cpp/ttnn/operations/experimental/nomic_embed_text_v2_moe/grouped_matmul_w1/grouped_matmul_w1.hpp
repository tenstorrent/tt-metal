// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Multiplies each 32-row block of an expert-sorted matrix by its own expert's weights, the expert of
// each block derived on device from routing_metadata's expert_counts. See the binding for the
// contract. Reads nothing back to the host.
Tensor grouped_matmul_w1(
    const Tensor& input_tensor,
    const Tensor& weight,
    const std::optional<Tensor>& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight = false,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation = std::nullopt,
    tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool per_expert_lanes = false,
    const std::optional<Tensor>& bias = std::nullopt,
    uint32_t weight_chunk_tiles = 0,
    uint32_t weight_n = 0,
    bool weight_has_bias = false);

// The weight (and bias) of grouped_matmul_w1 as a chunked weight for this device's grid: each core column's slice of
// each expert in the order the op reads it, in contiguous chunks of chunk_tiles tiles spread over the DRAM banks.
// The tiles are copied whole, so the op's results are the same bit for bit as with the TILE weight. Built once, at
// load; pass the result with weight_chunk_tiles = chunk_tiles, weight_n = an expert's N and weight_has_bias.
Tensor prepare_grouped_matmul_w1_weight(
    const Tensor& weight,
    uint32_t num_experts,
    bool transpose_weight,
    const std::optional<Tensor>& bias,
    uint32_t chunk_tiles,
    ttnn::MeshDevice* device);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
