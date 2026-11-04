// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Multiplies each 32-row block of an expert-sorted matrix by its own expert's weights, the expert of
// each block derived on device from routing_metadata's expert_counts. See the binding for the
// contract. Reads nothing back to the host.
Tensor grouped_matmul(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight = false,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation = std::nullopt,
    tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool per_expert_lanes = false);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
