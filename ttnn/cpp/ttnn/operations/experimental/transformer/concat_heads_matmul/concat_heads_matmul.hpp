// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

namespace ttnn::experimental {

// Fused concat-heads + output-projection matmul (one dispatch). attn [1, num_heads, seq, head_dim]
// is consumed directly as the matmul in0 (K = num_heads*head_dim); for seq <= 1 tile this is
// byte-identical to nlp_concat_heads(attn) @ weight (PCC 1.0). weight is [num_heads*head_dim, N].
// Returns [1, 1, seq, N].
ttnn::Tensor concat_heads_matmul(
    const Tensor& attn,
    const Tensor& weight,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt);

}  // namespace ttnn::experimental
