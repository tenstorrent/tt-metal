// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

namespace ttnn::experimental::prim {

// Fused concat-heads + output-projection matmul in ONE op (one dispatch).
// attn [1, num_heads, seq, head_dim] is consumed DIRECTLY as the matmul in0 with
// K = num_heads*head_dim: for seq <= 1 tile the concat-heads result is the attn buffer's
// contiguous tiles 0..K/32-1 (byte-identical to nlp_concat_heads, PCC 1.0), so the matmul
// in0 reader streams them in concat order with NO separate concat op and NO aliased tensor
// (attn is a normal consumed input -> trace-replay-safe). out = concat(attn) @ weight.
struct ConcatHeadsMatmulParams {
    uint32_t seq_len = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT16;
    DeviceComputeKernelConfig compute_kernel_config;
    // If set, the exact (tuned) matmul program config to use; else auto-derived.
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
};

struct ConcatHeadsMatmulInputs {
    Tensor attn;    // [1, num_heads, seq, head_dim]
    Tensor weight;  // [num_heads*head_dim, N]
};

}  // namespace ttnn::experimental::prim
