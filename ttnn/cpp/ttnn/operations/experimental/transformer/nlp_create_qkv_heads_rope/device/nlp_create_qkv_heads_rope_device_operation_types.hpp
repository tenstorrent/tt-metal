// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

// Fused create-qkv-heads + q/k RoPE in a SINGLE program (one dispatch). Splits a fused
// [1, 1, seq, (num_q_heads + 2*num_kv_heads)*head_dim] QKV tensor into q/k/v heads AND applies
// RoPE (GPT-J/rotate_half, identical to ttnn::experimental::rotary_embedding) to q and k, leaving
// v un-rotated. Purpose-built for the pi0.5 denoise suffix: requires Ht == 1 (seq <= one tile row),
// transpose_k_heads == False -> the head split is a contiguous tile slice folded into the RoPE
// reader addressing; v is a plain tile copy on its own core.
struct NlpCreateQkvHeadsRopeParams {
    uint32_t num_q_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t seq_len = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct NlpCreateQkvHeadsRopeInputs {
    Tensor qkv;
    Tensor cos;
    Tensor sin;
};

}  // namespace ttnn::experimental::prim
