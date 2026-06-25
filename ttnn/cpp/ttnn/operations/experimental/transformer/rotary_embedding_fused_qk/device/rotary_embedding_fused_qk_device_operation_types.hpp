// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

// Fused q+k RoPE: applies the SAME rotary embedding (GPT-J interleaved / rotate_half
// convention, identical to ttnn::experimental::rotary_embedding) to the q and k tensors
// in a SINGLE program (one dispatch) instead of two separate rotary_embedding launches.
// q/k share cos/sin and head_dim; they are processed on disjoint core sets within one
// program, each reading/writing its own interleaved buffer.
struct RotaryEmbeddingFusedQKParams {
    uint32_t seq_len = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct RotaryEmbeddingFusedQKInputs {
    Tensor q;
    Tensor k;
    Tensor cos;
    Tensor sin;
};

}  // namespace ttnn::experimental::prim
