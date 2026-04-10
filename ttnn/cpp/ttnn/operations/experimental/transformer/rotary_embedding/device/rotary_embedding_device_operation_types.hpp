// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingParams {
    uint32_t seq_len = 0;
    std::optional<uint32_t> token_idx;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Pi0.5 optimization: if set, output is written to `output_tensor` with this tile offset.
    // Enables fused rotary_embedding + write-to-cache without an intermediate tensor.
    uint32_t dst_tile_offset = 0;
};

struct RotaryEmbeddingInputs {
    Tensor input;
    Tensor cos;
    Tensor sin;
    // Optional pre-allocated output tensor (e.g., a KV cache). If provided, output is written
    // into this tensor at `dst_tile_offset` (from params) instead of a fresh allocation.
    std::optional<Tensor> output_tensor;
};

}  // namespace ttnn::experimental::prim
