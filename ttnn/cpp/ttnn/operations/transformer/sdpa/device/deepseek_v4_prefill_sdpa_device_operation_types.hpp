// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::prim {

// DeepSeek V4 prefill attention: sliding window plus selected compressed-cache rows.
struct DeepseekV4PrefillSDPAParams {
    float scale = 1.0f;
    uint32_t v_dim;
    uint32_t window_size = 128;
    uint32_t k_chunk_size = 128;
    uint32_t compress_ratio = 0;
    bool empty_tail = false;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct DeepseekV4PrefillSDPAInputs {
    Tensor q;                              // [1, H, S, K_DIM] bf16 ROW_MAJOR
    Tensor window_kv;                      // [1, 1, window_size + S, K_DIM] bf16 ROW_MAJOR: [tail | chunk]
    Tensor start;                          // 1-element uint32/int32 ROW_MAJOR: the chunk's absolute start position
    std::optional<Tensor> cache_kv;        // [1, 1, T, K_DIM] bf16 ROW_MAJOR compressed entries
    std::optional<Tensor> cache_idx;       // [1, 1, S, TOPK] uint32 ROW_MAJOR rows of cache_kv (0xFFFFFFFF = skip)
    std::optional<Tensor> attention_sink;  // [1, 1, 1, H] bf16 ROW_MAJOR
};

}  // namespace ttnn::prim
