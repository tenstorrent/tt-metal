// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetDecodeParams {
    uint32_t num_heads;
    uint32_t k_head_dim;
    uint32_t v_head_dim;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeltaNetDecodeInputs {
    const Tensor& query;         // [1, num_heads, 1, k_dim] post-L2norm, scaled
    const Tensor& key;           // [1, num_heads, 1, k_dim] post-L2norm
    const Tensor& value;         // [1, num_heads, 1, v_dim]
    const Tensor& decay;         // [1, num_heads, 1, 1] exp(g), scalar per head
    const Tensor& beta;          // [1, num_heads, 1, 1] sigmoid(b), scalar per head
    const Tensor& state;         // [1, num_heads, k_dim, v_dim] recurrent state
};

}  // namespace ttnn::operations::experimental::deltanet
