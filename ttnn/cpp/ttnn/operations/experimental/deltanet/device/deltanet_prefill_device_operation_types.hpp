// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetPrefillFullParams {
    uint32_t num_heads;
    uint32_t num_k_heads;
    uint32_t k_head_dim;
    uint32_t v_head_dim;
    uint32_t conv_dim;
    uint32_t conv_kernel_size;
    uint32_t head_expand_ratio;
    uint32_t seq_len;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DeltaNetPrefillFullInputs {
    const Tensor& qkv_proj;         // [1,1,S, conv_dim] raw linear projection output
    const Tensor& z_proj;           // [1,1,S, H*Dv]     gating projection
    const Tensor& b_proj;           // [1,1,S, H]        beta projection (padded to tile)
    const Tensor& a_proj;           // [1,1,S, H]        decay projection (padded to tile)
    const Tensor& conv_state;       // [1,1, conv_dim, 32] sliding window
    const Tensor& recurrent_state;  // [1, H, Dk, Dv]    main recurrent state
    const Tensor& conv1d_weight;    // [1,1, conv_dim, 32] convolution weights
    const Tensor& a_log;            // [1,1,1, H]        log-space decay base
    const Tensor& dt_bias;          // [1,1,1, H]        timestep bias
    const Tensor& norm_weight;      // [1,1,1, Dv]       per-head RMSNorm weight
};

}  // namespace ttnn::operations::experimental::deltanet
