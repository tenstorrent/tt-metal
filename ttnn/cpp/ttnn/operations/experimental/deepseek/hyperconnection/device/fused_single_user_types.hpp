// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedSingleUserParams {
    uint32_t num_streams;
    uint32_t sinkhorn_iters;
    float pre_scale;
    float post_scale;
    float comb_scale;
    float eps;
    MemoryConfig post_comb_output_mem_config;
    MemoryConfig collapsed_output_mem_config;
};

struct FusedSingleUserInputs {
    const Tensor& fused_w;         // [1,1,1,(2+H)*H], WIDTH_SHARDED on core 0.
    const Tensor& pre_bias;        // [1,1,1,H].
    const Tensor& post_bias;       // [1,1,1,H].
    const Tensor& comb_bias;       // [1,1,H,H].
    const Tensor& hidden_streams;  // [1,1,H,D], WIDTH_SHARDED on cores 0..7.
};

// Returns {post [1,1,H,1], comb [1,1,H,H], collapsed [1,1,1,D]}.
using FusedSingleUserSpecReturn = std::array<tt::tt_metal::TensorSpec, 3>;
using FusedSingleUserTensorReturn = std::array<Tensor, 3>;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
