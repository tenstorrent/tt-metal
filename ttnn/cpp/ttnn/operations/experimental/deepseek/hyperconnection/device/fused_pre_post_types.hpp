// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedPrePostParams {
    uint32_t num_streams;  // H (== hc_mult); drives the in-kernel split of fused_w.
    float pre_scale;
    float post_scale;
    float eps;
    MemoryConfig output_mem_config;
};

struct FusedPrePostInputs {
    const Tensor& fused_w;         // [1,1,T,(2+H)*H] packed pre/post/comb projection (T == 1).
    const Tensor& pre_bias;        // [1,1,1,H].
    const Tensor& post_bias;       // [1,1,1,H].
    const Tensor& hidden_streams;  // [1,1,H,D] (decode, T == 1).
};

// Returns {post [1,1,1,H], collapsed [1,1,1,D], comb_w_mat [1,1,H,H]}.
// pre_w / post_w are split out of fused_w and consumed inside the kernel; comb_w is
// split out, rearranged into the HxH grid layout, and returned as the third tensor.
using FusedPrePostSpecReturn = std::array<tt::tt_metal::TensorSpec, 3>;
using FusedPrePostTensorReturn = std::array<Tensor, 3>;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
