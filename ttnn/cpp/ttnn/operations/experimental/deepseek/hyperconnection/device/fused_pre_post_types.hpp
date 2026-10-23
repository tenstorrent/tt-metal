// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

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
    const Tensor& fused_w;         // [1,1,T,(2+H)*H] packed pre/post/comb projection.
    const Tensor& pre_bias;        // [1,1,1,H], shared by every token.
    const Tensor& post_bias;       // [1,1,1,H], shared by every token.
    const Tensor& hidden_streams;  // [B,S,H,D] with B*S == T.
    // [B,S,1,H] TILE collapse weights (token t is tile t); replaces the pre computed from
    // fused_w when present.
    std::optional<Tensor> pre_mix;
};

// Returns {post [1,T,H,1], collapsed [1,T,1,D], comb_w_mat [1,T,H,H]}.
// pre_w / post_w are split out of fused_w and consumed inside the kernel; comb_w is
// split out, rearranged into the HxH grid layout, and returned as the third tensor.
// post is emitted as a column ([H,1] per token) rather than the [1,H] row the compute
// produces, so the caller's [B,S,H,1] view is a metadata-only reshape. Given pre_mix, a fourth
// tensor pre [1,T,1,H] (one zero-padded tile per token) carries the pre computed from fused_w.
using FusedPrePostSpecReturn = std::vector<tt::tt_metal::TensorSpec>;
using FusedPrePostTensorReturn = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
