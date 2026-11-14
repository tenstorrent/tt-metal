// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

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
    // TILE, or ROW_MAJOR: collapsed is then the [1, D/8] row on each collapse core.
    Layout collapsed_layout = Layout::TILE;
    // With fn_weight: the cores the fn dot cuts K over (fn_weight's own shard grid when it is in L1), the fan-in
    // of the tree its partials are summed up, and the epsilon of the unweighted RMSNorm fused ahead of it.
    std::optional<CoreRangeSet> dot_core_grid;
    uint32_t reduce_fan_in = 2;
    std::optional<float> rms_norm_epsilon;
};

struct FusedSingleUserInputs {
    // [1,1,1,(2+H)*H], WIDTH_SHARDED on core 0: the fn projection, already computed. Exactly one of fused_w /
    // fn_weight is given.
    std::optional<Tensor> fused_w;
    const Tensor& pre_bias;        // [1,1,1,H].
    const Tensor& post_bias;       // [1,1,1,H].
    const Tensor& comb_bias;       // [1,1,H,H].
    const Tensor& hidden_streams;  // [1,1,H,D], WIDTH_SHARDED on cores 0..7.
    // [1,1,1,H] TILE collapse weights; replaces the pre computed from fused_w when present.
    std::optional<Tensor> pre_mix;
    // [1,1,H*D,N] TILE fn weight, K in the collapse cores' (core, stream, column) order: the op computes
    // fused_w = rmsnorm(streams) @ fn_weight itself. DRAM interleaved, or HEIGHT_SHARDED in L1 on dot_core_grid.
    std::optional<Tensor> fn_weight;
};

// Returns {post [1,1,H,1], comb [1,1,H,H], collapsed [1,1,1,D]}, plus pre [1,1,1,H] when pre_mix is
// given.
using FusedSingleUserSpecReturn = std::vector<tt::tt_metal::TensorSpec>;
using FusedSingleUserTensorReturn = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
