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
    // With collapsed_norm_weight: the epsilon of the RMSNorm the collapse cores apply to collapsed.
    float collapsed_norm_epsilon = 0.0F;
    // With a ROW_MAJOR collapsed: a rectangle of cores the collapse cores also multicast the (normed) collapsed row
    // to, returned last as all_gather_for_matmul's replica ([1, 1, cores, D] ROW_MAJOR HEIGHT_SHARDED [1, D]).
    std::optional<CoreRangeSet> collapsed_replica_grid;
};

// The mixed streams' place in the outputs: the mix step comes with pre_mix.
inline constexpr uint32_t kMixedOutputIndex = 4;

// A block of rows (DSpark verify): rows consecutive tokens of one session in one program. Everything per token is then
// a (rows x 32)-tiled block, rows in {2, 4, 8} (pad the block up to one):
//   hidden_streams / the mixed streams: TILE (rows, 32) [1, 1, H * rows, D] (row = stream * rows + j), WIDTH_SHARDED
//     on the 8 collapse cores, shard [H * rows, D / 8];
//   pre_mix [1, rows, 1, H], mix_post [1, rows, H, 1], mix_comb [1, rows, H, H] and the post / comb / pre outputs:
//     32x32 TILE, one tile per row (dim 1);
//   mix_sublayer_out and collapsed_norm_weight: TILE (rows, 32) [1, 1, rows, D], or ROW_MAJOR [1, 1, 1, rows * D] with
//     the block's bytes (tile_rows order); the norm's gain is one row repeated rows times;
//   collapsed: TILE (rows, 32) [1, 1, rows, D] WIDTH_SHARDED [rows, D / 8] on the collapse cores, or with the fused
//   norm
//     to DRAM, ROW_MAJOR [1, 1, 1, rows * D] interleaved (one page, the block's bytes);
//   the replica: TILE (rows, 32) [1, 1, cores * rows, D] HEIGHT_SHARDED [rows, D] on collapsed_replica_grid.
// A block needs fn_weight and pre_mix. The row count is the streams' tile height; 1 is the single-token program.
inline uint32_t hc_block_rows(const Tensor& hidden_streams) {
    if (hidden_streams.layout() != Layout::TILE) {
        return 1;
    }
    const uint32_t h = hidden_streams.tensor_spec().tile().get_height();
    return h < 32 ? h : 1;
}

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
    // With fn_weight and pre_mix: the previous sublayer's mix step, run first on the collapse cores. hidden_streams
    // are then the streams before it, and the op works on (and returns) the mixed streams
    //   mixed[i] = sum_j mix_comb[j, i] * hidden_streams[j] + mix_post[i] * mix_sublayer_out
    // (mix_streams' math). mix_post [1,1,H,1] and mix_comb [1,1,H,H] are BFLOAT16 TILE, mix_sublayer_out
    // [1,1,1,D] BFLOAT16 ROW_MAJOR (interleaved, or sharded with pages a multiple of 32 wide).
    std::optional<Tensor> mix_post;
    std::optional<Tensor> mix_comb;
    std::optional<Tensor> mix_sublayer_out;
    // [1,1,1,D] BFLOAT16 ROW_MAJOR, DRAM interleaved: the gain of an RMSNorm over D the collapse cores apply to
    // collapsed (collapsed_layout ROW_MAJOR, ROW_MAJOR streams). collapsed is then the normed row, ROW_MAJOR and DRAM
    // interleaved: each collapse core sums its D/8 squares, the eight swap their partials, and each scales and
    // writes its own D/8 columns.
    std::optional<Tensor> collapsed_norm_weight;
};

// Returns {post [1,1,H,1], comb [1,1,H,H], collapsed [1,1,1,D]}, plus pre [1,1,1,H] when pre_mix is
// given, plus the mixed streams [1,1,H,D] (hidden_streams' spec) when mix_post / mix_comb / mix_sublayer_out are,
// plus collapsed's replica with collapsed_replica_grid.
using FusedSingleUserSpecReturn = std::vector<tt::tt_metal::TensorSpec>;
using FusedSingleUserTensorReturn = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
