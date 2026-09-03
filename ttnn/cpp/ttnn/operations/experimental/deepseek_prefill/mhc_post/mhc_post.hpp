// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post {

// Fused mHC post-mix (issue #55173): the residual-mixing half of
//
//     X' = H_res . X + H_post^T . F(H_pre . X)
//
// applied per output stream j over the packed [1,1,T,n*C] layout:
//
//     out[:, j*C:(j+1)*C] = post[:,j] * y + sum_i comb[:, i*n+j] * residual[:, i*C:(i+1)*C]
//
// The composite ttnn form of this is n column slices, n*n addcmuls and a concat, each round-
// tripping a full [1,1,T,C] accumulator through DRAM. Here every (token-tile, column-tile) pair
// reads its 1+n input tiles once and writes its n output tiles once.
//
//   y:        [1,1,T,C]     FLOAT32 TILE  -- the sublayer output F(H_pre.X)
//   residual: [1,1,T,n*C]   FLOAT32 TILE  -- the n input streams, stream i at [i*C,(i+1)*C)
//   post:     [1,1,T,n]     FLOAT32 TILE  -- mhc_split_sinkhorn output
//   comb:     [1,1,T,n*n]   FLOAT32 TILE  -- mhc_split_sinkhorn output, entry (i,j) at column i*n+j
//   consts:   [n*n,32,32]   FLOAT32 TILE  -- host-prepared column-broadcast tiles; see the
//             Python wrapper. Tile k has row k all ones, so `coeff_tile @ consts[k]` fills every
//             element with that token's column-k coefficient -- a per-token broadcast with no
//             sub-tile broadcast LLK.
//
// Returns out [1,1,T,n*C] FLOAT32 TILE.
ttnn::Tensor mhc_post(
    const ttnn::Tensor& y,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& consts,
    uint32_t n);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_post::mhc_post;
}  // namespace ttnn
