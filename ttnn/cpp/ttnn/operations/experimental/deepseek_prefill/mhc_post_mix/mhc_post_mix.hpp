// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix {

// Fused mHC residual update (the "post" half of a hyper-connection wrap), exact fp32:
//
//   out[t, j*C + c] = sum_i comb[t, i*n + j] * x[t, i*C + c] + post[t, j] * y[t, c]
//
// per 32x32 tile with SFPU fp32 multiply-adds; the per-token coefficient columns are broadcast
// across the tile columns by the unpacker's 32-bit column-broadcast datacopy (exact). With
// ``collapse`` the op returns the unweighted mean over the n new streams instead (the LM head's
// stream collapse): out[t, c] = (1/n) sum_j (...).
//
//   x:    [1, 1, T, n*C] FLOAT32 TILE, the packed residual streams (stream i at columns [i*C, (i+1)*C))
//   y:    [1, 1, T, C]   FLOAT32 TILE, the sublayer output
//   post: [T, n]         FLOAT32 TILE (the mhc_split_sinkhorn output; column j = post_j)
//   comb: [T, n*n]       FLOAT32 TILE (the mhc_split_sinkhorn output; column i*n+j = comb[i, j])
//
// T is the padded row count (a multiple of 32; the rows are independent, pad rows produce pad rows).
// Returns [1, 1, T, n*C] FLOAT32 TILE (or [1, 1, T, C] with collapse), DRAM interleaved.
ttnn::Tensor mhc_post_mix(
    const ttnn::Tensor& x,
    const ttnn::Tensor& y,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    uint32_t n,
    bool collapse);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_post_mix::mhc_post_mix;
}  // namespace ttnn
