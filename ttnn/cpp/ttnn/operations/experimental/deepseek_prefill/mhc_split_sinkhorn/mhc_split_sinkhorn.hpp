// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn {

// Fused mHC parametrization (issue #40703 / #40707): splits the fused-projection output
// into (pre, post, comb) and Sinkhorn-normalizes comb, matching DeepSeek-V4's
// hc_split_sinkhorn_kernel and the reference in models/demos/deepseek_v4/reference.
//
//   mixes:  [T, (2+n)*n]  FLOAT32 TILE  -- RMSNorm(X) @ P (computed upstream)
//   consts: [8, 32, 32]   FLOAT32 TILE  -- host-prepared tiles with the scalars a folded
//           in and biases b baked in: SEL_pre, SEL_post, SEL_comb, base_pre, base_post,
//           base_comb, RB (row-sum bcast), CB (col-sum bcast). See the Python wrapper.
//
// Returns {pre [T,n], post [T,n], comb [T,n*n]} FLOAT32 TILE.
std::array<ttnn::Tensor, 3> mhc_split_sinkhorn(
    const ttnn::Tensor& mixes, const ttnn::Tensor& consts, uint32_t n, uint32_t sinkhorn_iters, float eps);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_split_sinkhorn::mhc_split_sinkhorn;
}  // namespace ttnn
