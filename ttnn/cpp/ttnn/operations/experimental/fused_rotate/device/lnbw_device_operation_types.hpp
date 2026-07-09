// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Fused LayerNorm backward (grad wrt LN input). gy = g_out*gamma (affine scale folded on host).
//   dx = rstd * (gy - mean_w(gy) - xhat*mean_w(gy*xhat)), xhat=(x-mean_w(x))*rstd,
//   rstd = rsqrt(mean_w((x-mean_w(x))^2) + eps).  W must be a multiple of TILE_WIDTH.
struct LnBwParams {
    uint32_t W;         // channels per row (multiple of 32)
    uint32_t eps_bits;  // fp32 bits of the LN epsilon
};

struct LnBwInputs {
    Tensor gy;     // [E, W] TILE bf16  (g_out, matmul result pre-silu-bw)
    Tensor x;      // [E, W] TILE bf16  (cached forward LN input)
    Tensor red;    // [32, 32] TILE bf16, column 0 = 1/W (reduction selector)
    Tensor n;      // [E, W] TILE bf16  (pre-silu activation = LN output; kernel applies silu'(n))
    Tensor gamma;  // [1, W] TILE bf16  (LN affine scale; folded in via row-broadcast)
};

}  // namespace ttnn::experimental::prim
