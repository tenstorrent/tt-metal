// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Fused per-edge coefficient adjoint (rotate_bw dE/dcoef):
//   gc[e, k] = sum_w gout[e, is[k]*W + w] * xin[e, js[k]*W + w]   for each structural nonzero k.
// Output is compact [E, ceil(nnz/32)*32]; column k holds the dot for nonzero k.
struct FusedGcParams {
    uint32_t n_out;   // number of gout coordinate blocks
    uint32_t n_in;    // number of xin coordinate blocks
    uint32_t W;       // channels per coordinate (multiple of TILE_WIDTH)
    uint32_t nnz;     // number of structural nonzeros
    std::vector<uint32_t> is_;  // length nnz: gout block per nonzero
    std::vector<uint32_t> js;   // length nnz: xin block per nonzero
};

struct FusedGcInputs {
    Tensor gout;  // [E, n_out*W] TILE bf16
    Tensor xin;   // [E, n_in*W]  TILE bf16
    Tensor sel;   // [32, 32*32]  TILE bf16 (tile c has column c all-ones)
};

}  // namespace ttnn::experimental::prim
