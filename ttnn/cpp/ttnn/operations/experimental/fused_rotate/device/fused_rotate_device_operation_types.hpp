// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Fused per-edge sparse Wigner rotation:
//   out[e, i*W + w] = sum_{(i,j,k)} coef_exp[e, k*32 + w'] * x[e, j*W + w]
// where the nonzero pattern (i,j) is fixed for the topology (block-diagonal in l).
// coef_exp is the packed [E, nnz] coefficients broadcast to [E, nnz*32] (each nonzero
// occupies one tile, its value replicated across the 32 tile columns). The rotation is a
// dense fan-in per output block; this op fuses all `nnz` multiply-accumulates into ONE
// kernel launch (1 DRAM read of x + 1 write of out) instead of `nnz` ttnn.addcmul dispatches.
struct FusedRotateParams {
    uint32_t n_in;    // number of input coordinate blocks
    uint32_t n_out;   // number of output coordinate blocks
    uint32_t W;       // channels per coordinate (must be a multiple of TILE_WIDTH)
    uint32_t nnz;     // number of structural nonzeros
    // Grouped-by-output-block sparsity pattern. deg[i] = nonzeros feeding output block i.
    // For each output block i (in order), the next deg[i] entries of (ks, js) give the
    // coefficient index k (into coef tiles) and the input block j.
    std::vector<uint32_t> deg;  // length n_out
    std::vector<uint32_t> ks;   // length nnz
    std::vector<uint32_t> js;   // length nnz
};

struct FusedRotateInputs {
    Tensor x_flat;    // [E, n_in*W] TILE bf16
    Tensor coef_exp;  // [E, nnz*32] TILE bf16 (each nonzero broadcast across 32 cols)
};

}  // namespace ttnn::experimental::prim
