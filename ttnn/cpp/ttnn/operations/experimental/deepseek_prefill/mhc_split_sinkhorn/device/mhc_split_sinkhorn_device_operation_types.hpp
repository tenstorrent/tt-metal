// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcSplitSinkhornParams {
    uint32_t n;  // expansion rate (streams); mix_hc = (2+n)*n
    uint32_t sinkhorn_iters;
    float eps;
};

// mixes: fused-projection output [T, (2+n)*n].  consts: [8, 32, 32] host-prepared tiles
// (SEL_pre, SEL_post, SEL_comb with a folded in; base_pre/post/comb; RB, CB) -- see the
// Python wrapper. The scalars a and biases b are baked into these tiles host-side.
struct MhcSplitSinkhornTensorArgs {
    const Tensor& mixes;
    const Tensor& consts;
};

}  // namespace ttnn::experimental::prim
