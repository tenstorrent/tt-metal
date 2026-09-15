// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcPostParams {
    uint32_t n;          // expansion rate (streams)
    uint32_t max_cores;  // 0 = full grid; 1 pins single-core (benchmark A/B). Hashed for cache correctness.
};

// y: the sublayer output [1,1,T,C].  residual: the n packed input streams [1,1,T,n*C], stream i
// at columns [i*C,(i+1)*C).  post [1,1,T,n] and comb [1,1,T,n*n] are mhc_split_sinkhorn's
// outputs.  consts: [n*n, 32, 32] host-prepared broadcast-extraction tiles -- see the Python
// wrapper.
struct MhcPostTensorArgs {
    const Tensor& y;
    const Tensor& residual;
    const Tensor& post;
    const Tensor& comb;
    const Tensor& consts;
};

}  // namespace ttnn::experimental::prim
