// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcPostMixParams {
    uint32_t n;     // streams; the coefficient columns n*n + n fit one tile
    bool collapse;  // emit the mean over the n new streams [T, C] instead of the streams [T, n*C]
};

struct MhcPostMixTensorArgs {
    const Tensor& x;     // [1,1,T,n*C] fp32 packed streams
    const Tensor& y;     // [1,1,T,C] fp32 sublayer output
    const Tensor& post;  // [T,n] fp32
    const Tensor& comb;  // [T,n*n] fp32
};

}  // namespace ttnn::experimental::prim
