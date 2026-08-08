// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct SinkhornParams {
    uint32_t num_streams;     // H (== hc_mult); valid HxH region of the comb tile.
    uint32_t sinkhorn_iters;  // Sinkhorn-Knopp iteration count.
    float comb_scale;         // Learned scale for the comb projection.
    float eps;                // Stability epsilon.
    MemoryConfig output_mem_config;
};

struct SinkhornInputs {
    const Tensor& comb_w;     // [1,T,H,H] (reshaped comb projection), one tile per token.
    const Tensor& comb_bias;  // [1,1,H,H] (reshaped comb bias), single tile shared by all tokens.
};

using SinkhornTensorReturn = Tensor;

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
