// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

Tensor deltanet_recurrence(
    const Tensor& conv_out,
    const Tensor& b_proj,
    const Tensor& a_proj,
    const Tensor& z_proj,
    const Tensor& dt_bias,
    const Tensor& A_exp,
    const Tensor& norm_weight,
    const Tensor& state,
    uint32_t num_heads,
    uint32_t head_k_dim,
    uint32_t head_v_dim,
    uint32_t num_k_heads,
    uint32_t gqa_ratio,
    float scale,
    float norm_eps);

}  // namespace ttnn::experimental
