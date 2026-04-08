// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

ttnn::Tensor gdn_fused(
    const Tensor& conv_out,
    const Tensor& a_fused,
    const Tensor& b_fused,
    const Tensor& neg_exp_A,
    const Tensor& dt_bias,
    const Tensor& norm_w,
    const Tensor& scale_tt,
    const Tensor& rms_scale_tt,
    const Tensor& rms_eps_tt,
    const Tensor& state,
    const Tensor& output,
    uint32_t num_pairs,
    uint32_t num_cores,
    uint32_t Nv_TP,
    uint32_t Nk_TP,
    uint32_t repeat_factor,
    uint32_t key_dim_tp);

}  // namespace ttnn::experimental
