// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct GdnFusedParams {
    const uint32_t num_pairs_total;
    const uint32_t num_cores;
    const bool state_in_l1;
    const bool state_is_sharded;
    const uint32_t Nv_TP;
    const uint32_t Nk_TP;
    const uint32_t repeat_factor;
    const uint32_t key_dim_tp;
};

struct GdnFusedInputs {
    Tensor conv_out;
    Tensor a_fused;
    Tensor b_fused;
    Tensor neg_exp_A;
    Tensor dt_bias;
    Tensor norm_w;
    Tensor scale_tt;
    Tensor rms_scale_tt;
    Tensor rms_eps_tt;
    Tensor state;
    Tensor output;
};

}  // namespace ttnn::experimental::prim
