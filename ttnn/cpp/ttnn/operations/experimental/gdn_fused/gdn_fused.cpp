// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/gdn_fused_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/gdn_fused/gdn_fused.hpp"

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
    uint32_t key_dim_tp) {
    return ttnn::prim::gdn_fused(
        conv_out,
        a_fused,
        b_fused,
        neg_exp_A,
        dt_bias,
        norm_w,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        state,
        output,
        num_pairs,
        num_cores,
        Nv_TP,
        Nk_TP,
        repeat_factor,
        key_dim_tp);
}

}  // namespace ttnn::experimental
