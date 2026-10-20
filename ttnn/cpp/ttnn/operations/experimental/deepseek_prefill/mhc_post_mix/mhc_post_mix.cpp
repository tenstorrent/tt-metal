// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post_mix.hpp"
#include "device/mhc_post_mix_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix {

ttnn::Tensor mhc_post_mix(
    const ttnn::Tensor& x,
    const ttnn::Tensor& y,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    uint32_t n,
    bool collapse) {
    return ttnn::prim::mhc_post_mix(x, y, post, comb, n, collapse);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix
