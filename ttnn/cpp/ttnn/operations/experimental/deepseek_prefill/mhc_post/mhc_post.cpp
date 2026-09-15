// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_post.hpp"
#include "device/mhc_post_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post {

ttnn::Tensor mhc_post(
    const ttnn::Tensor& y,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& consts,
    uint32_t n) {
    return ttnn::prim::mhc_post(y, residual, post, comb, consts, n);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post
