// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/lnbw_device_operation.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_ln_bw.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_ln_bw(
    const ttnn::Tensor& gy,
    const ttnn::Tensor& x,
    const ttnn::Tensor& red,
    const ttnn::Tensor& n,
    const ttnn::Tensor& gamma,
    uint32_t W,
    uint32_t eps_bits) {
    return ttnn::prim::fused_ln_bw(gy, x, red, n, gamma, W, eps_bits);
}

}  // namespace ttnn::operations::experimental
