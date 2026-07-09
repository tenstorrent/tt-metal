// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/fused_rotate_device_operation.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_rotate.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_rotate(
    const ttnn::Tensor& x_flat,
    const ttnn::Tensor& coef_exp,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t W,
    const std::vector<uint32_t>& deg,
    const std::vector<uint32_t>& ks,
    const std::vector<uint32_t>& js) {
    return ttnn::prim::fused_rotate(x_flat, coef_exp, n_in, n_out, W, deg, ks, js);
}

}  // namespace ttnn::operations::experimental
