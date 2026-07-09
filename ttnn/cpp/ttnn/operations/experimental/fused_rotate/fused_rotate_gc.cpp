// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/gc_device_operation.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_rotate_gc.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_rotate_gc(
    const ttnn::Tensor& gout,
    const ttnn::Tensor& xin,
    const ttnn::Tensor& sel,
    uint32_t n_out,
    uint32_t n_in,
    uint32_t W,
    const std::vector<uint32_t>& is_,
    const std::vector<uint32_t>& js) {
    return ttnn::prim::fused_rotate_gc(gout, xin, sel, n_out, n_in, W, is_, js);
}

}  // namespace ttnn::operations::experimental
