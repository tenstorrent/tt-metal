// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/gate_device_operation.hpp"
#include "ttnn/operations/experimental/fused_rotate/fused_gate.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor fused_gate(
    const ttnn::Tensor& a,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& b,
    uint32_t Wt,
    uint32_t Gt,
    uint32_t Ht,
    uint32_t mode) {
    return ttnn::prim::fused_gate(a, gate, b, Wt, Gt, Ht, mode);
}

}  // namespace ttnn::operations::experimental
