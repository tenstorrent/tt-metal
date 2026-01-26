// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gate_mm.hpp"
#include "device/moe_gate_mm_device_operation.hpp"

namespace ttnn::operations::experimental::moe_gate_mm {

ttnn::Tensor ExecuteMoEGateMM::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w_tensor,
    const ttnn::Tensor& output_tensor,
    const uint32_t layer_id) {
    return ttnn::prim::moe_gate_mm(input_tensor, w_tensor, output_tensor, layer_id);
}

}  // namespace ttnn::operations::experimental::moe_gate_mm
