// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe.hpp"
#include "device/moe_device_operation.hpp"

namespace ttnn::operations::experimental::moe {

ttnn::Tensor ExecuteMoE::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w0_tensor,
    const ttnn::Tensor& w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::Tensor& output_tensor,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en) {
    return ttnn::prim::moe(
        input_tensor, w0_tensor, w1_tensor, w2_tensor, output_tensor, math_fidelity, fp32_dest_acc_en);
}

}  // namespace ttnn::operations::experimental::moe
