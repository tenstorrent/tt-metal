// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe.hpp"
#include "device/moe_device_operation.hpp"

namespace ttnn::operations::experimental::moe {

ttnn::Tensor ExecuteMoE::invoke(const ttnn::Tensor& input_tensor, const ttnn::Tensor& weight_tensor) {
    return ttnn::prim::moe(input_tensor, weight_tensor);
}

}  // namespace ttnn::operations::experimental::moe
