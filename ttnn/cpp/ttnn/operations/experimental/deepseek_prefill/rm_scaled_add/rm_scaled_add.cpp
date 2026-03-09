// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/rm_scaled_add_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/rm_scaled_add/rm_scaled_add.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::deepseek_prefill {

ttnn::Tensor RmScaledAddOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    float scale) {
    return ttnn::prim::rm_scaled_add(input_a, input_b, scale);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill
