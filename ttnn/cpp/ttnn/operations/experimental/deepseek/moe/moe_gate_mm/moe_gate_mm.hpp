// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::deepseek::moe {

Tensor moe_gate_mm(
    const Tensor& input_tensor,
    const Tensor& w_tensor,
    const Tensor& output_tensor,
    uint32_t layer_id,
    uint32_t column_id);

}  // namespace ttnn::experimental::deepseek::moe
