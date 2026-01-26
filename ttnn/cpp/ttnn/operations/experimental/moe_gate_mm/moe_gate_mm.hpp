// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::moe_gate_mm {

struct ExecuteMoEGateMM {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& w_tensor,
        const ttnn::Tensor& output_tensor,
        const uint32_t layer_id);
};

}  // namespace ttnn::operations::experimental::moe_gate_mm

namespace ttnn::experimental {
constexpr auto moe_gate_mm = ttnn::register_operation<
    "ttnn::experimental::moe_gate_mm",
    ttnn::operations::experimental::moe_gate_mm::ExecuteMoEGateMM>();
}  // namespace ttnn::experimental
