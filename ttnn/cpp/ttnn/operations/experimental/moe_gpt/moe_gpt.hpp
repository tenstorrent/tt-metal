// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::moe_gpt {

struct ExecuteMoEGPT {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& w0_w1_tensor,
        const ttnn::Tensor& w2_tensor,
        const ttnn::Tensor& bias0_tensor,
        const ttnn::Tensor& bias1_tensor,
        const ttnn::Tensor& bias2_tensor,
        const ttnn::Tensor& output_tensor,
        const uint32_t num_experts,
        const uint32_t layer_id);
};

}  // namespace ttnn::operations::experimental::moe_gpt

namespace ttnn::experimental {
constexpr auto moe_gpt =
    ttnn::register_operation<"ttnn::experimental::moe_gpt", ttnn::operations::experimental::moe_gpt::ExecuteMoEGPT>();
}  // namespace ttnn::experimental
