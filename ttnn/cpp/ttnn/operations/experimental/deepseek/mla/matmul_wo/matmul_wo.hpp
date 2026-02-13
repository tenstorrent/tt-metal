// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::deepseek::mla {

struct ExecuteMatmulWO {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& w_tensor,
        const ttnn::Tensor& output_tensor,
        const uint32_t layer_id);
};

}  // namespace ttnn::operations::experimental::deepseek::mla

namespace ttnn::experimental {
constexpr auto matmul_wo = ttnn::register_operation<
    "ttnn::experimental::deepseek::mla::matmul_wo",
    ttnn::operations::experimental::deepseek::mla::ExecuteMatmulWO>();
}  // namespace ttnn::experimental
