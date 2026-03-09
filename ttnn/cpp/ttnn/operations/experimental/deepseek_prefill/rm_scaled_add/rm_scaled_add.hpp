// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::deepseek_prefill {

struct RmScaledAddOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        float scale);
};

}  // namespace operations::experimental::deepseek_prefill

constexpr auto rm_scaled_add =
    ttnn::register_operation<"ttnn::experimental::rm_scaled_add", ttnn::operations::experimental::deepseek_prefill::RmScaledAddOperation>();

}  // namespace ttnn
