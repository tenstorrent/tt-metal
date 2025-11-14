// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

struct ClipGradNormOperation {
    static Tensor invoke(const Tensor& input_tensor, float max_norm, float p = 2.0f, float eps = 1e-12f);
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto clip_grad_norm = ttnn::
    register_operation<"ttnn::experimental::clip_grad_norm", ttnn::operations::experimental::ClipGradNormOperation>();
}  // namespace ttnn::experimental
