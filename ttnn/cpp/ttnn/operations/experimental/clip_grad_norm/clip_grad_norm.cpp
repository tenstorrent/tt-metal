// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/clip_grad_norm_device_operation.hpp"
#include "clip_grad_norm.hpp"

namespace ttnn::operations::experimental {

Tensor ClipGradNormOperation::invoke(const Tensor& input_tensor, float max_norm, float p, float eps) {
    return ttnn::prim::clip_grad_norm(input_tensor, max_norm, p, eps, DataType::BFLOAT16);
}

}  // namespace ttnn::operations::experimental
