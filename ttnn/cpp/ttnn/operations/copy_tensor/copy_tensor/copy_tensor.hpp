// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::copy_tensor {

struct ExecuteCopyTensor {
    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& dst_tensor);
};

}  // namespace operations::copy_tensor

constexpr auto copy_tensor = ttnn::
    register_operation_with_auto_launch_op<"ttnn::copy_tensor", ttnn::operations::copy_tensor::ExecuteCopyTensor>();

}  // namespace ttnn
