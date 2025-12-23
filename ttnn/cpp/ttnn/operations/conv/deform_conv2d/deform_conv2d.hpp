// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::conv {
namespace deform_conv2d {
struct DeformConv2dOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const ttnn::Tensor& offset_tensor,
        int stride,
        uint32_t padding,
        int dilation,
        int groups,
        int offset_groups);
};
}  // namespace deform_conv2d
}  // namespace operations::conv
constexpr auto deform_conv2d =
    ttnn::register_operation<"ttnn::deform_conv2d", operations::conv::deform_conv2d::DeformConv2dOperation>();
}  // namespace ttnn
