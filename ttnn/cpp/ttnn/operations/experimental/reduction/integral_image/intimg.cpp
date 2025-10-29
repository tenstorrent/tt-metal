// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/intimg_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include "intimg.hpp"

namespace ttnn::operations::experimental::reduction {

void IntImgOperation::validate(const Tensor& input_tensor) {
    const auto& shape = input_tensor.logical_shape();
    const auto& dtype = input_tensor.dtype();
    TT_FATAL(shape.rank() == 4, "Input tensor's rank must be 4, it is {} instead", shape.rank());

    TT_FATAL(shape[0] == 1, "Only one batch expected, there are {} batches instead", shape[0]);

    TT_FATAL(
        dtype == DataType::BFLOAT16,
        "Only {} is supported, but {} has been provided",
        enchantum::to_string(DataType::BFLOAT16),
        enchantum::to_string(dtype));
}

Tensor IntImgOperation::invoke(const Tensor& input_tensor) { return ttnn::prim::intimg(input_tensor); }

}  // namespace ttnn::operations::experimental::reduction
