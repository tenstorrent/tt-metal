// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::core {

Tensor ToDtype::invoke(const ttnn::Tensor& input_tensor, const ttnn::DataType& dtype) {
    return tt::tt_metal::to_dtype(input_tensor, dtype);
};

}  // namespace ttnn::operations::core
