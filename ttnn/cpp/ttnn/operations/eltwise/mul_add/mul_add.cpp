// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mul_add.hpp"
#include "device/mul_add_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::mul_add {

ttnn::Tensor MulAddOperation::invoke(
    const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, const ttnn::Tensor& input_tensor_c) {
    return ttnn::prim::muladd(input_tensor_a, input_tensor_b, input_tensor_c);
}

}  // namespace ttnn::operations::mul_add
