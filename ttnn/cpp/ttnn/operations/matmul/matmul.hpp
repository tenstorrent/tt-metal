// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation);

ttnn::Tensor bound_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const struct Matmul& parameters,
    const uint8_t& queue_id);

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn

namespace ttnn {
    using ttnn::operations::matmul::matmul;
}
