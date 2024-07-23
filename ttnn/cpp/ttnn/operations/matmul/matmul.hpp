// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_op.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace ttnn {

using MatmulMultiCoreReuseProgramConfig = tt::operations::primary::MatmulMultiCoreReuseProgramConfig;
using MatmulMultiCoreReuseMultiCastProgramConfig = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig;
using MatmulMultiCoreReuseMultiCast1DProgramConfig =
    tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig;
// MatmulProgramConfig is the Union of the above types
using MatmulProgramConfig = tt::operations::primary::MatmulProgramConfig;

using ttnn::operations::unary::UnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

namespace operations {
namespace matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation);

ttnn::Tensor matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const struct tt::operations::primary::Matmul& parameters);

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
