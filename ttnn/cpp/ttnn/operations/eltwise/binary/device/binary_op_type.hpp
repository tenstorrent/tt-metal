// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <optional>
#include <string>

#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    GT,
    LT,
    LTE,
    GTE,
    EQ,
    NE,
    SQUARED_DIFFERENCE,
    BIAS_GELU,
    LOGADDEXP,
    LOGICAL_AND,
    LOGICAL_OR,
    LDEXP,
    LOGADDEXP2,
    DIV_FAST
};

using FusedActivations = std::vector<tt::tt_metal::UnaryWithParam>;

namespace utils {

std::map<string, string> get_defines(
    BinaryOpType op_type,
    const std::optional<DataType> in_dtype = std::nullopt,
    const std::optional<DataType> out_dtype = std::nullopt,
    const std::optional<FusedActivations> fused_activations = std::nullopt);

}  // namespace utils

}  // namespace ttnn::operations::binary
