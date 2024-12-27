// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::binary_ng {

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
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    LDEXP,
    LOGADDEXP,
    LOGADDEXP2,
    DIV,
    RSUB,
    POWER,
    BITWISE_XOR,
    BITWISE_AND,
    BITWISE_OR,
    LEFT_SHIFT,
    RIGHT_SHIFT
};
}
