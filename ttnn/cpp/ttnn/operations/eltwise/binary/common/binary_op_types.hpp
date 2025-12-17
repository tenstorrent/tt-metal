// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::binary {

enum class BinaryOpType : std::uint8_t {
    ADD,
    SUB,
    MUL,
    GT,
    LT,
    LE,
    GE,
    EQ,
    NE,
    SQUARED_DIFFERENCE,
    BIAS_GELU,
    LOGADDEXP,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    LDEXP,
    LOGADDEXP2,
    DIV,
    DIV_FLOOR,
    DIV_TRUNC,
    RSUB,
    POWER,
    BITWISE_XOR,
    BITWISE_AND,
    BITWISE_OR,
    LEFT_SHIFT,
    RIGHT_SHIFT,
    LOGICAL_RIGHT_SHIFT,
    QUANT,
    REQUANT,
    DEQUANT,
    MAXIMUM,
    MINIMUM,
    GCD,
    LCM,
    ADDALPHA,
    SUBALPHA,
    XLOGY,
    HYPOT,
    WHERE_TST,
    WHERE_TTS,
};

}  // namespace ttnn::operations::binary
