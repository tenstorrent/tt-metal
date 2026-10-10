// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

namespace ttnn::operations::binary {

enum class BinaryOpType {
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
    REMAINDER,
    FMOD,
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
    ATAN2,
    NEXTAFTER,
    WHERE_TST,
    WHERE_TTS,
    ISCLOSE,
};

// Parameters for ops that need more than their BinaryOpType to describe what to compile. One
// struct per such op, gathered in the variant below so a generic interface can carry any op's
// parameters. Plain aggregates rather than a class hierarchy: these ride in the operation
// attributes, which the program cache hashes structurally.
struct BiasGeluParams {
    bool fast_and_approximate = false;
};

using BinaryOpParams = std::variant<BiasGeluParams>;

}  // namespace ttnn::operations::binary
