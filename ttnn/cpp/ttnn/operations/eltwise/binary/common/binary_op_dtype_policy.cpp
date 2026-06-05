// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op_dtype_policy.hpp"

#include <algorithm>

namespace ttnn::operations::binary::dtype_policy {

using tt::tt_metal::DataType;

std::span<const DataType> supported_input_dtypes(BinaryOpType op) {
    using namespace dtype_sets;
    switch (op) {
        case BinaryOpType::ADD:
        case BinaryOpType::SUB:
        case BinaryOpType::MUL:
        case BinaryOpType::RSUB:
        case BinaryOpType::SQUARED_DIFFERENCE:
        case BinaryOpType::LOGICAL_AND:
        case BinaryOpType::LOGICAL_OR:
        case BinaryOpType::LOGICAL_XOR: return arithmetic_fpu;
        case BinaryOpType::GT:
        case BinaryOpType::LT:
        case BinaryOpType::LE:
        case BinaryOpType::GE:
        case BinaryOpType::EQ:
        case BinaryOpType::NE: return relational;
        case BinaryOpType::DIV:
        case BinaryOpType::REMAINDER:
        case BinaryOpType::FMOD:
        case BinaryOpType::MAXIMUM:
        case BinaryOpType::MINIMUM:
        case BinaryOpType::ISCLOSE: return float_and_int32;
        case BinaryOpType::BITWISE_XOR:
        case BinaryOpType::BITWISE_AND:
        case BinaryOpType::BITWISE_OR:
        case BinaryOpType::LEFT_SHIFT:
        case BinaryOpType::RIGHT_SHIFT: return bitwise_shift;
        case BinaryOpType::LOGICAL_RIGHT_SHIFT: return logical_right_shift;
        case BinaryOpType::GCD:
        case BinaryOpType::LCM:
        case BinaryOpType::DIV_FLOOR:
        case BinaryOpType::DIV_TRUNC:
        case BinaryOpType::REQUANT:
        case BinaryOpType::DEQUANT: return int32_only;
        case BinaryOpType::LOGADDEXP:
        case BinaryOpType::LOGADDEXP2:
        case BinaryOpType::LDEXP:
        case BinaryOpType::BIAS_GELU:
        case BinaryOpType::QUANT:
        case BinaryOpType::POWER:
        case BinaryOpType::XLOGY:
        case BinaryOpType::ATAN2:
        case BinaryOpType::HYPOT: return float_only;
        case BinaryOpType::WHERE_TST:
        case BinaryOpType::WHERE_TTS: return where;
        case BinaryOpType::ADDALPHA:
        case BinaryOpType::SUBALPHA: return unsupported;
        default: __builtin_unreachable();
    }
}

bool is_supported(BinaryOpType op, DataType dtype) {
    const auto supported_dtypes = supported_input_dtypes(op);
    return std::ranges::find(supported_dtypes, dtype) != supported_dtypes.end();
}

}  // namespace ttnn::operations::binary::dtype_policy
