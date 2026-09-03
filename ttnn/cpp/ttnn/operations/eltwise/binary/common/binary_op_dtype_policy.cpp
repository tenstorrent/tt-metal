// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op_dtype_policy.hpp"

#include <algorithm>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::binary::dtype_policy {

using tt::tt_metal::DataType;

std::span<const DataType> supported_tensor_a_dtypes(BinaryOpType op) {
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
        case BinaryOpType::REMAINDER:
        case BinaryOpType::MAXIMUM:
        case BinaryOpType::MINIMUM: return float_and_int32_uint32;
        case BinaryOpType::DIV:
        case BinaryOpType::FMOD:
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
        case BinaryOpType::DIV_TRUNC: return int32_only;
        case BinaryOpType::REQUANT:
        case BinaryOpType::DEQUANT: return requant_dequant;
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
        default: TT_THROW("Binary op type {} is missing from dtype policy", op);
    }
}

bool is_supported(BinaryOpType op, DataType dtype) {
    const auto supported_dtypes = supported_tensor_a_dtypes(op);
    return std::ranges::find(supported_dtypes, dtype) != supported_dtypes.end();
}

bool is_mixed_float_dtype(DataType dtype) {
    using namespace dtype_sets;
    return std::ranges::find(mixed_float, dtype) != mixed_float.end();
}

bool is_mixed_float_pair(DataType dtype_a, DataType dtype_b) {
    return dtype_a != dtype_b && is_mixed_float_dtype(dtype_a) && is_mixed_float_dtype(dtype_b);
}

bool supports_mixed_float_inputs(BinaryOpType op) {
    switch (op) {
        case BinaryOpType::ADD:
        case BinaryOpType::SUB:
        case BinaryOpType::MUL:
        case BinaryOpType::RSUB:
        case BinaryOpType::SQUARED_DIFFERENCE:
        case BinaryOpType::LOGICAL_AND:
        case BinaryOpType::LOGICAL_OR:
        case BinaryOpType::LOGICAL_XOR:
        case BinaryOpType::GT:
        case BinaryOpType::LT:
        case BinaryOpType::LE:
        case BinaryOpType::GE:
        case BinaryOpType::EQ:
        case BinaryOpType::NE:
        case BinaryOpType::DIV: return true;
        default: return false;
    }
}

bool is_quant_operand_pair_supported(BinaryOpType op, DataType dtype_a, DataType dtype_b) {
    return is_supported(op, dtype_a) && dtype_b == DataType::FLOAT32;
}

}  // namespace ttnn::operations::binary::dtype_policy
