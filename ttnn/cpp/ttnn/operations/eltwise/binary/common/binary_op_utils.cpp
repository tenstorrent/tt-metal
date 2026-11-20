// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op_utils.hpp"

#include <cmath>

#include "binary_op_dtype_policy.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary::utils {

bool is_typecast(tt::tt_metal::DataType input, tt::tt_metal::DataType output) {
    using enum tt::tt_metal::DataType;

    return (input == BFLOAT4_B && output == INT32) || (input == BFLOAT4_B && output == UINT16) ||
           (input == BFLOAT4_B && output == UINT32) || (input == BFLOAT8_B && output == INT32) ||
           (input == BFLOAT8_B && output == UINT16) || (input == BFLOAT8_B && output == UINT32) ||
           (input == BFLOAT16 && output == INT32) || (input == BFLOAT16 && output == UINT16) ||
           (input == BFLOAT16 && output == UINT32) || (input == FLOAT32 && output == BFLOAT16) ||
           (input == FLOAT32 && output == INT32) || (input == FLOAT32 && output == UINT16) ||
           (input == FLOAT32 && output == UINT32) || (input == INT32 && output == BFLOAT4_B) ||
           (input == INT32 && output == BFLOAT8_B) || (input == INT32 && output == BFLOAT16) ||
           (input == INT32 && output == FLOAT32) || (input == UINT16 && output == BFLOAT4_B) ||
           (input == UINT16 && output == BFLOAT8_B) || (input == UINT16 && output == BFLOAT16) ||
           (input == UINT16 && output == FLOAT32) || (input == UINT16 && output == UINT32) ||
           (input == UINT32 && output == BFLOAT4_B) || (input == UINT32 && output == BFLOAT8_B) ||
           (input == UINT32 && output == BFLOAT16) || (input == UINT32 && output == FLOAT32) ||
           (input == UINT16 && output == INT32) || (input == INT32 && output == UINT16) ||
           (input == UINT32 && output == UINT16);
}

bool scalar_fits_integer_dtype(float value, tt::tt_metal::DataType dtype) {
    if (!std::isfinite(value) || std::trunc(value) != value) {
        return false;
    }
    // The bounds are powers of two because those are exactly representable as floats, so the comparison
    // cannot round the limit it is testing -- INT32_MAX is not a float32, 2^31 is.
    if (dtype == tt::tt_metal::DataType::INT32) {
        return value >= -2147483648.0f && value < 2147483648.0f;
    }
    if (dtype == tt::tt_metal::DataType::INT8) {
        return value >= -128.0f && value < 128.0f;
    }
    if (dtype == tt::tt_metal::DataType::UINT8) {
        return value >= 0.0f && value < 256.0f;
    }
    return value >= 0.0f && value < 4294967296.0f;
}

bool is_quant_op(BinaryOpType op) {
    return op == BinaryOpType::QUANT || op == BinaryOpType::DEQUANT || op == BinaryOpType::REQUANT;
}

namespace {

bool is_isclose_mixed_dtype_pair(DataType dtype_a, DataType dtype_b) {
    return (dtype_a == DataType::FLOAT32 && dtype_b == DataType::BFLOAT16) ||
           (dtype_a == DataType::BFLOAT16 && dtype_b == DataType::FLOAT32);
}

}  // namespace

bool is_dtype_combination_supported(BinaryOpType op, DataType dtype_a, DataType dtype_b) {
    if (is_quant_op(op)) {
        return dtype_policy::is_quant_operand_pair_supported(op, dtype_a, dtype_b);
    }

    if (op == BinaryOpType::ISCLOSE) {
        return dtype_a == dtype_b ? dtype_policy::is_supported(op, dtype_a)
                                  : is_isclose_mixed_dtype_pair(dtype_a, dtype_b);
    }

    if (dtype_a == dtype_b) {
        return dtype_policy::is_supported(op, dtype_a);
    }

    if (dtype_policy::is_mixed_float_pair(dtype_a, dtype_b) && dtype_policy::supports_mixed_float_inputs(op)) {
        return dtype_policy::is_supported(op, dtype_a) && dtype_policy::is_supported(op, dtype_b);
    }

    return false;
}

}  // namespace ttnn::operations::binary::utils
