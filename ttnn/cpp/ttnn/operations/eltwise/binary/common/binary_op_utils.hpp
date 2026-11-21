// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "binary_op_types.hpp"

namespace tt::tt_metal {
enum class DataType;
}

namespace ttnn::operations::binary::utils {
bool is_typecast(tt::tt_metal::DataType input, tt::tt_metal::DataType output);

bool is_quant_op(BinaryOpType op);

// True when packing value as a datum of the 32-bit integer dtype keeps it unchanged: a finite integer
// within that dtype's range. Scalar runtime args are packed with a static_cast, which otherwise truncates
// fractions and is undefined for out-of-range or non-finite values.
bool scalar_fits_integer_dtype(float value, tt::tt_metal::DataType dtype);

// Returns true when dtype_a and dtype_b form a supported operand pair for op.
// Most ops require matching dtypes; quant ops additionally require float32 scale (B).
bool is_dtype_combination_supported(BinaryOpType op, tt::tt_metal::DataType dtype_a, tt::tt_metal::DataType dtype_b);

}  // namespace ttnn::operations::binary::utils
