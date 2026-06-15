// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <span>

#include "binary_op_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::binary::dtype_policy {

namespace dtype_sets {

using DT = tt::tt_metal::DataType;

inline constexpr std::array float_only{DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B};

inline constexpr std::array float_and_int32{DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::INT32};

inline constexpr std::array maximum_minimum{
    DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::UINT32, DT::INT32};

inline constexpr std::array int32_only{DT::INT32};

inline constexpr std::array bitwise_shift{DT::UINT32, DT::UINT16, DT::INT32};

inline constexpr std::array logical_right_shift{DT::UINT32, DT::INT32};

// FPU arithmetic and logical ops (ADD, SUB, MUL, LOGICAL_*, etc.).
inline constexpr std::array arithmetic_fpu{
    DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::UINT32, DT::UINT16, DT::INT32};

inline constexpr std::array relational{
    DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::UINT32, DT::UINT8, DT::UINT16, DT::INT32};

// SFPU where kernel: explicit Int32/UInt32/Float32; other floats use Float16_b tile path.
inline constexpr std::array where{DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::UINT32, DT::INT32};

// BFLOAT16 plus block-float tile dtypes.
inline constexpr std::array bfloat_tile{DT::BFLOAT16, DT::BFLOAT8_B, DT::BFLOAT4_B};

// Enum values that are not dispatched to binary_ng (e.g. ADDALPHA -> ADD).
inline constexpr std::array<DT, 0> unsupported{};

}  // namespace dtype_sets

// Dtypes allowed for tensor A. For symmetric ops both operands must match and use this set.
// For asymmetric quant operand pairs see is_quant_operand_pair_supported.
std::span<const tt::tt_metal::DataType> supported_tensor_a_dtypes(BinaryOpType op);

bool is_supported(BinaryOpType op, tt::tt_metal::DataType dtype);

bool is_bfloat_tile_dtype(tt::tt_metal::DataType dtype);

bool is_mixed_bfloat_tile_pair(tt::tt_metal::DataType dtype_a, tt::tt_metal::DataType dtype_b);

bool supports_mixed_bfloat_tile_inputs(BinaryOpType op);

// QUANT: float A × float32 scale B; REQUANT/DEQUANT: int32 A × float32 scale B.
bool is_quant_operand_pair_supported(BinaryOpType op, tt::tt_metal::DataType dtype_a, tt::tt_metal::DataType dtype_b);

}  // namespace ttnn::operations::binary::dtype_policy
