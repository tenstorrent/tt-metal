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

inline constexpr std::array int32_only{DT::INT32};

inline constexpr std::array bitwise_shift{DT::UINT32, DT::UINT16, DT::INT32};

inline constexpr std::array logical_right_shift{DT::UINT32, DT::INT32};

// FPU arithmetic and logical ops (ADD, SUB, MUL, LOGICAL_*, etc.).
inline constexpr std::array arithmetic_fpu{
    DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::UINT32, DT::UINT16, DT::INT32};

inline constexpr std::array relational{
    DT::BFLOAT16, DT::FLOAT32, DT::BFLOAT8_B, DT::BFLOAT4_B, DT::UINT32, DT::UINT8, DT::UINT16, DT::INT32};

// SFPU where kernel: where_tile<DataFormat::...> for bf16, fp32, uint32, int32.
inline constexpr std::array where{DT::BFLOAT16, DT::FLOAT32, DT::UINT32, DT::INT32};

// Enum values that are not dispatched to binary_ng (e.g. ADDALPHA -> ADD).
inline constexpr std::array<DT, 0> unsupported{};

}  // namespace dtype_sets

std::span<const tt::tt_metal::DataType> supported_input_dtypes(BinaryOpType op);

bool is_supported(BinaryOpType op, tt::tt_metal::DataType dtype);

}  // namespace ttnn::operations::binary::dtype_policy
