// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>

#include "binary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace tt::tt_metal {
enum class DataType;
}

namespace ttnn::operations::binary::utils {
bool is_typecast(tt::tt_metal::DataType input, tt::tt_metal::DataType output);

bool is_quant_op(BinaryOpType op);

// Returns true when dtype_a and dtype_b form a supported operand pair for op.
// Most ops require matching dtypes; quant ops additionally require float32 scale (B).
bool is_dtype_combination_supported(BinaryOpType op, tt::tt_metal::DataType dtype_a, tt::tt_metal::DataType dtype_b);

std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    std::optional<tt::tt_metal::DataType> input_dtype = std::nullopt,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt,
    const std::optional<ttnn::operations::unary::EltwiseFusedActivations>& fused_activations = std::nullopt,
    const std::optional<ttnn::operations::unary::EltwiseUnaryWithParam>& input_tensor_a_activation = std::nullopt);

std::map<std::string, std::string> get_defines_fp32(
    BinaryOpType op_type,
    std::optional<tt::tt_metal::DataType> input_a_dtype = std::nullopt,
    std::optional<tt::tt_metal::DataType> input_b_dtype = std::nullopt,
    const std::optional<ttnn::operations::unary::EltwiseFusedActivations>& fused_activations = std::nullopt,
    const std::optional<ttnn::operations::unary::EltwiseUnaryWithParam>& input_tensor_a_activation = std::nullopt);

}  // namespace ttnn::operations::binary::utils
