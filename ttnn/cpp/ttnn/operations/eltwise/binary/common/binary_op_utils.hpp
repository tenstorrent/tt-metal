// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
