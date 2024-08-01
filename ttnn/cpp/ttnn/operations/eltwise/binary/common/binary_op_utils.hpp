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
std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> in_dtype = std::nullopt,
    const std::optional<tt::tt_metal::DataType> out_dtype = std::nullopt,
    const std::optional<ttnn::operations::unary::FusedActivations> fused_activations = std::nullopt,
    const std::optional<ttnn::operations::unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

}
