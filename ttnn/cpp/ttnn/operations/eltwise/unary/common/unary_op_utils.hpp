// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>

#include "unary_op_types.hpp"

namespace tt::tt_metal {
enum class DataType;
}

namespace ttnn::operations::unary::utils {

UnaryWithParam string_to_unary_with_param(const std::string& name);

bool get_op_approx_mode(UnaryOpType op_type);

std::pair<std::string, std::string> get_op_init_and_func(UnaryOpType op_type,
                                                         const std::vector<float>& params = {},
                                                         const std::string& idst = "0");

std::map<std::string, std::string> get_defines(UnaryOpType op_type,
                                               const std::optional<std::vector<float>>& params = std::nullopt,
                                               const std::string& id = "0",
                                               const std::string& idst = "0");

std::map<std::string, std::string> get_block_defines(const std::vector<UnaryWithParam>& op_chain,
                                                     const std::string& block_id = "0",
                                                     const std::string& idst = "0");

template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::POWER:
        case UnaryOpType::LEAKY_RELU:
        case UnaryOpType::ELU:
        case UnaryOpType::GELU:
        case UnaryOpType::RSQRT:
        case UnaryOpType::HEAVISIDE:
        case UnaryOpType::ERF:
        case UnaryOpType::ERFC:
        case UnaryOpType::RSUB:
        case UnaryOpType::RDIV:
        case UnaryOpType::EXP:
        case UnaryOpType::SOFTPLUS:
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU:
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::TYPECAST:
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
        case UnaryOpType::RIGHT_SHIFT:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::REMAINDER:
        case UnaryOpType::DROPOUT:
        case UnaryOpType::FMOD: return true;
        default: return false;
    }
    return false;
}

}  // namespace ttnn::operations::unary::utils
