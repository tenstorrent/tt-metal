// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>

#include "unary_op_types.hpp"
#include "ttnn/tensor/types.hpp"
namespace ttnn::operations::unary::utils {

UnaryWithParam string_to_unary_with_param(const std::string& name);

std::string unary_with_param_to_string(const UnaryWithParam& unary_op);

bool get_op_approx_mode(UnaryOpType op_type);
using DataType = tt::tt_metal::DataType;

template <typename T = float>
std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type,
    std::span<const T> params = {},
    const std::string& idst = "0",
    std::optional<DataType> input_dtype = std::nullopt);

// type_identity_t suppresses template argument deduction
// this allows get_defines(...) without a template list to use the default type float
template <typename T = float>
std::map<std::string, std::string> get_defines(
    UnaryOpType op_type,
    std::optional<std::span<const std::type_identity_t<T>>> params = std::nullopt,
    const std::string& id = "0",
    const std::string& idst = "0",
    std::optional<DataType> input_dtype = std::nullopt);

std::map<std::string, std::string> get_block_defines(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::string& block_id = "0",
    const std::string& idst = "0",
    std::optional<DataType> input_dtype = std::nullopt);

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
        case UnaryOpType::UNARY_EQ:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::UNARY_GE:
        case UnaryOpType::UNARY_LE:
        case UnaryOpType::TYPECAST:
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
        case UnaryOpType::RIGHT_SHIFT:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FILL:
        case UnaryOpType::ROUND:
        case UnaryOpType::SIGMOID:
        case UnaryOpType::PRELU_SFPU:
        case UnaryOpType::FMOD:
        case UnaryOpType::MINIMUM:
        case UnaryOpType::MAXIMUM:
        case UnaryOpType::LOG:
        case UnaryOpType::LOG10:
        case UnaryOpType::LOG2:
        case UnaryOpType::LOG1P:
        case UnaryOpType::SOFTSHRINK:
        case UnaryOpType::HARDSHRINK:
        case UnaryOpType::WHERE_TSS:
        case UnaryOpType::CELU:
        case UnaryOpType::HARDTANH:
        case UnaryOpType::THRESHOLD:
        case UnaryOpType::CLAMP_TSS:
        case UnaryOpType::SELU: return true;
        default: return false;
    }
    return false;
}

void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines);

std::string get_compute_kernel_path(
    UnaryOpType op_type, const std::string& compute_root, std::optional<DataType> input_dtype = std::nullopt);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype);

}  // namespace ttnn::operations::unary::utils
