// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>
#include <span>

#include "unary_op_types.hpp"
#include "ttnn/tensor/types.hpp"
namespace ttnn::operations::unary::utils {

UnaryWithParam string_to_unary_with_param(const std::string& name);

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
        default: return false;
    }
    return false;
}

void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines);

std::string_view get_compute_kernel_path(UnaryOpType op_type, std::optional<DataType> input_dtype = std::nullopt);

uint32_t pack_scalar_runtime_arg_impl(float param, DataType dtype);
uint32_t pack_scalar_runtime_arg_impl(std::uint32_t param, DataType dtype);
uint32_t pack_scalar_runtime_arg_impl(std::int32_t param, DataType dtype);

uint32_t pack_scalar_runtime_arg(const EltwiseUnaryWithParam& op, size_t index, DataType dtype);
}  // namespace ttnn::operations::unary::utils
