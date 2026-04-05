// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_op_utils.hpp"

#include <bit>
#include <optional>
#include <tt_stl/assert.hpp>
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary::utils {

namespace {

std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::COSH: return "SFPU_OP_COSH_INCLUDE";
        case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
        case UnaryOpType::SELU: return "SFPU_OP_SELU_INCLUDE";
        case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
        case UnaryOpType::SOFTSIGN: return "SFPU_OP_SOFTSIGN_INCLUDE";
        case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
        case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
        case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";
        case UnaryOpType::SOFTSHRINK: return "SFPU_OP_SOFTSHRINK_INCLUDE";
        case UnaryOpType::SINH: return "SFPU_OP_SINH_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}

template <typename T>
std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type,
    std::span<const T> params,
    [[maybe_unused]] const std::string& idst,
    [[maybe_unused]] std::optional<DataType> input_dtype) {
    TT_FATAL(
        is_parametrized_type(op_type),
        "operator should support at least one parameter but op_type {} does not",
        op_type);
    // TODO don't cast T to float when precision needs to be preserved
    [[maybe_unused]] const T param0_raw = params[0];
    [[maybe_unused]] float param0 = static_cast<float>(params[0]);
    switch (op_type) {
        case UnaryOpType::HARDTANH: {
            float min_val = params.size() > 0 ? param0 : -1.0f;
            float max_val = params.size() > 1 ? static_cast<float>(params[1]) : 1.0f;
            return {
                "hardtanh_tile_init();",
                fmt::format(
                    "hardtanh_tile({}, {:#010x}u, {:#010x}u);",
                    idst,
                    std::bit_cast<uint32_t>(min_val),
                    std::bit_cast<uint32_t>(max_val))};
        }
        case UnaryOpType::RPOW:
            return {
                "rpow_tile_init();", fmt::format("rpow_tile({}, {:#010x}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::SOFTSHRINK: {
            float lambda_val = params.size() > 0 ? param0 : 0.5f;
            return {
                "softshrink_tile_init();",
                fmt::format("softshrink_tile({}, {:#010x}u);", idst, std::bit_cast<uint32_t>(lambda_val))};
        }
        default: TT_THROW("unexpected parameterized op type {}", op_type);
    };
}

std::pair<std::string, std::string> get_op_init_and_func_default(
    UnaryOpType op_type, std::string idst, [[maybe_unused]] std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
        case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
        case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
        case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};
        case UnaryOpType::SELU: return {"selu_tile_init();", fmt::format("selu_tile({});", idst)};
        case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
        case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
        case UnaryOpType::SINH: return {"sinh_tile_init();", fmt::format("sinh_tile({});", idst)};
        default: TT_THROW("unexpected op type {}", op_type);
    };
}

template <typename T>
std::map<std::string, std::string> get_defines_impl(
    UnaryOpType op_type,
    std::span<const T> params,
    const std::string& idst,
    std::string init_def,
    std::string func_def,
    std::optional<DataType> input_dtype) {
    std::pair<std::string, std::string> op_init_and_name = get_op_init_and_func(op_type, params, idst, input_dtype);
    std::map<std::string, std::string> defines = {
        {init_def, op_init_and_name.first}, {func_def, op_init_and_name.second}};
    update_macro_defines(op_type, defines);
    return defines;
}
}  // namespace

bool get_op_approx_mode(UnaryOpType op_type) {
    switch (op_type) {
        default: return false;
    }
}

UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "cosh") {
        return UnaryWithParam(UnaryOpType::COSH);
    }
    if (name == "sinh") {
        return UnaryWithParam(UnaryOpType::SINH);
    }
    TT_THROW("Unknown unary op: {}", name);
}

template <typename T>
std::map<std::string, std::string> get_defines(
    UnaryOpType op_type,
    std::optional<std::span<const std::type_identity_t<T>>> params,
    const std::string& id,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    std::string init_def = fmt::format("SFPU_OP_INIT_{}", id);
    std::string func_def = fmt::format("SFPU_OP_FUNC_{}", id);
    return get_defines_impl(op_type, params.value_or(std::span<const T>{}), idst, init_def, func_def, input_dtype);
}

template std::map<std::string, std::string> get_defines<float>(
    UnaryOpType op_type,
    std::optional<std::span<const float>> params,
    const std::string& id,
    const std::string& idst,
    std::optional<DataType> input_dtype);

template std::map<std::string, std::string> get_defines<std::int32_t>(
    UnaryOpType op_type,
    std::optional<std::span<const std::int32_t>> params,
    const std::string& id,
    const std::string& idst,
    std::optional<DataType> input_dtype);

template std::map<std::string, std::string> get_defines<std::uint32_t>(
    UnaryOpType op_type,
    std::optional<std::span<const std::uint32_t>> params,
    const std::string& id,
    const std::string& idst,
    std::optional<DataType> input_dtype);

template <typename T>
std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type, std::span<const T> params, const std::string& idst, std::optional<DataType> input_dtype) {
    return !params.empty() ? get_op_init_and_func_parameterized(op_type, params, idst, input_dtype)
                           : get_op_init_and_func_default(op_type, idst, input_dtype);
}

template std::pair<std::string, std::string> get_op_init_and_func<float>(
    UnaryOpType op_type, std::span<const float> params, const std::string& idst, std::optional<DataType> input_dtype);

template std::pair<std::string, std::string> get_op_init_and_func<std::int32_t>(
    UnaryOpType op_type,
    std::span<const std::int32_t> params,
    const std::string& idst,
    std::optional<DataType> input_dtype);

template std::pair<std::string, std::string> get_op_init_and_func<std::uint32_t>(
    UnaryOpType op_type,
    std::span<const std::uint32_t> params,
    const std::string& idst,
    std::optional<DataType> input_dtype);

std::map<std::string, std::string> get_block_defines(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::string& block_id,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    std::map<std::string, std::string> block_defines;
    std::string block_define;
    for (uint32_t i = 0; i < op_chain.size(); i++) {
        std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
        std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
        block_define += init_def + " " + func_def + " ";
        block_defines.merge(std::visit(
            [&](auto params) {
                return get_defines_impl(op_chain[i].type(), params, idst, init_def, func_def, input_dtype);
            },
            op_chain[i].get_params()));
    }
    block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
    return block_defines;
}

// update split eltwise ops include macros
void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines) {
    defines[get_macro_definition(op_type)] = "1";
}

std::string_view get_compute_kernel_path(UnaryOpType op_type, [[maybe_unused]] std::optional<DataType> input_dtype) {
    switch (op_type) {
        default: return "eltwise_sfpu.cpp";
    }
}

std::uint32_t pack_scalar_runtime_arg_impl(float param, DataType dtype) {
    if (dtype == DataType::UINT32 || dtype == DataType::INT32) {
        return std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(param));
    }
    // This handles the case where dtype is not an integer type
    return std::bit_cast<std::uint32_t>(param);
}

std::uint32_t pack_scalar_runtime_arg_impl(std::uint32_t param, DataType dtype) {
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        return param;
    }
    return std::bit_cast<std::uint32_t>(static_cast<float>(param));
}

std::uint32_t pack_scalar_runtime_arg_impl(std::int32_t param, DataType dtype) {
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        return std::bit_cast<std::uint32_t>(param);
    }
    return std::bit_cast<std::uint32_t>(static_cast<float>(param));
}

uint32_t pack_scalar_runtime_arg(const EltwiseUnaryWithParam& op, size_t index, DataType dtype) {
    return std::visit(
        [index, dtype](const auto& op_specialization) -> uint32_t {
            if (auto param = op_specialization.get_param_if(index)) {
                return pack_scalar_runtime_arg_impl(*param, dtype);
            }
            TT_THROW("Unsupported parameter type for index {}", index);
        },
        op.base);
}
}  // namespace ttnn::operations::unary::utils
