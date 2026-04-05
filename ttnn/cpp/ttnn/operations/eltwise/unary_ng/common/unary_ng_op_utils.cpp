// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <cstdint>
#include <bit>
#include <fmt/core.h>

namespace ttnn::operations::unary_ng {

using tt::tt_metal::DataType;
using unary::EltwiseUnaryWithParam;
using unary::UnaryOpType;

namespace {

std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::COSH: return "SFPU_OP_COSH_INCLUDE";
        case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
        case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
        case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";
        case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    }
}

// Init and func strings for SFPU op chain. Only migrated ops (e.g. ABS) are implemented here.
std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type, const std::string& idst, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::LTZ:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"ltz_tile_init();", fmt::format("ltz_tile_int32({});", idst)};
            }
            return {"ltz_tile_init();", fmt::format("ltz_tile({});", idst)};
        case UnaryOpType::NEZ:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"nez_tile_init();", fmt::format("nez_tile_int32({});", idst)};
            }
            if (input_dtype.has_value() && *input_dtype == DataType::UINT16) {
                return {"nez_tile_init();", fmt::format("nez_tile_uint16({});", idst)};
            }
            if (input_dtype.has_value() && *input_dtype == DataType::UINT32) {
                return {"nez_tile_init();", fmt::format("nez_tile_uint32({});", idst)};
            }
            return {"nez_tile_init();", fmt::format("nez_tile({});", idst)};
        case UnaryOpType::RECIP: return {"recip_tile_init<false>();", fmt::format("recip_tile<false>({});", idst)};
        case UnaryOpType::RELU:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"relu_tile_init();", fmt::format("relu_tile_int32({});", idst)};
            }
            return {"relu_tile_init();", fmt::format("relu_tile({});", idst)};
        case UnaryOpType::RELU6: return {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
        case UnaryOpType::SIGN: return {"sign_tile_init();", fmt::format("sign_tile({});", idst)};
        case UnaryOpType::SIGNBIT:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"signbit_tile_init();", fmt::format("signbit_tile_int32({});", idst)};
            }
            return {"signbit_tile_init();", fmt::format("signbit_tile({});", idst)};
        case UnaryOpType::SIN: return {"sin_tile_init();", fmt::format("sin_tile({});", idst)};
        case UnaryOpType::SQUARE:
            TT_FATAL(input_dtype.has_value(), "SQUARE requires input_dtype");
            if (*input_dtype == DataType::INT32) {
                return {
                    "mul_int_tile_init<DataFormat::Int32>();",
                    fmt::format("mul_int_tile<DataFormat::Int32>({0}, {0}, {0});", idst)};
            }
            if (*input_dtype == DataType::UINT32) {
                return {
                    "mul_int_tile_init<DataFormat::UInt32>();",
                    fmt::format("mul_int_tile<DataFormat::UInt32>({0}, {0}, {0});", idst)};
            }
            if (*input_dtype == DataType::UINT16) {
                return {
                    "mul_int_tile_init<DataFormat::UInt16>();",
                    fmt::format("mul_int_tile<DataFormat::UInt16>({0}, {0}, {0});", idst)};
            }
            return {"square_tile_init();", fmt::format("square_tile({});", idst)};
        case UnaryOpType::TAN: return {"tan_tile_init();", fmt::format("tan_tile({});", idst)};
        case UnaryOpType::TILED_PROD: return {"tiled_prod_tile_init();", fmt::format("tiled_prod_tile({});", idst)};
        case UnaryOpType::BITWISE_NOT: return {"bitwise_not_tile_init();", fmt::format("bitwise_not_tile({});", idst)};
        case UnaryOpType::ALT_COMPLEX_ROTATE90:
            return {"alt_complex_rotate90_tile_init();", fmt::format("alt_complex_rotate90_tile({});", idst)};
        case UnaryOpType::FLOOR: return {"rounding_op_tile_init();", fmt::format("floor_tile({});", idst)};
        case UnaryOpType::CEIL: return {"rounding_op_tile_init();", fmt::format("ceil_tile({});", idst)};
        case UnaryOpType::TRUNC: return {"rounding_op_tile_init();", fmt::format("trunc_tile({});", idst)};
        case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
        case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};
        case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
        case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
        case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
        case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
        case UnaryOpType::SWISH: return {"swish_tile_init();", fmt::format("swish_tile({});", idst)};
        default: TT_FATAL(false, "Undefined unary_ng op type {}", op_type);
    }
}

std::map<std::string, std::string> get_defines_impl(
    UnaryOpType op_type,
    const std::string& idst,
    const std::string& init_def,
    const std::string& func_def,
    std::optional<DataType> input_dtype) {
    auto [init_str, func_str] = get_op_init_and_func(op_type, idst, input_dtype);
    std::map<std::string, std::string> defines = {
        {init_def, init_str},
        {func_def, func_str},
    };
    defines[get_macro_definition(op_type)] = "1";
    return defines;
}

}  // namespace

std::string_view get_compute_kernel_path(UnaryOpType op_type, [[maybe_unused]] std::optional<DataType> input_dtype) {
    switch (op_type) {
        default: return "eltwise_sfpu.cpp";
    }
}

uint32_t pack_scalar_runtime_arg_impl(float param, DataType dtype) {
    if (dtype == DataType::UINT32 || dtype == DataType::INT32) {
        return std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(param));
    }
    return std::bit_cast<std::uint32_t>(param);
}

uint32_t pack_scalar_runtime_arg_impl(std::uint32_t param, DataType dtype) {
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        return param;
    }
    return std::bit_cast<std::uint32_t>(static_cast<float>(param));
}

uint32_t pack_scalar_runtime_arg_impl(std::int32_t param, DataType dtype) {
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

bool get_op_approx_mode(UnaryOpType) { return false; }

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
        block_defines.merge(get_defines_impl(op_chain[i].type(), idst, init_def, func_def, input_dtype));
    }
    block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
    return block_defines;
}

}  // namespace ttnn::operations::unary_ng
