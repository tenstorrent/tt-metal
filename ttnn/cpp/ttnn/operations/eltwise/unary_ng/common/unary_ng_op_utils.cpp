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
        case UnaryOpType::ASINH:
        case UnaryOpType::ATANH:
        case UnaryOpType::COS:
        case UnaryOpType::ACOSH:
        case UnaryOpType::COSH:
        case UnaryOpType::SINH:
        case UnaryOpType::SIN:
        case UnaryOpType::TAN: return "SFPU_OP_TRIG_FAMILY_INCLUDE";
        case UnaryOpType::NEG: return "SFPU_OP_NEG_INCLUDE";
        case UnaryOpType::ERFINV: return "SFPU_OP_ERFINV_INCLUDE";
        case UnaryOpType::I0: return "SFPU_OP_I0_INCLUDE";
        case UnaryOpType::I1: return "SFPU_OP_I1_INCLUDE";
        case UnaryOpType::ISFINITE:
        case UnaryOpType::ISINF:
        case UnaryOpType::ISNAN:
        case UnaryOpType::ISNEGINF:
        case UnaryOpType::ISPOSINF: return "SFPU_OP_ISINF_ISNAN_INCLUDE";
        case UnaryOpType::GEZ:
        case UnaryOpType::GTZ:
        case UnaryOpType::LEZ:
        case UnaryOpType::LTZ:
        case UnaryOpType::NEZ: return "SFPU_OP_UNARY_COMP_INCLUDE";
        case UnaryOpType::LOGICAL_NOT_UNARY: return "SFPU_OP_LOGICAL_NOT_INCLUDE";
        case UnaryOpType::RECIP: return "SFPU_OP_RECIP_INCLUDE";
        case UnaryOpType::RELU:
        case UnaryOpType::RELU6: return "SFPU_OP_RELU_FAMILY_INCLUDE";
        case UnaryOpType::BITWISE_NOT: return "SFPU_OP_BITWISE_NOT_INCLUDE";
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::TRUNC:
        case UnaryOpType::FRAC: return "SFPU_OP_ROUND_FAMILY_INCLUDE";
        case UnaryOpType::HARDSIGMOID:
        case UnaryOpType::SOFTSIGN: return "SFPU_OP_ACTIVATIONS_INCLUDE";
        case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
        case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    }
}

// Init and func strings for SFPU op chain. Only migrated ops (e.g. ABS) are implemented here.
std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type, const std::string& idst, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::ABS: return {"abs_tile_init();", fmt::format("abs_tile({});", idst)};
        case UnaryOpType::ABS_INT32: return {"abs_tile_init();", fmt::format("abs_tile_int32({});", idst)};
        case UnaryOpType::NEG:
            if (input_dtype.has_value() && input_dtype.value() == DataType::INT32) {
                return {"negative_tile_init();", fmt::format("negative_tile_int32({});", idst)};
            }
            return {"negative_tile_init();", fmt::format("negative_tile({});", idst)};
        case UnaryOpType::ACOS: return {"acos_tile_init();", fmt::format("acos_tile({});", idst)};
        case UnaryOpType::ASIN: return {"asin_tile_init();", fmt::format("asin_tile({});", idst)};
        case UnaryOpType::ASINH: return {"asinh_tile_init();", fmt::format("asinh_tile({});", idst)};
        case UnaryOpType::ATAN: return {"atan_tile_init();", fmt::format("atan_tile({});", idst)};
        case UnaryOpType::ATANH: return {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)};
        case UnaryOpType::COS: return {"cos_tile_init();", fmt::format("cos_tile({});", idst)};
        case UnaryOpType::ACOSH: return {"acosh_tile_init();", fmt::format("acosh_tile({});", idst)};
        case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
        case UnaryOpType::SINH: return {"sinh_tile_init();", fmt::format("sinh_tile({});", idst)};
        case UnaryOpType::ERFINV: return {"erfinv_tile_init();", fmt::format("erfinv_tile({});", idst)};
        case UnaryOpType::EXP2: return {"exp2_tile_init();", fmt::format("exp2_tile({});", idst)};
        case UnaryOpType::EXPM1: return {"expm1_tile_init();", fmt::format("expm1_tile({});", idst)};
        case UnaryOpType::GEZ:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"gez_tile_init();", fmt::format("gez_tile_int32({});", idst)};
            }
            return {"gez_tile_init();", fmt::format("gez_tile({});", idst)};
        case UnaryOpType::GTZ:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"gtz_tile_init();", fmt::format("gtz_tile_int32({});", idst)};
            }
            return {"gtz_tile_init();", fmt::format("gtz_tile({});", idst)};
        case UnaryOpType::I0: return {"i0_tile_init();", fmt::format("i0_tile({});", idst)};
        case UnaryOpType::I1: return {"i1_tile_init();", fmt::format("i1_tile({});", idst)};
        case UnaryOpType::ISFINITE: return {"isfinite_tile_init();", fmt::format("isfinite_tile({});", idst)};
        case UnaryOpType::ISINF: return {"isinf_tile_init();", fmt::format("isinf_tile({});", idst)};
        case UnaryOpType::ISNAN: return {"isnan_tile_init();", fmt::format("isnan_tile({});", idst)};
        case UnaryOpType::ISNEGINF: return {"isneginf_tile_init();", fmt::format("isneginf_tile({});", idst)};
        case UnaryOpType::ISPOSINF: return {"isposinf_tile_init();", fmt::format("isposinf_tile({});", idst)};
        case UnaryOpType::LEZ:
            if (input_dtype.has_value() && *input_dtype == DataType::INT32) {
                return {"lez_tile_init();", fmt::format("lez_tile_int32({});", idst)};
            }
            return {"lez_tile_init();", fmt::format("lez_tile({});", idst)};
        case UnaryOpType::LOGICAL_NOT_UNARY: {
            TT_FATAL(input_dtype.has_value(), "LOGICAL_NOT_UNARY requires input_dtype");
            const char* df;
            switch (*input_dtype) {
                case DataType::INT32: df = "Int32"; break;
                case DataType::UINT32: df = "UInt32"; break;
                case DataType::UINT16: df = "UInt16"; break;
                case DataType::FLOAT32: df = "Float32"; break;
                case DataType::BFLOAT16: df = "Float16_b"; break;
                case DataType::BFLOAT8_B: df = "Bfp8_b"; break;
                case DataType::BFLOAT4_B: df = "Bfp4_b"; break;
                default: TT_THROW("Unsupported dtype for logical_not: {}", *input_dtype);
            }
            return {"logical_not_tile_init();", fmt::format("logical_not_tile<DataFormat::{}>({});", df, idst)};
        }
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
                return {"signbit_tile_int32_init();", fmt::format("signbit_tile_int32({});", idst)};
            }
            return {"signbit_tile_init();", fmt::format("signbit_tile({});", idst)};
        case UnaryOpType::SILU: return {"silu_tile_init();", fmt::format("silu_tile({});", idst)};
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
        case UnaryOpType::FRAC: return {"rounding_op_tile_init();", fmt::format("frac_tile({});", idst)};
        case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
        case UnaryOpType::HARDSWISH:
        case UnaryOpType::LGAMMA: return {};
        case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
        case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
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

std::string_view get_compute_kernel_path(UnaryOpType op_type, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::LGAMMA:
            TT_FATAL(input_dtype.has_value(), "UnaryNg lgamma: missing input dtype for kernel selection.");
            if (input_dtype.value() == DataType::BFLOAT16) {
                return "lgamma_fast_kernel.cpp";
            }
            return "lgamma_kernel.cpp";
        case UnaryOpType::MISH: return "mish_kernel.cpp";
        case UnaryOpType::TANHSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "tanhshrink_sfpu_kernel.cpp";
            }
            return "tanhshrink_kernel.cpp";
        case UnaryOpType::IDENTITY: return "eltwise_identity_kernel.cpp";
        case UnaryOpType::WHERE_TSS: return "where_tss_kernel.cpp";
        case UnaryOpType::LOGIT: return "logit_kernel.cpp";
        case UnaryOpType::HARDSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "hardshrink_kernel_sfpu.cpp";
            }
            return "hardshrink_kernel.cpp";
        case UnaryOpType::HARDSWISH: return "hardswish_kernel.cpp";
        case UnaryOpType::LOGSIGMOID: return "logsigmoid_kernel.cpp";
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
