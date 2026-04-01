// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include <fmt/core.h>
#include <tt_stl/assert.hpp>
#include "ttnn/tensor/types.hpp"
namespace ttnn::operations::unary_ng {

using tt::tt_metal::DataType;
using unary::EltwiseUnaryWithParam;
using unary::UnaryOpType;
using unary::VecMode;

namespace {

uint32_t as_uint(float f) { return std::bit_cast<uint32_t>(f); }

std::pair<std::string, std::string> make_simple(const char* name, const std::string& idst) {
    return {fmt::format("{}_tile_init();", name), fmt::format("{}_tile({});", name, idst)};
}

std::pair<std::string, std::string> make_with_int32(
    const char* name, const std::string& idst, std::optional<DataType> input_dtype) {
    TT_FATAL(input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
    if (*input_dtype == DataType::INT32) {
        return {fmt::format("{}_tile_init();", name), fmt::format("{}_tile_int32({});", name, idst)};
    }
    return make_simple(name, idst);
}

template <typename T>
std::pair<std::string, std::string> make_fast_approx(const char* name, T param0_raw, const std::string& idst) {
    uint32_t fast = static_cast<uint32_t>(param0_raw);
    return {fmt::format("{}_tile_init<{}u>();", name, fast), fmt::format("{0}_tile<{2}u>({1});", name, idst, fast)};
}

template <typename T>
std::pair<std::string, std::string> make_unary_comp(
    const char* name, T param0_raw, const std::string& idst, std::optional<DataType> input_dtype) {
    float param0 = static_cast<float>(param0_raw);
    TT_FATAL(input_dtype.has_value(), "{} requires input_dtype", name);
    if (*input_dtype == DataType::INT32 || *input_dtype == DataType::UINT32) {
        return {
            fmt::format("{}_tile_init();", name),
            fmt::format("{}_tile_int32({}, {}u);", name, idst, std::bit_cast<uint32_t>(param0_raw))};
    }
    return {fmt::format("{}_tile_init();", name), fmt::format("{}_tile({}, {:#x}u);", name, idst, as_uint(param0))};
}

// For ops whose parameter is an integer value (e.g. shift count, bitmask): value-cast to uint32.
template <typename T>
std::pair<std::string, std::string> make_uint_op(const char* name, T param0_raw, const std::string& idst) {
    return {
        fmt::format("{}_tile_init();", name),
        fmt::format("{}_tile({}, {}u);", name, idst, static_cast<uint32_t>(param0_raw))};
}

// For ops whose parameter is a float encoded as its IEEE-754 bit pattern: bit-cast to uint32.
template <typename T>
std::pair<std::string, std::string> make_float_param_op(const char* name, T param0_raw, const std::string& idst) {
    float param0 = static_cast<float>(param0_raw);
    return {fmt::format("{}_tile_init();", name), fmt::format("{}_tile({}, {:#x}u);", name, idst, as_uint(param0))};
}

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
        case UnaryOpType::EQZ:
        case UnaryOpType::GEZ:
        case UnaryOpType::GTZ:
        case UnaryOpType::LEZ:
        case UnaryOpType::LTZ:
        case UnaryOpType::NEZ:
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::UNARY_EQ:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::UNARY_GE:
        case UnaryOpType::UNARY_LE: return "SFPU_OP_UNARY_COMP_INCLUDE";
        case UnaryOpType::LOGICAL_NOT_UNARY: return "SFPU_OP_LOGICAL_NOT_INCLUDE";
        case UnaryOpType::RECIP: return "SFPU_OP_RECIP_INCLUDE";
        case UnaryOpType::RELU:
        case UnaryOpType::RELU6:
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::LEAKY_RELU: return "SFPU_OP_RELU_FAMILY_INCLUDE";
        case UnaryOpType::BITWISE_NOT: return "SFPU_OP_BITWISE_NOT_INCLUDE";
        case UnaryOpType::BITWISE_XOR: return "SFPU_OP_BITWISE_XOR_INCLUDE";
        case UnaryOpType::BITWISE_AND: return "SFPU_OP_BITWISE_AND_INCLUDE";
        case UnaryOpType::BITWISE_OR: return "SFPU_OP_BITWISE_OR_INCLUDE";
        case UnaryOpType::RIGHT_SHIFT: return "SFPU_OP_RIGHT_SHIFT_INCLUDE";
        case UnaryOpType::LEFT_SHIFT: return "SFPU_OP_LEFT_SHIFT_INCLUDE";
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::TRUNC:
        case UnaryOpType::FRAC:
        case UnaryOpType::ROUND: return "SFPU_OP_ROUND_FAMILY_INCLUDE";
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU: return "SFPU_OP_BINOP_WITH_SCALAR_INCLUDE";
        case UnaryOpType::RSUB: return "SFPU_OP_RSUB_INCLUDE";
        case UnaryOpType::HARDSIGMOID:
        case UnaryOpType::SOFTSIGN:
        case UnaryOpType::SOFTSHRINK:
        case UnaryOpType::HARDSHRINK:
        case UnaryOpType::CELU: return "SFPU_OP_ACTIVATIONS_INCLUDE";
        case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
        case UnaryOpType::POLYGAMMA: return "SFPU_OP_POLYGAMMA_INCLUDE";
        case UnaryOpType::LOGSIGMOID: return "SFPU_OP_LOGSIGMOID_INCLUDE";
        case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
        case UnaryOpType::EXP: return "SFPU_OP_EXP_INCLUDE";
        case UnaryOpType::GELU: return "SFPU_OP_GELU_INCLUDE";
        case UnaryOpType::SQRT: return "SFPU_OP_SQRT_INCLUDE";
        case UnaryOpType::RSQRT: return "SFPU_OP_RSQRT_INCLUDE";
        case UnaryOpType::ERFC:
        case UnaryOpType::ERF: return "SFPU_OP_ERF_ERFC_INCLUDE";
        case UnaryOpType::ELU: return "SFPU_OP_ELU_INCLUDE";
        case UnaryOpType::LOG:
        case UnaryOpType::LOG10:
        case UnaryOpType::LOG2: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
        case UnaryOpType::LOG1P: return "SFPU_OP_LOG1P_INCLUDE";
        case UnaryOpType::SOFTPLUS: return "SFPU_OP_SOFTPLUS_INCLUDE";
        case UnaryOpType::XIELU: return "SFPU_OP_XIELU_INCLUDE";
        case UnaryOpType::SELU: return "SFPU_OP_SELU_INCLUDE";
        case UnaryOpType::PRELU_SFPU: return "SFPU_OP_PRELU_INCLUDE";
        case UnaryOpType::TYPECAST: return "SFPU_OP_TYPECAST_INCLUDE";
        case UnaryOpType::REMAINDER: return "SFPU_OP_REMAINDER_INCLUDE";
        case UnaryOpType::FMOD: return "SFPU_OP_FMOD_INCLUDE";
        case UnaryOpType::FILL: return "SFPU_OP_FILL_INCLUDE";
        case UnaryOpType::WHERE_TSS: return "SFPU_OP_WHERE_INCLUDE";
        case UnaryOpType::CLAMP_TSS: return "SFPU_OP_CLAMP_INCLUDE";
        case UnaryOpType::THRESHOLD: return "SFPU_OP_THRESHOLD_INCLUDE";
        case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
        case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
        case UnaryOpType::HARDMISH: return "SFPU_OP_HARDMISH_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    }
}

template <typename T>
std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type, std::span<const T> params, const std::string& idst, std::optional<DataType> input_dtype) {
    // TODO don't cast T to float when precision needs to be preserved
    const T param0_raw = params[0];
    float param0 = static_cast<float>(params[0]);
    switch (op_type) {
        case UnaryOpType::FILL: {
            TT_FATAL(input_dtype.has_value(), "FILL requires input_dtype");
            if (*input_dtype == DataType::INT32 || *input_dtype == DataType::UINT32 ||
                *input_dtype == DataType::UINT16) {
                uint32_t fill_val;
                if (*input_dtype == DataType::UINT16) {
                    auto as_int = static_cast<int32_t>(param0_raw);
                    TT_FATAL(
                        as_int >= 0 && as_int <= std::numeric_limits<uint16_t>::max(),
                        "FILL value {} out of range for UInt16",
                        as_int);
                    fill_val = static_cast<uint32_t>(as_int);
                } else {
                    fill_val = static_cast<uint32_t>(param0_raw);
                }
                const char* df = (*input_dtype == DataType::INT32)    ? "Int32"
                                 : (*input_dtype == DataType::UINT32) ? "UInt32"
                                                                      : "UInt16";
                return {
                    "fill_tile_init();", fmt::format("fill_tile_int<DataFormat::{}>({}, {:#x}u);", df, idst, fill_val)};
            }
            return {"fill_tile_init();", fmt::format("fill_tile_bitcast({}, {:#x}u);", idst, as_uint(param0))};
        }
        case UnaryOpType::ROUND:
            return {"rounding_op_tile_init();", fmt::format("round_tile({}, {});", idst, static_cast<int>(params[0]))};
        case UnaryOpType::RELU_MAX: {
            TT_FATAL(input_dtype.has_value(), "RELU_MAX requires input_dtype");
            if (*input_dtype == DataType::INT32) {
                return {
                    "relu_max_tile_init();", fmt::format("relu_max_tile_int32({}, {:#x}u);", idst, as_uint(param0))};
            }
            return {"relu_max_tile_init();", fmt::format("relu_max_tile({}, {:#x}u);", idst, as_uint(param0))};
        }
        case UnaryOpType::RELU_MIN: {
            TT_FATAL(input_dtype.has_value(), "RELU_MIN requires input_dtype");
            if (*input_dtype == DataType::INT32) {
                return {
                    "relu_min_tile_init();",
                    fmt::format("relu_min_tile_int32({}, {}u);", idst, static_cast<uint32_t>(params[0]))};
            }
            return {"relu_min_tile_init();", fmt::format("relu_min_tile({}, {:#x}u);", idst, as_uint(param0))};
        }
        case UnaryOpType::POWER: return make_float_param_op("power", param0_raw, idst);
        case UnaryOpType::POWER_ITERATIVE: {
            TT_FATAL(
                param0 >= 0.0f && param0 == std::floor(param0),
                "POWER_ITERATIVE requires non-negative integer exponent, got {}",
                param0);
            return {"power_iterative_tile_init();", fmt::format("power_iterative_tile({}, {});", idst, param0_raw)};
        }
        case UnaryOpType::LEAKY_RELU: return make_float_param_op("leaky_relu", param0_raw, idst);
        case UnaryOpType::ELU: return make_float_param_op("elu", param0_raw, idst);
        case UnaryOpType::HEAVISIDE: return make_float_param_op("heaviside", param0_raw, idst);
        case UnaryOpType::RPOW: return make_float_param_op("rpow", param0_raw, idst);
        case UnaryOpType::CELU: {
            return {
                "celu_tile_init();",
                fmt::format("celu_tile({}, {:#x}u, {:#x}u);", idst, as_uint(param0), as_uint(1.0f / param0))};
        }
        case UnaryOpType::PRELU_SFPU: return make_float_param_op("prelu", param0_raw, idst);
        case UnaryOpType::SOFTSHRINK:
            return {"softshrink_tile_init();", fmt::format("softshrink_tile({}, {}u);", idst, as_uint(param0))};
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FMOD: {
            const char* name = op_type == UnaryOpType::REMAINDER ? "remainder" : "fmod";
            uint32_t u = as_uint(param0), inv = as_uint(1.0f / param0);
            return {
                fmt::format("{}_tile_init({:#x}u, {:#x}u);", name, u, inv),
                fmt::format("{}_tile({}, {:#x}u, {:#x}u);", name, idst, u, inv)};
        }
        case UnaryOpType::BITWISE_XOR: return make_uint_op("bitwise_xor", param0_raw, idst);
        case UnaryOpType::BITWISE_AND: return make_uint_op("bitwise_and", param0_raw, idst);
        case UnaryOpType::BITWISE_OR: return make_uint_op("bitwise_or", param0_raw, idst);
        case UnaryOpType::RIGHT_SHIFT: return make_uint_op("right_shift", param0_raw, idst);
        case UnaryOpType::LEFT_SHIFT: return make_uint_op("left_shift", param0_raw, idst);
        case UnaryOpType::EXP: return make_fast_approx("exp", param0_raw, idst);
        case UnaryOpType::GELU: return make_fast_approx("gelu", param0_raw, idst);
        case UnaryOpType::RSQRT:
            return {"rsqrt_tile_init<false>();", fmt::format("rsqrt_tile<false, {1}>({0});", idst, param0_raw)};
        case UnaryOpType::SQRT: return {"sqrt_tile_init();", fmt::format("sqrt_tile<{1}>({0});", idst, param0_raw)};
        case UnaryOpType::ERF: return make_fast_approx("erf", param0_raw, idst);
        case UnaryOpType::ERFC: return make_fast_approx("erfc", param0_raw, idst);
        case UnaryOpType::LOG: return make_fast_approx("log", param0_raw, idst);
        case UnaryOpType::LOG10:
        case UnaryOpType::LOG2: {
            uint32_t fast = static_cast<uint32_t>(param0);
            const char* base_hex = op_type == UnaryOpType::LOG10 ? "0x3ede5bd9u" : "0x3fb8aa3bu";
            return {
                fmt::format("log_with_base_tile_init<{}u>();", fast),
                fmt::format("log_with_base_tile<{1}u>({0}, {2});", idst, fast, base_hex)};
        }
        case UnaryOpType::LOG1P: return make_fast_approx("log1p", param0_raw, idst);
        case UnaryOpType::TANH: return make_fast_approx("tanh", param0_raw, idst);
        case UnaryOpType::HARDMISH: return make_fast_approx("hardmish", param0_raw, idst);
        case UnaryOpType::SIGMOID: {
            uint32_t param1 = static_cast<uint32_t>(params[1]);
            TT_FATAL(
                static_cast<int32_t>(param0) == static_cast<int32_t>(VecMode::C) ||
                    static_cast<int32_t>(param0) == static_cast<int32_t>(VecMode::RC),
                "Invalid Vector mode value. Expected vector mode C (2) or RC (4) for sigmoid");
            return {
                fmt::format("sigmoid_tile_init<{}u>();", param1),
                fmt::format("sigmoid_tile<{1}, {2}u>({0});", idst, static_cast<int32_t>(param0), param1)};
        }
        case UnaryOpType::RDIV: {
            uint32_t rounding_mode_value = static_cast<uint32_t>(params[1]);
            static constexpr const char* rounding_mode_strs[] = {
                "ckernel::RoundingMode::None", "ckernel::RoundingMode::Trunc", "ckernel::RoundingMode::Floor"};
            return {
                "rdiv_tile_init();",
                fmt::format(
                    "rdiv_tile<{}>({}, {:#x}u);", rounding_mode_strs[rounding_mode_value], idst, as_uint(param0))};
        }
        case UnaryOpType::RSUB: {
            TT_FATAL(input_dtype.has_value(), "RSUB requires input_dtype");
            if (*input_dtype == DataType::UINT16 || *input_dtype == DataType::UINT8) {
                TT_THROW("Unsupported data type");
            }
            if (*input_dtype == DataType::INT32 || *input_dtype == DataType::UINT32) {
                return {
                    "rsub_unary_int32_tile_init();",
                    fmt::format(
                        "rsub_unary_int32_tile({}, {}u);",
                        idst,
                        std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))};
            }
            return {"rsub_tile_init();", fmt::format("rsub_tile({}, {:#x}u);", idst, as_uint(param0))};
        }
        case UnaryOpType::ADD_UNARY_SFPU: {
            TT_FATAL(input_dtype.has_value(), "ADD_UNARY_SFPU requires input_dtype");
            if (*input_dtype == DataType::INT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format(
                        "add_unary_tile_int32({}, {}u);",
                        idst,
                        std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))};
            }
            if (*input_dtype == DataType::UINT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format("add_unary_tile_int32({}, {}u);", idst, static_cast<uint32_t>(param0_raw))};
            }
            return {
                "binop_with_scalar_tile_init();", fmt::format("add_unary_tile({}, {:#x}u);", idst, as_uint(param0))};
        }
        case UnaryOpType::SUB_UNARY_SFPU: {
            TT_FATAL(input_dtype.has_value(), "SUB_UNARY_SFPU requires input_dtype");
            if (input_dtype == DataType::INT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format(
                        "sub_unary_tile_int32({}, {}u);",
                        idst,
                        std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))};
            }
            if (input_dtype == DataType::UINT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format("sub_unary_tile_int32({}, {}u);", idst, static_cast<uint32_t>(param0_raw))};
            }
            return {
                "binop_with_scalar_tile_init();", fmt::format("sub_unary_tile({}, {:#x}u);", idst, as_uint(param0))};
        }
        case UnaryOpType::MUL_UNARY_SFPU:
            return {
                "binop_with_scalar_tile_init();", fmt::format("mul_unary_tile({}, {:#x}u);", idst, as_uint(param0))};
        case UnaryOpType::DIV_UNARY_SFPU:
            return {
                "binop_with_scalar_tile_init();",
                fmt::format("div_unary_tile({}, {:#x}u);", idst, as_uint(1.0f / param0))};
        case UnaryOpType::UNARY_NE: return make_unary_comp("unary_ne", param0_raw, idst, input_dtype);
        case UnaryOpType::UNARY_EQ: return make_unary_comp("unary_eq", param0_raw, idst, input_dtype);
        case UnaryOpType::UNARY_GT: return make_unary_comp("unary_gt", param0_raw, idst, input_dtype);
        case UnaryOpType::UNARY_LT: return make_unary_comp("unary_lt", param0_raw, idst, input_dtype);
        case UnaryOpType::UNARY_GE: return make_unary_comp("unary_ge", param0_raw, idst, input_dtype);
        case UnaryOpType::UNARY_LE: return make_unary_comp("unary_le", param0_raw, idst, input_dtype);
        case UnaryOpType::MAXIMUM:
        case UnaryOpType::MINIMUM: {
            const char* prefix = op_type == UnaryOpType::MAXIMUM ? "unary_max" : "unary_min";
            TT_FATAL(input_dtype.has_value(), "{} requires input_dtype", prefix);
            if (*input_dtype == DataType::INT32) {
                return {
                    fmt::format("{}_int32_tile_init();", prefix),
                    fmt::format("{}_int32_tile({}, {}u);", prefix, idst, static_cast<uint32_t>(params[0]))};
            }
            if (*input_dtype == DataType::UINT32) {
                return {
                    fmt::format("{}_uint32_tile_init();", prefix),
                    fmt::format("{}_uint32_tile({}, {}u);", prefix, idst, static_cast<uint32_t>(params[0]))};
            }
            return {
                fmt::format("{}_tile_init();", prefix),
                fmt::format("{}_tile({}, {:#x}u);", prefix, idst, as_uint(param0))};
        }
        case UnaryOpType::SOFTPLUS: {
            TT_ASSERT(params.size() == 2, "Expected softplus to take 2 parameters");
            float param1 = static_cast<float>(params[1]);
            return {
                "softplus_tile_init();",
                fmt::format(
                    "softplus_tile({}, {:#x}u, {:#x}u, {:#x}u);",
                    idst,
                    as_uint(param0),
                    as_uint(1.0f / param0),
                    as_uint(param1))};
        }
        case UnaryOpType::XIELU: {
            TT_ASSERT(params.size() == 2, "Expected xielu to take 2 parameters (alpha_p, alpha_n)");
            return {
                "xielu_tile_init();",
                fmt::format(
                    "xielu_tile({}, {:#x}u, {:#x}u);", idst, as_uint(param0), as_uint(static_cast<float>(params[1])))};
        }
        case UnaryOpType::HARDTANH:
        case UnaryOpType::THRESHOLD:
        case UnaryOpType::SELU: {
            const char* name = op_type == UnaryOpType::HARDTANH    ? "hardtanh"
                               : op_type == UnaryOpType::THRESHOLD ? "threshold"
                                                                   : "selu";
            if (op_type == UnaryOpType::SELU) {
                TT_FATAL(params.size() == 2, "Expected selu to take 2 parameters");
            }
            float param1 = static_cast<float>(params[1]);
            return {
                fmt::format("{}_tile_init();", name),
                fmt::format("{}_tile({}, {:#x}u, {:#x}u);", name, idst, as_uint(param0), as_uint(param1))};
        }
        case UnaryOpType::CLAMP_TSS: {
            float param1 = static_cast<float>(params[1]);
            TT_FATAL(input_dtype.has_value(), "CLAMP_TSS requires input_dtype");
            if (*input_dtype == DataType::INT32) {
                return {
                    "clamp_tile_init();",
                    fmt::format(
                        "clamp_tile_int32({}, {}, {});",
                        idst,
                        static_cast<uint32_t>(params[0]),
                        static_cast<uint32_t>(params[1]))};
            }
            return {
                "clamp_tile_init();", fmt::format("clamp_tile({}, {}, {});", idst, as_uint(param0), as_uint(param1))};
        }
        case UnaryOpType::WHERE_TSS: {
            TT_FATAL(input_dtype.has_value(), "WHERE_TSS requires input_dtype");
            const char* data_format = (*input_dtype == DataType::INT32)     ? "Int32"
                                      : (*input_dtype == DataType::UINT32)  ? "UInt32"
                                      : (*input_dtype == DataType::FLOAT32) ? "Float32"
                                                                            : "Float16_b";
            return {
                "where_tile_init();",
                fmt::format("where_tile<DataFormat::{0}>({1}, {2}, {3}, {1});", data_format, idst, 1, 2)};
        }
        case UnaryOpType::TYPECAST: {
            TT_ASSERT(params.size() == 2, "Expected eltwise_typecast to take 2 parameters");
            uint32_t src_fmt = static_cast<uint32_t>(
                tt::tt_metal::datatype_to_dataformat_converter(static_cast<DataType>(static_cast<int>(params[0]))));
            uint32_t dst_fmt = static_cast<uint32_t>(
                tt::tt_metal::datatype_to_dataformat_converter(static_cast<DataType>(static_cast<int>(params[1]))));
            return {
                fmt::format("typecast_tile_init<{0}u, {1}u>();", src_fmt, dst_fmt),
                fmt::format("typecast_tile<{1}u, {2}u>({0});", idst, src_fmt, dst_fmt)};
        }
        case UnaryOpType::POLYGAMMA: {
            TT_ASSERT(params.size() == 2, "Expected polygamma to take 2 parameters (n, scale)");
            float param1 = static_cast<float>(params[1]);
            return {
                "polygamma_tile_init();",
                fmt::format("polygamma_tile({}, {:#x}u, {:#x}u);", idst, as_uint(param0), as_uint(param1))};
        }
        case UnaryOpType::HARDSHRINK:
            return {"hardshrink_tile_init();", fmt::format("hardshrink_tile({}, {:#x}u);", idst, as_uint(param0))};
        case UnaryOpType::LOGIT:
        case UnaryOpType::BITCAST:
            // Bitcast uses identity kernel (copy_tile + pack_tile) - no LLK needed
            // Parameters are input_dtype and output_dtype, but we don't need them for the kernel
        case UnaryOpType::MISH: return {};
        default: TT_THROW("unexpected parameterized op type {}", op_type);
    };
}

std::pair<std::string, std::string> get_op_init_and_func_default(
    UnaryOpType op_type, const std::string& idst, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::ABS: return make_simple("abs", idst);
        case UnaryOpType::ABS_INT32: return {"abs_tile_init();", fmt::format("abs_tile_int32({});", idst)};
        case UnaryOpType::NEG: return make_with_int32("negative", idst, input_dtype);
        case UnaryOpType::ACOS: return make_simple("acos", idst);
        case UnaryOpType::ASIN: return make_simple("asin", idst);
        case UnaryOpType::ASINH: return make_simple("asinh", idst);
        case UnaryOpType::ATAN: return make_simple("atan", idst);
        case UnaryOpType::ATANH: return make_simple("atanh", idst);
        case UnaryOpType::COS: return make_simple("cos", idst);
        case UnaryOpType::ACOSH: return make_simple("acosh", idst);
        case UnaryOpType::COSH: return make_simple("cosh", idst);
        case UnaryOpType::SINH: return make_simple("sinh", idst);
        case UnaryOpType::ERFINV: return make_simple("erfinv", idst);
        case UnaryOpType::EXP2: return make_simple("exp2", idst);
        case UnaryOpType::EXPM1: return make_simple("expm1", idst);
        case UnaryOpType::EQZ: {
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (*input_dtype == DataType::INT32) {
                return {"eqz_tile_init();", fmt::format("eqz_tile_int32({});", idst)};
            }
            if (*input_dtype == DataType::UINT16) {
                return {"eqz_tile_init();", fmt::format("eqz_tile_uint16({});", idst)};
            }
            if (*input_dtype == DataType::UINT32) {
                return {"eqz_tile_init();", fmt::format("eqz_tile_uint32({});", idst)};
            }
            return make_simple("eqz", idst);
        }
        case UnaryOpType::GEZ: return make_with_int32("gez", idst, input_dtype);
        case UnaryOpType::GTZ: return make_with_int32("gtz", idst, input_dtype);
        case UnaryOpType::I0: return make_simple("i0", idst);
        case UnaryOpType::I1: return make_simple("i1", idst);
        case UnaryOpType::ISFINITE: return make_simple("isfinite", idst);
        case UnaryOpType::ISINF: return make_simple("isinf", idst);
        case UnaryOpType::ISNAN: return make_simple("isnan", idst);
        case UnaryOpType::ISNEGINF: return make_simple("isneginf", idst);
        case UnaryOpType::ISPOSINF: return make_simple("isposinf", idst);
        case UnaryOpType::LEZ: return make_with_int32("lez", idst, input_dtype);
        case UnaryOpType::LTZ: return make_with_int32("ltz", idst, input_dtype);
        case UnaryOpType::LOGICAL_NOT_UNARY: {
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
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
        case UnaryOpType::NEZ: {
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (*input_dtype == DataType::INT32) {
                return {"nez_tile_init();", fmt::format("nez_tile_int32({});", idst)};
            }
            if (*input_dtype == DataType::UINT16) {
                return {"nez_tile_init();", fmt::format("nez_tile_uint16({});", idst)};
            }
            if (*input_dtype == DataType::UINT32) {
                return {"nez_tile_init();", fmt::format("nez_tile_uint32({});", idst)};
            }
            return make_simple("nez", idst);
        }
        case UnaryOpType::RECIP: return {"recip_tile_init<false>();", fmt::format("recip_tile<false>({});", idst)};
        case UnaryOpType::RELU: return make_with_int32("relu", idst, input_dtype);
        case UnaryOpType::RELU6: return {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
        case UnaryOpType::SIGN: return make_simple("sign", idst);
        case UnaryOpType::SIGNBIT:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (*input_dtype == DataType::INT32) {
                return {"signbit_tile_int32_init();", fmt::format("signbit_tile_int32({});", idst)};
            }
            return make_simple("signbit", idst);
        case UnaryOpType::SILU: return make_simple("silu", idst);
        case UnaryOpType::SIN: return make_simple("sin", idst);
        case UnaryOpType::SQUARE: {
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
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
            return make_simple("square", idst);
        }
        case UnaryOpType::TAN: return make_simple("tan", idst);
        case UnaryOpType::TILED_PROD: return make_simple("tiled_prod", idst);
        case UnaryOpType::BITWISE_NOT: return make_simple("bitwise_not", idst);
        case UnaryOpType::ALT_COMPLEX_ROTATE90: return make_simple("alt_complex_rotate90", idst);
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::TRUNC:
        case UnaryOpType::FRAC: {
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            const char* name = op_type == UnaryOpType::FLOOR   ? "floor"
                               : op_type == UnaryOpType::CEIL  ? "ceil"
                               : op_type == UnaryOpType::TRUNC ? "trunc"
                                                               : "frac";
            return {"rounding_op_tile_init();", fmt::format("{}_tile({});", name, idst)};
        }
        case UnaryOpType::HARDSIGMOID: return make_simple("hardsigmoid", idst);
        case UnaryOpType::SOFTSIGN: return make_simple("softsign", idst);
        case UnaryOpType::CBRT: return make_simple("cbrt", idst);
        case UnaryOpType::EXP: return make_simple("exp", idst);
        case UnaryOpType::GELU: return make_simple("gelu", idst);
        case UnaryOpType::RSQRT: return make_simple("rsqrt", idst);
        case UnaryOpType::SQRT: return make_simple("sqrt", idst);
        case UnaryOpType::ERF: return make_simple("erf", idst);
        case UnaryOpType::ERFC: return make_simple("erfc", idst);
        case UnaryOpType::LOG: return make_simple("log", idst);
        case UnaryOpType::LOG10:
            return {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3ede5bd9u);", idst)};
        case UnaryOpType::LOG2:
            return {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3fb8aa3bu);", idst)};
        case UnaryOpType::LOG1P: return make_simple("log1p", idst);
        case UnaryOpType::TANH: return make_simple("tanh", idst);
        case UnaryOpType::SIGMOID: return make_simple("sigmoid", idst);
        case UnaryOpType::HARDMISH: return make_simple("hardmish", idst);
        case UnaryOpType::HARDSWISH:
        case UnaryOpType::LGAMMA:
        case UnaryOpType::TANHSHRINK:
        case UnaryOpType::LOGSIGMOID:
        case UnaryOpType::BITCAST:
            // Bitcast uses identity kernel (copy_tile + pack_tile) - no LLK needed
            // Parameters are input_dtype and output_dtype, but we don't need them for the kernel
        case UnaryOpType::IDENTITY: return {};
        default: TT_THROW("Undefined non-parametrized op type {}", op_type);
    }
}

template <typename T>
std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type, std::span<const T> params, const std::string& idst, std::optional<DataType> input_dtype) {
    return !params.empty() ? get_op_init_and_func_parameterized(op_type, params, idst, input_dtype)
                           : get_op_init_and_func_default(op_type, idst, input_dtype);
}

void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines) {
    defines[get_macro_definition(op_type)] = "1";
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

std::string_view get_compute_kernel_path(UnaryOpType op_type, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::LGAMMA:
            TT_FATAL(input_dtype.has_value(), "UnaryNg lgamma: missing input dtype for kernel selection.");
            if (input_dtype.value() == DataType::BFLOAT16) {
                return "lgamma_fast_kernel.cpp";
            }
            return "lgamma_kernel.cpp";
        case UnaryOpType::MISH: return "mish_kernel.cpp";
        case UnaryOpType::TANHSHRINK: return "tanhshrink_kernel.cpp";
        case UnaryOpType::IDENTITY: return "eltwise_identity_kernel.cpp";
        case UnaryOpType::WHERE_TSS: return "where_tss_kernel.cpp";
        case UnaryOpType::LOGIT: return "logit_kernel.cpp";
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

}  // namespace ttnn::operations::unary_ng
