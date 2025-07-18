// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_op_utils.hpp"

#include <tt-metalium/assert.hpp>
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary::utils {

namespace {

std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::EXP: return "SFPU_OP_EXP_INCLUDE";
        case UnaryOpType::GELU: return "SFPU_OP_GELU_INCLUDE";
        case UnaryOpType::RECIP: return "SFPU_OP_RECIP_INCLUDE";
        case UnaryOpType::SQRT: return "SFPU_OP_SQRT_INCLUDE";
        case UnaryOpType::ERFINV: return "SFPU_OP_ERFINV_INCLUDE";
        case UnaryOpType::ERFC:
        case UnaryOpType::ERF: return "SFPU_OP_ERF_ERFC_INCLUDE";
        case UnaryOpType::ELU: return "SFPU_OP_ELU_INCLUDE";
        case UnaryOpType::RELU:
        case UnaryOpType::RELU6:
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::LEAKY_RELU: return "SFPU_OP_RELU_FAMILY_INCLUDE";
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU: return "SFPU_OP_BINOP_WITH_SCALAR_INCLUDE";
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::TRUNC:
        case UnaryOpType::FRAC:
        case UnaryOpType::ROUND: return "SFPU_OP_ROUND_FAMILY_INCLUDE";
        case UnaryOpType::RDIV:
        case UnaryOpType::RSUB: return "SFPU_OP_REVERSE_FAMILY_INCLUDE";
        case UnaryOpType::ISINF:
        case UnaryOpType::ISNAN:
        case UnaryOpType::ISNEGINF:
        case UnaryOpType::ISPOSINF:
        case UnaryOpType::ISFINITE: return "SFPU_OP_ISINF_ISNAN_INCLUDE";
        case UnaryOpType::LOGICAL_NOT_UNARY: return "SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE";
        case UnaryOpType::I0: return "SFPU_OP_I0_INCLUDE";
        case UnaryOpType::I1: return "SFPU_OP_I1_INCLUDE";
        case UnaryOpType::ACOSH:
        case UnaryOpType::COS:
        case UnaryOpType::SIN:
        case UnaryOpType::ASINH:
        case UnaryOpType::TAN:
        case UnaryOpType::ATANH: return "SFPU_OP_TRIG_FAMILY_INCLUDE";
        case UnaryOpType::NEG: return "SFPU_OP_NEG_INCLUDE";
        case UnaryOpType::SOFTPLUS: return "SFPU_OP_SOFTPLUS_INCLUDE";
        case UnaryOpType::PRELU_SFPU: return "SFPU_OP_PRELU_INCLUDE";
        case UnaryOpType::TYPECAST: return "SFPU_OP_TYPECAST_INCLUDE";
        case UnaryOpType::BITWISE_XOR: return "SFPU_OP_BITWISE_XOR_INCLUDE";
        case UnaryOpType::BITWISE_NOT: return "SFPU_OP_BITWISE_NOT_INCLUDE";
        case UnaryOpType::BITWISE_AND: return "SFPU_OP_BITWISE_AND_INCLUDE";
        case UnaryOpType::BITWISE_OR: return "SFPU_OP_BITWISE_OR_INCLUDE";
        case UnaryOpType::RIGHT_SHIFT: return "SFPU_OP_RIGHT_SHIFT_INCLUDE";
        case UnaryOpType::LEFT_SHIFT: return "SFPU_OP_LEFT_SHIFT_INCLUDE";
        case UnaryOpType::REMAINDER: return "SFPU_OP_REMAINDER_INCLUDE";
        case UnaryOpType::FMOD: return "SFPU_OP_FMOD_INCLUDE";
        case UnaryOpType::FILL: return "SFPU_OP_FILL_INCLUDE";
        case UnaryOpType::LOG1P: return "SFPU_OP_LOG1P_INCLUDE";
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::UNARY_EQ:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::UNARY_GE:
        case UnaryOpType::UNARY_LE:
        case UnaryOpType::GTZ:
        case UnaryOpType::LTZ:
        case UnaryOpType::EQZ:
        case UnaryOpType::LEZ:
        case UnaryOpType::GEZ:
        case UnaryOpType::NEZ: return "SFPU_OP_UNARY_COMP_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}

std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type,
    const std::vector<float>& params,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    std::pair<std::string, std::string> op_init_and_name;
    TT_FATAL(is_parametrized_type(op_type), "operator should support at least one parameter", "Error");
    float param0 = params[0];
    switch (op_type) {
        case UnaryOpType::FILL:
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"fill_tile_init();", fmt::format("fill_tile_int({}, {}u);", idst, (uint)param0)};
            } else {
                // Note: bit casted to int float is used to properly pass nan/+-inf
                op_init_and_name = {
                    "fill_tile_init();",
                    fmt::format("fill_tile_bitcast({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::ROUND:
            op_init_and_name = {"rounding_op_tile_init();", fmt::format("round_tile({}, {});", idst, (int)param0)};
            break;
        case UnaryOpType::RELU_MAX:
            op_init_and_name = {
                "relu_max_tile_init();",
                fmt::format("relu_max_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::RELU_MIN:
            op_init_and_name = {
                "relu_min_tile_init();",
                fmt::format("relu_min_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::POWER:
            op_init_and_name = {"power_tile_init();", fmt::format("power_tile({}, {}u);", idst, (uint32_t)param0)};
            break;
        case UnaryOpType::LEAKY_RELU:
            op_init_and_name = {
                "leaky_relu_tile_init();",
                fmt::format("leaky_relu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::ELU:
            op_init_and_name = {
                "elu_tile_init();", fmt::format("elu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::GELU:
            op_init_and_name = {
                fmt::format("gelu_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("gelu_tile<{1}u>({0});", idst, (uint32_t)param0)};
            break;
        case UnaryOpType::RSQRT:
            op_init_and_name = {
                fmt::format("rsqrt_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("rsqrt_tile<{1}u>({0});", idst, (uint32_t)param0)};
            break;
        case UnaryOpType::HEAVISIDE:
            op_init_and_name = {
                "heaviside_tile_init();",
                fmt::format("heaviside_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::BITWISE_XOR:
            op_init_and_name = {
                "bitwise_xor_tile_init();", fmt::format("bitwise_xor_tile({}, {}u);", idst, (uint)param0)};
            break;
        case UnaryOpType::BITWISE_AND:
            op_init_and_name = {
                "bitwise_and_tile_init();", fmt::format("bitwise_and_tile({}, {}u);", idst, (uint)param0)};
            break;
        case UnaryOpType::BITWISE_OR:
            op_init_and_name = {
                "bitwise_or_tile_init();", fmt::format("bitwise_or_tile({}, {}u);", idst, (uint)param0)};
            break;
        case UnaryOpType::RIGHT_SHIFT:
            op_init_and_name = {
                "right_shift_tile_init();", fmt::format("right_shift_tile({}, {}u);", idst, (uint)param0)};
            break;
        case UnaryOpType::LEFT_SHIFT:
            op_init_and_name = {
                "left_shift_tile_init();", fmt::format("left_shift_tile({}, {}u);", idst, (uint)param0)};
            break;
        case UnaryOpType::REMAINDER:
            op_init_and_name = {
                fmt::format(
                    "remainder_tile_init({:#x}u, {:#x}u);",
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0)),
                fmt::format(
                    "remainder_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0))};
            break;
        case UnaryOpType::FMOD:
            op_init_and_name = {
                fmt::format(
                    "fmod_tile_init({:#x}u, {:#x}u);",
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0)),
                fmt::format(
                    "fmod_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0))};
            break;
        case UnaryOpType::EXP:
            op_init_and_name = {
                fmt::format("exp_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("exp_tile<{1}u>({0});", idst, (uint32_t)param0)};
            break;
        case UnaryOpType::SIGMOID: {
            uint32_t param1 = (uint32_t)params[1];
            TT_FATAL(
                (int32_t)param0 == (int32_t)VecMode::C || (int32_t)param0 == (int32_t)VecMode::RC,
                "Invalid Vector mode value. Expected vector mode C (2) or RC (4) for sigmoid");
            op_init_and_name = {
                fmt::format("sigmoid_tile_init<{}u>();", param1),
                fmt::format("sigmoid_tile<{1}, {2}u>({0});", idst, (int32_t)param0, param1)};
            break;
        }
        case UnaryOpType::ERF:
            op_init_and_name = {
                fmt::format("erf_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("erf_tile<{1}u>({0});", idst, (uint32_t)param0)};
            break;
        case UnaryOpType::ERFC:
            op_init_and_name = {
                fmt::format("erfc_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("erfc_tile<{1}u>({0});", idst, (uint32_t)param0)};
            break;
        case UnaryOpType::RDIV: op_init_and_name = {}; break;
        case UnaryOpType::RSUB:
            op_init_and_name = {
                "rsub_tile_init();", fmt::format("rsub_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::SUB_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("sub_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::ADD_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("add_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::MUL_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("mul_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        case UnaryOpType::DIV_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("div_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(1.0f / param0))};
            break;
        case UnaryOpType::UNARY_NE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                op_init_and_name = {
                    "unary_ne_tile_init();", fmt::format("unary_ne_tile_int32({}, {}u);", idst, (uint)param0)};
            } else {
                op_init_and_name = {
                    "unary_ne_tile_init();",
                    fmt::format("unary_ne_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::UNARY_EQ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                op_init_and_name = {
                    "unary_eq_tile_init();", fmt::format("unary_eq_tile_int32({}, {}u);", idst, (uint)param0)};
            } else {
                op_init_and_name = {
                    "unary_eq_tile_init();",
                    fmt::format("unary_eq_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::UNARY_GT:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                op_init_and_name = {"unary_gt_tile_init();", fmt::format("unary_gt_tile_int32({}, {});", idst, param0)};
            } else {
                op_init_and_name = {
                    "unary_gt_tile_init();",
                    fmt::format("unary_gt_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::UNARY_LT:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                op_init_and_name = {"unary_lt_tile_init();", fmt::format("unary_lt_tile_int32({}, {});", idst, param0)};
            } else {
                op_init_and_name = {
                    "unary_lt_tile_init();",
                    fmt::format("unary_lt_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::UNARY_GE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                op_init_and_name = {"unary_ge_tile_init();", fmt::format("unary_ge_tile_int32({}, {});", idst, param0)};
            } else {
                op_init_and_name = {
                    "unary_ge_tile_init();",
                    fmt::format("unary_ge_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::UNARY_LE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                op_init_and_name = {"unary_le_tile_init();", fmt::format("unary_le_tile_int32({}, {});", idst, param0)};
            } else {
                op_init_and_name = {
                    "unary_le_tile_init();",
                    fmt::format("unary_le_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::SOFTPLUS: {
            TT_ASSERT(params.size() == 2, "Expected softplus to take 2 parameters");
            float param1 = params[1];
            op_init_and_name = {
                "softplus_tile_init();",
                fmt::format(
                    "softplus_tile({}, {:#x}u, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0),  // Pass reciprocal to avoid doing it on device
                    std::bit_cast<uint32_t>(param1))};
            break;
        }
        case UnaryOpType::PRELU_SFPU: {
            op_init_and_name = {
                "prelu_tile_init();", fmt::format("prelu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            break;
        }
        case UnaryOpType::TYPECAST:
            TT_ASSERT(params.size() == 2, "Expected eltwise_typecast to take 2 parameters");
            op_init_and_name = {
                "typecast_tile_init();",
                fmt::format(
                    "typecast_tile<{1}u, {2}u>({0});",
                    idst,
                    (uint32_t)datatype_to_dataformat_converter((DataType)params[0]),
                    (uint32_t)datatype_to_dataformat_converter((DataType)params[1]))};
            break;
        case UnaryOpType::MAXIMUM:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {
                    "unary_max_tile_init();", fmt::format("unary_max_int32_tile({}, {}u);", idst, (uint)param0)};
            } else {
                op_init_and_name = {
                    "unary_max_tile_init();",
                    fmt::format("unary_max_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::MINIMUM:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {
                    "unary_min_tile_init();", fmt::format("unary_min_int32_tile({}, {}u);", idst, (uint)param0)};
            } else {
                op_init_and_name = {
                    "unary_min_tile_init();",
                    fmt::format("unary_min_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            break;
        case UnaryOpType::HARDSHRINK: op_init_and_name = {}; break;
        case UnaryOpType::WHERE_TSS: op_init_and_name = {}; break;

        default: TT_THROW("unexpected parameterized op type {}", op_type);
    };
    return op_init_and_name;
}

std::pair<std::string, std::string> get_op_init_and_func_default(
    UnaryOpType op_type, std::string idst, std::optional<DataType> input_dtype) {
    std::pair<std::string, std::string> op_init_and_name;
    switch (op_type) {
        case UnaryOpType::BITWISE_NOT:
            op_init_and_name = {"bitwise_not_tile_init();", fmt::format("bitwise_not_tile({});", idst)};
            break;
        case UnaryOpType::RECIP: op_init_and_name = {"recip_tile_init();", fmt::format("recip_tile({});", idst)}; break;
        case UnaryOpType::GELU: op_init_and_name = {"gelu_tile_init();", fmt::format("gelu_tile({});", idst)}; break;
        case UnaryOpType::RSQRT: op_init_and_name = {"rsqrt_tile_init();", fmt::format("rsqrt_tile({});", idst)}; break;
        case UnaryOpType::RELU: op_init_and_name = {"relu_tile_init();", fmt::format("relu_tile({});", idst)}; break;
        case UnaryOpType::SQRT: op_init_and_name = {"sqrt_tile_init();", fmt::format("sqrt_tile({});", idst)}; break;
        case UnaryOpType::LOG: op_init_and_name = {"log_tile_init();", fmt::format("log_tile({});", idst)}; break;
        case UnaryOpType::LOG1P: op_init_and_name = {"log1p_tile_init();", fmt::format("log1p_tile({});", idst)}; break;
        case UnaryOpType::TANH: op_init_and_name = {"tanh_tile_init();", fmt::format("tanh_tile({});", idst)}; break;
        case UnaryOpType::SIGNBIT:
            op_init_and_name = {"signbit_tile_init();", fmt::format("signbit_tile({});", idst)};
            break;
        case UnaryOpType::SIN: op_init_and_name = {"sin_tile_init();", fmt::format("sin_tile({});", idst)}; break;
        case UnaryOpType::COS: op_init_and_name = {"cos_tile_init();", fmt::format("cos_tile({});", idst)}; break;
        case UnaryOpType::ISFINITE:
            op_init_and_name = {"isfinite_tile_init();", fmt::format("isfinite_tile({});", idst)};
            break;
        case UnaryOpType::ISINF: op_init_and_name = {"isinf_tile_init();", fmt::format("isinf_tile({});", idst)}; break;
        case UnaryOpType::ISPOSINF:
            op_init_and_name = {"isposinf_tile_init();", fmt::format("isposinf_tile({});", idst)};
            break;
        case UnaryOpType::ISNEGINF:
            op_init_and_name = {"isneginf_tile_init();", fmt::format("isneginf_tile({});", idst)};
            break;
        case UnaryOpType::ISNAN: op_init_and_name = {"isnan_tile_init();", fmt::format("isnan_tile({});", idst)}; break;
        case UnaryOpType::LOGICAL_NOT_UNARY:
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {
                    "logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile({});", idst)};
            }
            break;
        case UnaryOpType::I0: op_init_and_name = {"i0_tile_init();", fmt::format("i0_tile({});", idst)}; break;
        case UnaryOpType::I1: op_init_and_name = {"i1_tile_init();", fmt::format("i1_tile({});", idst)}; break;
        case UnaryOpType::EXP: op_init_and_name = {"exp_tile_init();", fmt::format("exp_tile({});", idst)}; break;
        case UnaryOpType::SIGMOID:
            op_init_and_name = {"sigmoid_tile_init();", fmt::format("sigmoid_tile({});", idst)};
            break;
        case UnaryOpType::ERF: op_init_and_name = {"erf_tile_init();", fmt::format("erf_tile({0});", idst)}; break;
        case UnaryOpType::ERFC: op_init_and_name = {"erfc_tile_init();", fmt::format("erfc_tile({});", idst)}; break;
        case UnaryOpType::ERFINV:
            op_init_and_name = {"erfinv_tile_init();", fmt::format("erfinv_tile({});", idst)};
            break;
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            op_init_and_name = {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x36f3u);", idst)};
            break;
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            op_init_and_name = {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3dc5u);", idst)};
            break;
        case UnaryOpType::ABS: op_init_and_name = {"abs_tile_init();", fmt::format("abs_tile({});", idst)}; break;
        case UnaryOpType::ABS_INT32:
            op_init_and_name = {"abs_tile_init();", fmt::format("abs_tile_int32({});", idst)};
            break;
        case UnaryOpType::SIGN: op_init_and_name = {"sign_tile_init();", fmt::format("sign_tile({});", idst)}; break;
        case UnaryOpType::SQUARE:
            op_init_and_name = {"square_tile_init();", fmt::format("square_tile({});", idst)};
            break;
        case UnaryOpType::TILED_PROD:
            op_init_and_name = {"tiled_prod_tile_init();", fmt::format("tiled_prod_tile({});", idst)};
            break;
        case UnaryOpType::EQZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype.value() == DataType::INT32) {
                op_init_and_name = {"eqz_tile_init();", fmt::format("eqz_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"eqz_tile_init();", fmt::format("eqz_tile({});", idst)};
            }
            break;
        case UnaryOpType::NEZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"nez_tile_init();", fmt::format("nez_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"nez_tile_init();", fmt::format("nez_tile({});", idst)};
            }
            break;
        case UnaryOpType::LTZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"ltz_tile_init();", fmt::format("ltz_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"ltz_tile_init();", fmt::format("ltz_tile({});", idst)};
            }
            break;
        case UnaryOpType::GTZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"gtz_tile_init();", fmt::format("gtz_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"gtz_tile_init();", fmt::format("gtz_tile({});", idst)};
            }
            break;
        case UnaryOpType::GEZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"gez_tile_init();", fmt::format("gez_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"gez_tile_init();", fmt::format("gez_tile({});", idst)};
            }
            break;
        case UnaryOpType::LEZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"lez_tile_init();", fmt::format("lez_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"lez_tile_init();", fmt::format("lez_tile({});", idst)};
            }
            break;
        case UnaryOpType::EXP2: op_init_and_name = {"exp2_tile_init();", fmt::format("exp2_tile({});", idst)}; break;
        case UnaryOpType::EXPM1: op_init_and_name = {"expm1_tile_init();", fmt::format("expm1_tile({});", idst)}; break;
        case UnaryOpType::ASIN: op_init_and_name = {"asin_tile_init();", fmt::format("asin_tile({});", idst)}; break;
        case UnaryOpType::ASINH: op_init_and_name = {"asinh_tile_init();", fmt::format("asinh_tile({});", idst)}; break;
        case UnaryOpType::ACOS: op_init_and_name = {"acos_tile_init();", fmt::format("acos_tile({});", idst)}; break;
        case UnaryOpType::ACOSH: op_init_and_name = {"acosh_tile_init();", fmt::format("acosh_tile({});", idst)}; break;
        case UnaryOpType::ATAN: op_init_and_name = {"atan_tile_init();", fmt::format("atan_tile({});", idst)}; break;
        case UnaryOpType::ATANH: op_init_and_name = {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)}; break;
        case UnaryOpType::TAN: op_init_and_name = {"tan_tile_init();", fmt::format("tan_tile({});", idst)}; break;
        case UnaryOpType::SILU: op_init_and_name = {"silu_tile_init();", fmt::format("silu_tile({});", idst)}; break;
        case UnaryOpType::FLOOR:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::FLOAT32) {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("floor_tile_float32({});", idst)};
            } else {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("floor_tile({});", idst)};
            }
            break;
        case UnaryOpType::CEIL:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::FLOAT32) {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("ceil_tile_float32({});", idst)};
            } else {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("ceil_tile({});", idst)};
            }
            break;
        case UnaryOpType::TRUNC:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::FLOAT32) {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("trunc_tile_float32({});", idst)};
            } else {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("trunc_tile({});", idst)};
            }
            break;
        case UnaryOpType::FRAC:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::FLOAT32) {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("frac_tile_float32({});", idst)};
            } else {
                op_init_and_name = {"rounding_op_tile_init();", fmt::format("frac_tile({});", idst)};
            }
            break;
        case UnaryOpType::RELU6:
            op_init_and_name = {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
            break;
        case UnaryOpType::NEG:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                op_init_and_name = {"negative_tile_init();", fmt::format("negative_tile_int32({});", idst)};
            } else {
                op_init_and_name = {"negative_tile_init();", fmt::format("negative_tile({});", idst)};
            }
            break;
        case UnaryOpType::ALT_COMPLEX_ROTATE90:
            op_init_and_name = {"alt_complex_rotate90_tile_init();", fmt::format("alt_complex_rotate90_tile({});", idst)};
            break;
        case UnaryOpType::MISH: op_init_and_name = {}; break;
        case UnaryOpType::IDENTITY: op_init_and_name = {}; break;
        case UnaryOpType::TANHSHRINK: op_init_and_name = {}; break;
        default: TT_THROW("Undefined non-parametrized op type {}", op_type);
    }
    return op_init_and_name;
}

std::map<std::string, std::string> get_defines_impl(
    UnaryOpType op_type,
    const std::vector<float>& params,
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
    if (name == "relu") {
        return UnaryWithParam(UnaryOpType::RELU);
    } else if (name == "relu6") {
        return UnaryWithParam(UnaryOpType::RELU6);
    } else if (name == "gelu") {
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(true));
    } else if (name == "silu") {
        return UnaryWithParam(UnaryOpType::SILU);
    } else if (name == "sigmoid") {
        return UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(VecMode::RC), static_cast<float>(false)});
    } else if (name == "sigmoid_approx") {
        return UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(VecMode::RC), static_cast<float>(true)});
    } else if (name == "sqrt") {
        return UnaryWithParam(UnaryOpType::SQRT);
    } else if (name == "exp") {
        return UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true));
    } else if (name == "recip") {
        return UnaryWithParam(UnaryOpType::RECIP);
    } else if (name == "log") {
        return UnaryWithParam(UnaryOpType::LOG);
    } else if (name == "log1p") {
        return UnaryWithParam(UnaryOpType::LOG1P);
    } else if (name == "tanh") {
        return UnaryWithParam(UnaryOpType::TANH);
    } else if (name == "log2") {
        return UnaryWithParam(UnaryOpType::LOG2);
    } else if (name == "log10") {
        return UnaryWithParam(UnaryOpType::LOG10);
    } else if (name == "sin") {
        return UnaryWithParam(UnaryOpType::SIN);
    } else if (name == "cos") {
        return UnaryWithParam(UnaryOpType::COS);
    } else if (name == "abs") {
        return UnaryWithParam(UnaryOpType::ABS);
    } else if (name == "abs_int32") {
        return UnaryWithParam(UnaryOpType::ABS_INT32);
    } else if (name == "sign") {
        return UnaryWithParam(UnaryOpType::SIGN);
    } else if (name == "square") {
        return UnaryWithParam(UnaryOpType::SQUARE);
    } else if (name == "softplus") {
        return UnaryWithParam(UnaryOpType::SOFTPLUS);
    } else if (name == "alt_complex_rotate90") {
        return UnaryWithParam(UnaryOpType::ALT_COMPLEX_ROTATE90);
    }
    TT_THROW("Unknown unary op: {}", name);
}

std::map<std::string, std::string> get_defines(
    UnaryOpType op_type,
    const std::optional<std::vector<float>>& params,
    const std::string& id,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    std::string init_def = fmt::format("SFPU_OP_INIT_{}", id);
    std::string func_def = fmt::format("SFPU_OP_FUNC_{}", id);
    return get_defines_impl(op_type, params.value_or(std::vector<float>{}), idst, init_def, func_def, input_dtype);
}

std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type,
    const std::vector<float>& params,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    return params.size() > 0 ? get_op_init_and_func_parameterized(op_type, params, idst, input_dtype)
                             : get_op_init_and_func_default(op_type, idst, input_dtype);
}

std::map<std::string, std::string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain,
    const std::string& block_id,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    std::vector<std::pair<std::string, std::string>> op_init_and_name;
    std::map<std::string, std::string> block_defines;
    std::string block_define = "";
    for (uint32_t i = 0; i < op_chain.size(); i++) {
        std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
        std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
        block_define += init_def + " " + func_def + " ";
        block_defines.merge(
            get_defines_impl(op_chain[i].op_type, op_chain[i].params, idst, init_def, func_def, input_dtype));
    }
    block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
    return block_defines;
}

// update split eltwise ops include macros
void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines) {
    defines[get_macro_definition(op_type)] = "1";
}

std::string get_compute_kernel_path(
    UnaryOpType op_type, const std::string& compute_root, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::MISH: return fmt::format("{}/{}", compute_root, "mish_kernel.cpp");
        case UnaryOpType::TANHSHRINK: return fmt::format("{}/{}", compute_root, "tanhshrink_kernel.cpp");
        case UnaryOpType::IDENTITY: return fmt::format("{}/{}", compute_root, "eltwise_identity_kernel.cpp");
        case UnaryOpType::WHERE_TSS: return fmt::format("{}/{}", compute_root, "where_tss_kernel.cpp");
        case UnaryOpType::HARDSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return fmt::format("{}/{}", compute_root, "hardshrink_kernel_sfpu.cpp");
            } else {
                return fmt::format("{}/{}", compute_root, "hardshrink_kernel.cpp");
            }
        default: return fmt::format("{}/{}", compute_root, "eltwise_sfpu.cpp");
    }
}

}  // namespace ttnn::operations::unary::utils
