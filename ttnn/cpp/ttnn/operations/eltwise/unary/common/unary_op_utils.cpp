// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_op_utils.hpp"

#include <optional>
#include <tt_stl/assert.hpp>
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
        case UnaryOpType::RSQRT: return "SFPU_OP_RSQRT_INCLUDE";
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
        case UnaryOpType::RSUB: return "SFPU_OP_RSUB_INCLUDE";
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
        case UnaryOpType::COSH:
        case UnaryOpType::SINH:
        case UnaryOpType::SIN:
        case UnaryOpType::ASINH:
        case UnaryOpType::TAN:
        case UnaryOpType::ATANH: return "SFPU_OP_TRIG_FAMILY_INCLUDE";
        case UnaryOpType::NEG: return "SFPU_OP_NEG_INCLUDE";
        case UnaryOpType::SOFTPLUS: return "SFPU_OP_SOFTPLUS_INCLUDE";
        case UnaryOpType::LOGSIGMOID: return "SFPU_OP_LOGSIGMOID_INCLUDE";
        case UnaryOpType::SELU: return "SFPU_OP_SELU_INCLUDE";
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
        case UnaryOpType::WHERE_TSS: return "SFPU_OP_WHERE_INCLUDE";
        case UnaryOpType::CLAMP_TSS: return "SFPU_OP_CLAMP_INCLUDE";
        case UnaryOpType::SOFTSHRINK:
        case UnaryOpType::SOFTSIGN:
        case UnaryOpType::HARDSIGMOID:
        case UnaryOpType::CELU: return "SFPU_OP_ACTIVATIONS_INCLUDE";
        case UnaryOpType::THRESHOLD: return "SFPU_OP_THRESHOLD_INCLUDE";
        case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
        case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
        case UnaryOpType::HARDMISH: return "SFPU_OP_HARDMISH_INCLUDE";
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}

template <typename T>
std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type, std::span<const T> params, const std::string& idst, std::optional<DataType> input_dtype) {
    TT_FATAL(
        is_parametrized_type(op_type),
        "operator should support at least one parameter but op_type {} does not",
        op_type);
    // TODO don't cast T to float when precision needs to be preserved
    const T param0_raw = params[0];
    float param0 = static_cast<float>(params[0]);
    switch (op_type) {
        case UnaryOpType::FILL:
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (input_dtype == DataType::INT32) {
                return {"fill_tile_init();", fmt::format("fill_tile_int({}, {}u);", idst, (uint)params[0])};
            } else if (input_dtype == DataType::UINT32) {
                // TODO: Use uint32_t tile API here once implemented
                return {"fill_tile_init();", fmt::format("fill_tile_int({}, {}u);", idst, (uint)params[0])};
            } else {
                // Note: bit casted to int float is used to properly pass nan/+-inf
                return {
                    "fill_tile_init();",
                    fmt::format("fill_tile_bitcast({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
        case UnaryOpType::ROUND:
            return {"rounding_op_tile_init();", fmt::format("round_tile({}, {});", idst, (int)params[0])};
        case UnaryOpType::RELU_MAX:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {
                    "relu_max_tile_init();",
                    fmt::format("relu_max_tile_int32({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
            return {
                "relu_max_tile_init();",
                fmt::format("relu_max_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::RELU_MIN:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"relu_min_tile_init();", fmt::format("relu_min_tile_int32({}, {}u);", idst, (uint)params[0])};
            }
            return {
                "relu_min_tile_init();",
                fmt::format("relu_min_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::POWER:
            return {"power_tile_init();", fmt::format("power_tile({}, {}u);", idst, (uint32_t)param0)};
        case UnaryOpType::LEAKY_RELU:
            return {
                "leaky_relu_tile_init();",
                fmt::format("leaky_relu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::ELU:
            return {"elu_tile_init();", fmt::format("elu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::GELU:
            return {
                fmt::format("gelu_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("gelu_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::LOG:
            return {
                fmt::format("log_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("log_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            return {
                fmt::format("log_with_base_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("log_with_base_tile<{1}u>({0}, 0x3ede5bd9u);", idst, (uint32_t)param0)};

        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            return {
                fmt::format("log_with_base_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("log_with_base_tile<{1}u>({0}, 0x3fb8aa3bu);", idst, (uint32_t)param0)};
        case UnaryOpType::LOG1P:
            return {
                fmt::format("log1p_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("log1p_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::TANH:
            return {
                fmt::format("tanh_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("tanh_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::HEAVISIDE:
            return {
                "heaviside_tile_init();",
                fmt::format("heaviside_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::BITWISE_XOR:
            return {"bitwise_xor_tile_init();", fmt::format("bitwise_xor_tile({}, {}u);", idst, (uint)params[0])};
        case UnaryOpType::BITWISE_AND:
            return {"bitwise_and_tile_init();", fmt::format("bitwise_and_tile({}, {}u);", idst, (uint)params[0])};
        case UnaryOpType::BITWISE_OR:
            return {"bitwise_or_tile_init();", fmt::format("bitwise_or_tile({}, {}u);", idst, (uint)params[0])};
        case UnaryOpType::RIGHT_SHIFT:
            return {"right_shift_tile_init();", fmt::format("right_shift_tile({}, {}u);", idst, (uint)params[0])};
        case UnaryOpType::LEFT_SHIFT:
            return {"left_shift_tile_init();", fmt::format("left_shift_tile({}, {}u);", idst, (uint)params[0])};
        case UnaryOpType::REMAINDER:
            return {
                fmt::format(
                    "remainder_tile_init({:#x}u, {:#x}u);",
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0)),
                fmt::format(
                    "remainder_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0))};
        case UnaryOpType::FMOD:
            return {
                fmt::format(
                    "fmod_tile_init({:#x}u, {:#x}u);",
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0)),
                fmt::format(
                    "fmod_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0))};
        case UnaryOpType::EXP:
            return {
                fmt::format("exp_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("exp_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::SIGMOID: {
            uint32_t param1 = (uint32_t)params[1];
            TT_FATAL(
                (int32_t)param0 == (int32_t)VecMode::C || (int32_t)param0 == (int32_t)VecMode::RC,
                "Invalid Vector mode value. Expected vector mode C (2) or RC (4) for sigmoid");
            return {
                fmt::format("sigmoid_tile_init<{}u>();", param1),
                fmt::format("sigmoid_tile<{1}, {2}u>({0});", idst, (int32_t)param0, param1)};
        }
        case UnaryOpType::ERF:
            return {
                fmt::format("erf_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("erf_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::ERFC:
            return {
                fmt::format("erfc_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("erfc_tile<{1}u>({0});", idst, (uint32_t)param0)};
        case UnaryOpType::RDIV: {
            uint32_t rounding_mode_value = params[1];
            static constexpr const char* rounding_mode_strs[] = {
                "ckernel::RoundingMode::None", "ckernel::RoundingMode::Trunc", "ckernel::RoundingMode::Floor"};
            return {
                "rdiv_tile_init();",
                fmt::format(
                    "rdiv_tile<{}>({}, {:#x}u);",
                    rounding_mode_strs[rounding_mode_value],
                    idst,
                    std::bit_cast<uint32_t>(param0))};
        }
        case UnaryOpType::RSUB:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::UINT16 || input_dtype == DataType::UINT8) {
                TT_THROW("Unsupported data type");
            } else if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "rsub_unary_int32_tile_init();",
                    fmt::format(
                        "rsub_unary_int32_tile({}, {}u);",
                        idst,
                        std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))};
            } else {
                return {
                    "rsub_tile_init();", fmt::format("rsub_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
        case UnaryOpType::RPOW:
            return {"rpow_tile_init();", fmt::format("rpow_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::SUB_UNARY_SFPU:
            if (input_dtype == DataType::INT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format(
                        "sub_unary_tile_int32({}, {}u);",
                        idst,
                        std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))};
            } else if (input_dtype == DataType::UINT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    // TODO: Use uint32_t tile API here once implemented #27621
                    fmt::format("sub_unary_tile_int32({}, {}u);", idst, (uint)param0_raw)};
            } else {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format("sub_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
            }
        case UnaryOpType::ADD_UNARY_SFPU:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    fmt::format(
                        "add_unary_tile_int32({}, {}u);",
                        idst,
                        std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))};
            }
            if (input_dtype == DataType::UINT32) {
                return {
                    "binop_with_scalar_tile_init();",
                    // TODO: Use uint32_t tile API here once implemented
                    fmt::format("add_unary_tile_int32({}, {}u);", idst, (uint)param0_raw)};
            }
            return {
                "binop_with_scalar_tile_init();",
                fmt::format("add_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::MUL_UNARY_SFPU:
            return {
                "binop_with_scalar_tile_init();",
                fmt::format("mul_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::DIV_UNARY_SFPU:
            return {
                "binop_with_scalar_tile_init();",
                fmt::format("div_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(1.0f / param0))};
        case UnaryOpType::UNARY_NE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "unary_ne_tile_init();",
                    fmt::format("unary_ne_tile_int32({}, {}u);", idst, std::bit_cast<uint32_t>(param0_raw))};
            }
            return {
                "unary_ne_tile_init();",
                fmt::format("unary_ne_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::UNARY_EQ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "unary_eq_tile_init();",
                    fmt::format("unary_eq_tile_int32({}, {}u);", idst, std::bit_cast<uint32_t>(param0_raw))};
            }
            return {
                "unary_eq_tile_init();",
                fmt::format("unary_eq_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::UNARY_GT:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "unary_gt_tile_init();",
                    fmt::format("unary_gt_tile_int32({}, {});", idst, std::bit_cast<uint32_t>(param0_raw))};
            }
            return {
                "unary_gt_tile_init();",
                fmt::format("unary_gt_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::UNARY_LT:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "unary_lt_tile_init();",
                    fmt::format("unary_lt_tile_int32({}, {});", idst, std::bit_cast<uint32_t>(param0_raw))};
            }
            return {
                "unary_lt_tile_init();",
                fmt::format("unary_lt_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::UNARY_GE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "unary_ge_tile_init();",
                    fmt::format("unary_ge_tile_int32({}, {});", idst, std::bit_cast<uint32_t>(param0_raw))};
            }
            return {
                "unary_ge_tile_init();",
                fmt::format("unary_ge_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::UNARY_LE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {
                    "unary_le_tile_init();",
                    fmt::format("unary_le_tile_int32({}, {});", idst, std::bit_cast<uint32_t>(param0_raw))};
            }
            return {
                "unary_le_tile_init();",
                fmt::format("unary_le_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::SOFTPLUS: {
            TT_ASSERT(params.size() == 2, "Expected softplus to take 2 parameters");
            float param1 = params[1];
            return {
                "softplus_tile_init();",
                fmt::format(
                    "softplus_tile({}, {:#x}u, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0),  // Pass reciprocal to avoid doing it on device
                    std::bit_cast<uint32_t>(param1))};
        }
        case UnaryOpType::PRELU_SFPU: {
            return {
                "prelu_tile_init();", fmt::format("prelu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
        }
        case UnaryOpType::TYPECAST:
            TT_ASSERT(params.size() == 2, "Expected eltwise_typecast to take 2 parameters");
            return {
                fmt::format(
                    "typecast_tile_init<{0}u, {1}u>();",
                    static_cast<uint32_t>(datatype_to_dataformat_converter((DataType)params[0])),
                    static_cast<uint32_t>(datatype_to_dataformat_converter((DataType)params[1]))),
                fmt::format(
                    "typecast_tile<{1}u, {2}u>({0});",
                    idst,
                    static_cast<uint32_t>(datatype_to_dataformat_converter((DataType)params[0])),
                    static_cast<uint32_t>(datatype_to_dataformat_converter((DataType)params[1])))};
        case UnaryOpType::BITCAST:
            // Bitcast uses identity kernel (copy_tile + pack_tile) - no LLK needed
            // Parameters are input_dtype and output_dtype, but we don't need them for the kernel
            return {};
        case UnaryOpType::MAXIMUM:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {"unary_max_tile_init();", fmt::format("unary_max_int32_tile({}, {}u);", idst, (uint)params[0])};
            }
            return {
                "unary_max_tile_init();",
                fmt::format("unary_max_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::MINIMUM:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
                return {"unary_min_tile_init();", fmt::format("unary_min_int32_tile({}, {}u);", idst, (uint)params[0])};
            }
            return {
                "unary_min_tile_init();",
                fmt::format("unary_min_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};

        case UnaryOpType::CELU:
            return {
                "celu_tile_init();",
                fmt::format(
                    "celu_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(1.0f / param0))};
        case UnaryOpType::HARDSHRINK:
        case UnaryOpType::LOGIT: return {};
        case UnaryOpType::SOFTSHRINK:
            return {
                "softshrink_tile_init();",
                fmt::format("softshrink_tile({}, {}u);", idst, std::bit_cast<uint32_t>(param0))};
        case UnaryOpType::WHERE_TSS: {
            std::string where_call;
            if (input_dtype == DataType::INT32) {
                where_call = fmt::format("where_int32_tile({0}, {1}, {2}, {0});", idst, 1, 2);
            } else if (input_dtype == DataType::UINT32) {
                where_call = fmt::format("where_uint32_tile({0}, {1}, {2}, {0});", idst, 1, 2);
            } else if (input_dtype == DataType::FLOAT32) {
                where_call = fmt::format("where_fp32_tile({0}, {1}, {2}, {0});", idst, 1, 2);
            } else {
                where_call = fmt::format("where_tile({0}, {1}, {2}, {0});", idst, 1, 2);
            }
            return std::make_pair("where_tile_init();", where_call);
        }
        case UnaryOpType::CLAMP_TSS: {
            float param1 = params[1];
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {
                    "clamp_tile_init();",
                    fmt::format("clamp_tile_int32({}, {}, {});", idst, (uint)params[0], (uint)params[1])};
            }
            return {
                "clamp_tile_init();",
                fmt::format(
                    "clamp_tile({}, {}, {});",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(param1))};
        }
        case UnaryOpType::HARDTANH: {
            float param1 = params[1];
            return {
                "hardtanh_tile_init();",
                fmt::format(
                    "hardtanh_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(param1))};
        }
        case UnaryOpType::THRESHOLD: {
            float param1 = params[1];
            return {
                "threshold_tile_init();",
                fmt::format(
                    "threshold_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(param1))};
        }
        case UnaryOpType::SELU: {
            TT_FATAL(params.size() == 2, "Expected selu to take 2 parameters");
            float param1 = params[1];
            return {
                "selu_tile_init();",
                fmt::format(
                    "selu_tile({}, {:#x}u, {:#x}u);",
                    idst,
                    std::bit_cast<uint32_t>(param0),
                    std::bit_cast<uint32_t>(param1))};
        }
        case UnaryOpType::HARDMISH: {
            return {
                fmt::format("hardmish_tile_init<{}u>();", (uint32_t)param0),
                fmt::format("hardmish_tile<{1}u>({0});", idst, (uint32_t)param0)};
        }
        case UnaryOpType::RSQRT: {
            return {"rsqrt_tile_init<false>();", fmt::format("rsqrt_tile<false, {1}>({0});", idst, param0_raw)};
        }
        case UnaryOpType::SQRT: {
            return {"sqrt_tile_init();", fmt::format("sqrt_tile<{1}>({0});", idst, param0_raw)};
        }
        default: TT_THROW("unexpected parameterized op type {}", op_type);
    };
}

std::pair<std::string, std::string> get_op_init_and_func_default(
    UnaryOpType op_type, std::string idst, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::BITWISE_NOT: return {"bitwise_not_tile_init();", fmt::format("bitwise_not_tile({});", idst)};
        case UnaryOpType::RECIP: return {"recip_tile_init<false>();", fmt::format("recip_tile<false>({});", idst)};
        case UnaryOpType::GELU: return {"gelu_tile_init();", fmt::format("gelu_tile({});", idst)};
        case UnaryOpType::LOG: return {"log_tile_init();", fmt::format("log_tile({});", idst)};
        case UnaryOpType::LOG1P: return {"log1p_tile_init();", fmt::format("log1p_tile({});", idst)};
        case UnaryOpType::TANH: return {"tanh_tile_init();", fmt::format("tanh_tile({});", idst)};
        case UnaryOpType::RELU:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"relu_tile_init();", fmt::format("relu_tile_int32({});", idst)};
            }
            return {"relu_tile_init();", fmt::format("        relu_tile({});", idst)};

        case UnaryOpType::SIGNBIT:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"signbit_tile_init();", fmt::format("signbit_tile_int32({});", idst)};
            }
            return {"signbit_tile_init();", fmt::format("        signbit_tile({});", idst)};

        case UnaryOpType::SIN: return {"sin_tile_init();", fmt::format("sin_tile({});", idst)};
        case UnaryOpType::COS: return {"cos_tile_init();", fmt::format("cos_tile({});", idst)};
        case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
        case UnaryOpType::SINH: return {"sinh_tile_init();", fmt::format("sinh_tile({});", idst)};
        case UnaryOpType::ISFINITE: return {"isfinite_tile_init();", fmt::format("isfinite_tile({});", idst)};
        case UnaryOpType::ISINF: return {"isinf_tile_init();", fmt::format("isinf_tile({});", idst)};
        case UnaryOpType::ISPOSINF: return {"isposinf_tile_init();", fmt::format("isposinf_tile({});", idst)};
        case UnaryOpType::ISNEGINF: return {"isneginf_tile_init();", fmt::format("isneginf_tile({});", idst)};
        case UnaryOpType::ISNAN: return {"isnan_tile_init();", fmt::format("isnan_tile({});", idst)};
        case UnaryOpType::LOGICAL_NOT_UNARY:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile_int32({});", idst)};
            }
            if (input_dtype == DataType::UINT32) {
                return {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile_uint32({});", idst)};
            }
            if (input_dtype == DataType::UINT16) {
                return {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile_uint16({});", idst)};
            }
            return {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile({});", idst)};
        case UnaryOpType::I0: return {"i0_tile_init();", fmt::format("i0_tile({});", idst)};
        case UnaryOpType::I1: return {"i1_tile_init();", fmt::format("i1_tile({});", idst)};
        case UnaryOpType::EXP: return {"exp_tile_init();", fmt::format("exp_tile({});", idst)};
        case UnaryOpType::SIGMOID: return {"sigmoid_tile_init();", fmt::format("sigmoid_tile({});", idst)};
        case UnaryOpType::ERF: return {"erf_tile_init();", fmt::format("erf_tile({0});", idst)};
        case UnaryOpType::ERFC: return {"erfc_tile_init();", fmt::format("erfc_tile({});", idst)};
        case UnaryOpType::ERFINV: return {"erfinv_tile_init();", fmt::format("erfinv_tile({});", idst)};
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            return {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3ede5bd9u);", idst)};
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            return {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3fb8aa3bu);", idst)};
        case UnaryOpType::ABS: return {"abs_tile_init();", fmt::format("abs_tile({});", idst)};
        case UnaryOpType::ABS_INT32: return {"abs_tile_init();", fmt::format("abs_tile_int32({});", idst)};
        case UnaryOpType::SIGN: return {"sign_tile_init();", fmt::format("sign_tile({});", idst)};
        case UnaryOpType::SQUARE:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype.value() == DataType::INT32) {
                return {"mul_int32_tile_init();", fmt::format("mul_int32_tile({0}, {0}, {0});", idst)};
            }
            if (input_dtype.value() == DataType::UINT32) {
                return {"mul_int32_tile_init();", fmt::format("mul_uint32_tile({0},         {0}, {0});", idst)};
            }
            if (input_dtype.value() == DataType::UINT16) {
                return {"mul_int_tile_init();", fmt::format("mul_uint16_tile({0},         {0}, {0});", idst)};
            }
            return {"        square_tile_init();", fmt::format("square_tile({});", idst)};
        case UnaryOpType::TILED_PROD: return {"tiled_prod_tile_init();", fmt::format("tiled_prod_tile({});", idst)};
        case UnaryOpType::EQZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype.value() == DataType::INT32) {
                return {"eqz_tile_init();", fmt::format("eqz_tile_int32({});", idst)};
            }
            if (input_dtype.value() == DataType::UINT16) {
                return {"eqz_tile_init();", fmt::format("eqz_tile_uint16({});", idst)};
            }
            if (input_dtype.value() == DataType::UINT32) {
                return {"eqz_tile_init();", fmt::format("eqz_tile_uint32({});", idst)};
            }
            return {"eqz_tile_init();", fmt::format("eqz_tile({});", idst)};
        case UnaryOpType::NEZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"nez_tile_init();", fmt::format("nez_tile_int32({});", idst)};
            }
            if (input_dtype == DataType::UINT16) {
                return {"nez_tile_init();", fmt::format("nez_tile_uint16({});", idst)};
            }
            if (input_dtype.value() == DataType::UINT32) {
                return {"nez_tile_init();", fmt::format("nez_tile_uint32({});", idst)};
            }
            return {"nez_tile_init();", fmt::format("nez_tile({});", idst)};
        case UnaryOpType::LTZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"ltz_tile_init();", fmt::format("ltz_tile_int32({});", idst)};
            }
            return {"ltz_tile_init();", fmt::format("        ltz_tile({});", idst)};

        case UnaryOpType::GTZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"gtz_tile_init();", fmt::format("gtz_tile_int32({});", idst)};
            }
            return {"gtz_tile_init();", fmt::format("        gtz_tile({});", idst)};

        case UnaryOpType::GEZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"gez_tile_init();", fmt::format("gez_tile_int32({});", idst)};
            }
            return {"gez_tile_init();", fmt::format("        gez_tile({});", idst)};

        case UnaryOpType::LEZ:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"lez_tile_init();", fmt::format("lez_tile_int32({});", idst)};
            }
            return {"lez_tile_init();", fmt::format("        lez_tile({});", idst)};

        case UnaryOpType::SQRT: return {"sqrt_tile_init();", fmt::format("sqrt_tile({});", idst)};
        case UnaryOpType::RSQRT: return {"rsqrt_tile_init();", fmt::format("rsqrt_tile({});", idst)};
        case UnaryOpType::EXP2: return {"exp2_tile_init();", fmt::format("exp2_tile({});", idst)};
        case UnaryOpType::EXPM1: return {"expm1_tile_init();", fmt::format("expm1_tile({});", idst)};
        case UnaryOpType::ASIN: return {"asin_tile_init();", fmt::format("asin_tile({});", idst)};
        case UnaryOpType::ASINH: return {"asinh_tile_init();", fmt::format("asinh_tile({});", idst)};
        case UnaryOpType::ACOS: return {"acos_tile_init();", fmt::format("acos_tile({});", idst)};
        case UnaryOpType::ACOSH: return {"acosh_tile_init();", fmt::format("acosh_tile({});", idst)};
        case UnaryOpType::ATAN: return {"atan_tile_init();", fmt::format("atan_tile({});", idst)};
        case UnaryOpType::ATANH: return {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)};
        case UnaryOpType::TAN: return {"tan_tile_init();", fmt::format("tan_tile({});", idst)};
        case UnaryOpType::SILU: return {"silu_tile_init();", fmt::format("silu_tile({});", idst)};
        case UnaryOpType::FLOOR:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            return {"rounding_op_tile_init();", fmt::format("floor_tile({});", idst)};
        case UnaryOpType::CEIL:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            return {"rounding_op_tile_init();", fmt::format("ceil_tile({});", idst)};
        case UnaryOpType::TRUNC:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            return {"rounding_op_tile_init();", fmt::format("trunc_tile({});", idst)};
        case UnaryOpType::FRAC:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            return {"rounding_op_tile_init();", fmt::format("frac_tile({});", idst)};
        case UnaryOpType::RELU6: return {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
        case UnaryOpType::NEG:
            TT_FATAL(
                input_dtype.has_value(), "Missing input dtype: Expected a valid input dtype, but none was provided.");
            if (input_dtype == DataType::INT32) {
                return {"negative_tile_init();", fmt::format("negative_tile_int32({});", idst)};
            }
            return {"negative_tile_init();", fmt::format("        negative_tile({});", idst)};

        case UnaryOpType::ALT_COMPLEX_ROTATE90:
            return {"alt_complex_rotate90_tile_init();", fmt::format("alt_complex_rotate90_tile({});", idst)};
        case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
        case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
        case UnaryOpType::MISH:
        case UnaryOpType::IDENTITY:
        case UnaryOpType::BITCAST:
            // Bitcast uses identity kernel (copy_tile + pack_tile) - no LLK needed
            // Parameters are input_dtype and output_dtype, but we don't need them for the kernel
        case UnaryOpType::TANHSHRINK:
        case UnaryOpType::HARDSWISH:
        case UnaryOpType::CBRT:
        case UnaryOpType::LOGSIGMOID: return {};
        case UnaryOpType::HARDMISH: return {"hardmish_tile_init();", fmt::format("hardmish_tile({});", idst)};
        default: TT_THROW("Undefined non-parametrized op type {}", op_type);
    }
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
    if (name == "relu") {
        return UnaryWithParam(UnaryOpType::RELU);
    }
    if (name == "relu6") {
        return UnaryWithParam(UnaryOpType::RELU6);
    }
    if (name == "gelu") {
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(false));
    }
    if (name == "gelu_approx") {
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(true));
    }
    if (name == "silu") {
        return UnaryWithParam(UnaryOpType::SILU);
    }
    if (name == "sigmoid") {
        return UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(VecMode::RC), static_cast<float>(false)});
    }
    if (name == "sigmoid_approx") {
        return UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(VecMode::RC), static_cast<float>(true)});
    }
    if (name == "hardsigmoid") {
        return UnaryWithParam(UnaryOpType::HARDSIGMOID);
    }
    if (name == "sqrt") {
        return UnaryWithParam(UnaryOpType::SQRT, {static_cast<float>(false)});
    }
    if (name == "rsqrt") {
        return UnaryWithParam(UnaryOpType::RSQRT, {static_cast<float>(false)});
    }
    if (name == "exp") {
        return UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true));
    }
    if (name == "recip") {
        return UnaryWithParam(UnaryOpType::RECIP);
    }
    if (name == "log") {
        return UnaryWithParam(UnaryOpType::LOG, static_cast<float>(true));
    }
    if (name == "log1p") {
        return UnaryWithParam(UnaryOpType::LOG1P, static_cast<float>(true));
    }
    if (name == "tanh") {
        return UnaryWithParam(UnaryOpType::TANH, static_cast<float>(false));
    }
    if (name == "log2") {
        return UnaryWithParam(UnaryOpType::LOG2, static_cast<float>(true));
    }
    if (name == "log10") {
        return UnaryWithParam(UnaryOpType::LOG10, static_cast<float>(true));
    }
    if (name == "sin") {
        return UnaryWithParam(UnaryOpType::SIN);
    }
    if (name == "cos") {
        return UnaryWithParam(UnaryOpType::COS);
    }
    if (name == "cosh") {
        return UnaryWithParam(UnaryOpType::COSH);
    }
    if (name == "sinh") {
        return UnaryWithParam(UnaryOpType::SINH);
    }
    if (name == "abs") {
        return UnaryWithParam(UnaryOpType::ABS);
    }
    if (name == "abs_int32") {
        return UnaryWithParam(UnaryOpType::ABS_INT32);
    }
    if (name == "sign") {
        return UnaryWithParam(UnaryOpType::SIGN);
    }
    if (name == "square") {
        return UnaryWithParam(UnaryOpType::SQUARE);
    }
    if (name == "softplus") {
        return UnaryWithParam(UnaryOpType::SOFTPLUS, {1.0f, 20.0f, 0.0f});  // beta=1, threshold=20, approx_mode=0
    }
    if (name == "selu") {
        return UnaryWithParam(UnaryOpType::SELU);
    }
    if (name == "alt_complex_rotate90") {
        return UnaryWithParam(UnaryOpType::ALT_COMPLEX_ROTATE90);
    }
    if (name == "hardmish") {
        return UnaryWithParam(UnaryOpType::HARDMISH, static_cast<float>(true));
    }
    if (name == "mish") {
        return UnaryWithParam(UnaryOpType::MISH);
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

std::string_view get_compute_kernel_path(UnaryOpType op_type, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::MISH: return "mish_kernel.cpp";
        case UnaryOpType::TANHSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "tanhshrink_sfpu_kernel.cpp";
            } else {
                return "tanhshrink_kernel.cpp";
            }
        case UnaryOpType::IDENTITY: return "eltwise_identity_kernel.cpp";
        case UnaryOpType::WHERE_TSS: return "where_tss_kernel.cpp";
        case UnaryOpType::LOGIT: return "logit_kernel.cpp";
        case UnaryOpType::HARDSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "hardshrink_kernel_sfpu.cpp";
            } else {
                return "hardshrink_kernel.cpp";
            }
        case UnaryOpType::HARDSWISH:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "hardswish_kernel_sfpu.cpp";
            } else {
                return "hardswish_kernel.cpp";
            }
        case UnaryOpType::CBRT: return "cbrt_kernel.cpp";
        case UnaryOpType::LOGSIGMOID: return "logsigmoid_kernel.cpp";
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

std::uint32_t pack_scalar_runtime_arg_impl(std::uint32_t param, DataType /*dtype*/) { return param; }

std::uint32_t pack_scalar_runtime_arg_impl(std::int32_t param, DataType /*dtype*/) {
    return std::bit_cast<std::uint32_t>(param);
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
