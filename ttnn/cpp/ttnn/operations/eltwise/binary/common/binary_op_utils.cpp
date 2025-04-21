// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op_utils.hpp"

#include <tt-metalium/assert.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "cpp/ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::binary::utils {

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

bool is_typecast(tt::tt_metal::DataType input, tt::tt_metal::DataType output) {
    using enum tt::tt_metal::DataType;

    return (input == BFLOAT4_B && output == INT32) || (input == BFLOAT4_B && output == UINT16) ||
           (input == BFLOAT4_B && output == UINT32) || (input == BFLOAT8_B && output == INT32) ||
           (input == BFLOAT8_B && output == UINT16) || (input == BFLOAT8_B && output == UINT32) ||
           (input == BFLOAT16 && output == INT32) || (input == BFLOAT16 && output == UINT16) ||
           (input == BFLOAT16 && output == UINT32) || (input == FLOAT32 && output == BFLOAT16) ||
           (input == FLOAT32 && output == INT32) || (input == FLOAT32 && output == UINT16) ||
           (input == FLOAT32 && output == UINT32) || (input == INT32 && output == BFLOAT4_B) ||
           (input == INT32 && output == BFLOAT8_B) || (input == INT32 && output == BFLOAT16) ||
           (input == INT32 && output == FLOAT32) || (input == UINT16 && output == BFLOAT4_B) ||
           (input == UINT16 && output == BFLOAT8_B) || (input == UINT16 && output == BFLOAT16) ||
           (input == UINT16 && output == FLOAT32) || (input == UINT16 && output == UINT32) ||
           (input == UINT32 && output == BFLOAT4_B) || (input == UINT32 && output == BFLOAT8_B) ||
           (input == UINT32 && output == BFLOAT16) || (input == UINT32 && output == FLOAT32);
}

void append_defines(
    tt::tt_metal::KernelDescriptor::Defines& defines,
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_dtype,
    const std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    std::string op_name = "sub_tiles";
    std::string op_binary_type = "EltwiseBinaryType::ELWSUB";
    std::string idst = "i";

    using ttnn::operations::unary::utils::append_defines;

    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::GT: append_defines(defines, UnaryOpType::GTZ, std::nullopt, "0", idst); break;
        case BinaryOpType::LT: append_defines(defines, UnaryOpType::LTZ, std::nullopt, "0", idst); break;
        case BinaryOpType::GTE: append_defines(defines, UnaryOpType::GEZ, std::nullopt, "0", idst); break;
        case BinaryOpType::LTE: append_defines(defines, UnaryOpType::LEZ, std::nullopt, "0", idst); break;
        case BinaryOpType::EQ: append_defines(defines, UnaryOpType::EQZ, std::nullopt, "0", idst, input_dtype); break;
        case BinaryOpType::NE: append_defines(defines, UnaryOpType::NEZ, std::nullopt, "0", idst, input_dtype); break;
        case BinaryOpType::SQUARED_DIFFERENCE:
            append_defines(defines, UnaryOpType::SQUARE, std::nullopt, "0", idst);
            break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "0", idst);
            break;
        case BinaryOpType::BIAS_GELU:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            append_defines(defines, UnaryOpType::GELU, std::vector<float>{0}, "0", idst);
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            append_defines(defines, UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0");
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            append_defines(defines, UnaryOpType::LOG, std::nullopt, "0", idst);
            break;
        case BinaryOpType::RSUB:
            //  rsub(a,b) = b - a
            append_defines(defines, UnaryOpType::NEG, std::nullopt, "PRE_IN0_0");
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::DIV:
            // Divide by a non-zero tensor
            append_defines(defines, UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0");
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGICAL_OR:
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0");
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            append_defines(defines, UnaryOpType::GTZ, std::nullopt, "0", idst);
            break;
        case BinaryOpType::LOGICAL_XOR:
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0");
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "0", idst);
            break;
        case BinaryOpType::LDEXP:
            append_defines(defines, UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0");
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGADDEXP2:
            append_defines(defines, UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0");
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            append_defines(defines, UnaryOpType::LOG2, std::nullopt, "0", idst);
            break;
        default: TT_THROW("Undefined op type {}", op_type);
    }

    if (input_dtype.has_value() && output_dtype.has_value() && is_typecast(*input_dtype, *output_dtype)) {
        auto in_dataformat = (uint32_t)datatype_to_dataformat_converter(input_dtype.value());
        auto out_dataformat = (uint32_t)datatype_to_dataformat_converter(output_dtype.value());
        defines.emplace_back(
            "SFPU_OP_CHAIN_0",
            fmt::format("typecast_tile_init(); typecast_tile<{0}u, {1}u>(i);", in_dataformat, out_dataformat));
        defines.emplace_back("SFPU_OP_TYPECAST_INCLUDE", "1");
    }

    defines.emplace_back("ELTWISE_OP", op_name.c_str());
    defines.emplace_back("ELTWISE_OP_TYPE", op_binary_type.c_str());
    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations->size() == 1 and
            fused_activations->at(0).op_type == UnaryOpType::RELU and not input_tensor_a_activation.has_value()) {
            defines.emplace_back("PACK_RELU", "1");
        } else {
            ttnn::operations::unary::utils::append_block_defines(defines, *fused_activations, "0", idst);
        }
    }

    if (input_tensor_a_activation.has_value()) {
        ttnn::operations::unary::utils::append_defines(
            defines, input_tensor_a_activation.value().op_type, std::nullopt, "PRE_IN0_0", idst);
    }
}

std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_a_dtype,
    const std::optional<tt::tt_metal::DataType> input_b_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    tt::tt_metal::KernelDescriptor::Defines defines;
    append_defines(defines, op_type, input_a_dtype, input_b_dtype, fused_activations, input_tensor_a_activation);
    return std::map<std::string, std::string>(defines.begin(), defines.end());
}

void append_defines_fp32(
    tt::tt_metal::KernelDescriptor::Defines& defines,
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_a_dtype,
    const std::optional<tt::tt_metal::DataType> input_b_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    std::string op_name = "sub_binary_tile";
    std::string idst1 = "i*2";    // tile index for input A in dst and final output
    std::string idst2 = "i*2+1";  // tile index for input B in dst
    std::string idst = "i";       // tile index for input prescaling

    using ttnn::operations::unary::utils::append_defines;
    switch (op_type) {
        case BinaryOpType::ADD:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                defines.emplace_back("ADD_INT32_INIT", fmt::format("add_int32_tile_init();"));
                op_name = "add_int32_tile";
            } else {
                defines.emplace_back("BINOP_INIT", fmt::format("add_binary_tile_init();"));
                op_name = "add_binary_tile";
            }
            break;
        case BinaryOpType::SUB:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                defines.emplace_back("SUB_INT32_INIT", "sub_int32_tile_init();");
                op_name = "sub_int32_tile";
            } else {
                defines.emplace_back("BINOP_INIT", "sub_binary_tile_init();");
                op_name = "sub_binary_tile";
            }
            break;
        case BinaryOpType::MUL:
            defines.emplace_back("BINOP_INIT", fmt::format("mul_binary_tile_init();"));
            op_name = "mul_binary_tile";
            break;
        case BinaryOpType::RSUB:
            defines.emplace_back("BINOP_INIT", fmt::format("rsub_binary_tile_init();"));
            op_name = "rsub_binary_tile";
            break;
        case BinaryOpType::POWER:
            defines.emplace_back("BINOP_INIT", fmt::format("power_binary_tile_init();"));
            op_name = "power_binary_tile";
            break;
        case BinaryOpType::DIV:
            defines.emplace_back("BINOP_INIT", fmt::format("div_binary_tile_init();"));
            op_name = "div_binary_tile";
            break;
        case BinaryOpType::BITWISE_AND:
            defines.emplace_back("BITWISE_INIT", fmt::format("binary_bitwise_tile_init();"));
            op_name = "and_binary_tile";
            break;
        case BinaryOpType::BITWISE_OR:
            defines.emplace_back("BITWISE_INIT", fmt::format("binary_bitwise_tile_init();"));
            op_name = "or_binary_tile";
            break;
        case BinaryOpType::BITWISE_XOR:
            defines.emplace_back("BITWISE_INIT", fmt::format("binary_bitwise_tile_init();"));
            op_name = "xor_binary_tile";
            break;
        case BinaryOpType::LEFT_SHIFT:
            defines.emplace_back("SHIFT_INIT", fmt::format("binary_shift_tile_init();"));
            op_name = "binary_left_shift_tile";
            break;
        case BinaryOpType::RIGHT_SHIFT:
            defines.emplace_back("SHIFT_INIT", fmt::format("binary_shift_tile_init();"));
            op_name = "binary_right_shift_tile";
            break;
        case BinaryOpType::MAXIMUM:
            defines.emplace_back("BINOP_INIT", fmt::format("binary_max_tile_init();"));
            op_name = "binary_max_tile";
            break;
        case BinaryOpType::MINIMUM:
            defines.emplace_back("BINOP_INIT", fmt::format("binary_min_tile_init();"));
            op_name = "binary_min_tile";
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            append_defines(defines, UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0");
            defines.emplace_back("BINOP_INIT", fmt::format("add_binary_tile_init();"));
            op_name = "add_binary_tile";
            append_defines(defines, UnaryOpType::LOG, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::LOGADDEXP2:
            append_defines(defines, UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0");
            defines.emplace_back("BINOP_INIT", fmt::format("add_binary_tile_init();"));
            op_name = "add_binary_tile";
            append_defines(defines, UnaryOpType::LOG2, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::LDEXP:
            append_defines(defines, UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0");
            op_name = "mul_binary_tile";
            break;
        case BinaryOpType::SQUARED_DIFFERENCE:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::SQUARE, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_binary_tile";
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::BIAS_GELU:
            defines.emplace_back("BINOP_INIT", fmt::format("add_binary_tile_init();"));
            op_name = "add_binary_tile";
            append_defines(defines, UnaryOpType::GELU, std::vector<float>{0}, "0", idst1);
            break;
        case BinaryOpType::LOGICAL_OR:
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0");
            defines.emplace_back("BINOP_INIT", fmt::format("add_binary_tile_init();"));
            op_name = "add_binary_tile";
            append_defines(defines, UnaryOpType::GTZ, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::LOGICAL_XOR:
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0");
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0");
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "0", idst1);
            break;
        // applied on A-B
        case BinaryOpType::GT:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::GTZ, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::LT:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::LTZ, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::GTE:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::GEZ, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::LTE:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::LEZ, std::nullopt, "0", idst1);
            break;
        case BinaryOpType::EQ:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::EQZ, std::nullopt, "0", idst1, input_a_dtype);
            break;
        case BinaryOpType::NE:
            op_name = "sub_binary_tile";
            append_defines(defines, UnaryOpType::NEZ, std::nullopt, "0", idst1);
            break;
        default:
            tt::log_debug(tt::LogOp, "Undefined op type {}", op_type);
            TT_FATAL(false, "Undefined op type for binary sfpu operation {}", op_type);
    }

    defines.emplace_back("BINARY_SFPU_OP", fmt::format("{}({}, {});", op_name, idst1, idst2));

    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations.value().size() == 1 and
            fused_activations.value().at(0).op_type == UnaryOpType::RELU) {
            defines.emplace_back("PACK_RELU", "1");
        } else {
            ttnn::operations::unary::utils::append_block_defines(defines, *fused_activations, "0", idst1);
        }
    }

    if (input_tensor_a_activation.has_value()) {
        ttnn::operations::unary::utils::append_defines(
            defines, input_tensor_a_activation.value().op_type, std::nullopt, "PRE_IN0_0", idst);
    }
}

std::map<std::string, std::string> get_defines_fp32(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> in_a_dtype,
    const std::optional<tt::tt_metal::DataType> in_b_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    tt::tt_metal::KernelDescriptor::Defines defines;
    append_defines_fp32(defines, op_type, in_a_dtype, in_b_dtype, fused_activations, input_tensor_a_activation);
    return std::map<std::string, std::string>(defines.begin(), defines.end());
}
}  // namespace ttnn::operations::binary::utils
