// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op_utils.hpp"

#include <tt-metalium/assert.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/types.hpp"

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
           (input == UINT32 && output == BFLOAT16) || (input == UINT32 && output == FLOAT32) ||
           (input == UINT16 && output == INT32) || (input == INT32 && output == UINT16) ||
           (input == UINT32 && output == UINT16);
}

std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_dtype,
    const std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    std::map<std::string, std::string> defines;
    std::string op_name = "sub_tiles";
    std::string op_binary_type = "EltwiseBinaryType::ELWSUB";
    std::string idst = "i";

    using ttnn::operations::unary::utils::get_defines;

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
        case BinaryOpType::GT:
            defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::LT:
            defines.merge(get_defines(UnaryOpType::LTZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::GE:
            defines.merge(get_defines(UnaryOpType::GEZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::LE:
            defines.merge(get_defines(UnaryOpType::LEZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::EQ:
            defines.merge(get_defines(UnaryOpType::EQZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::NE:
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::SQUARED_DIFFERENCE:
            defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::BIAS_GELU:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(get_defines(UnaryOpType::GELU, std::vector<float>{0}, "0", idst));
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0"));
            defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(get_defines(UnaryOpType::LOG, std::nullopt, "0", idst));
            break;
        case BinaryOpType::RSUB:
            //  rsub(a,b) = b - a
            defines.merge(get_defines(UnaryOpType::NEG, std::nullopt, "PRE_IN0_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::DIV:
            // Divide by a non-zero tensor
            defines.merge(get_defines(UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGICAL_OR:
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0", "0", input_dtype));
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0", "0", input_dtype));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::LOGICAL_XOR:
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0", "0", input_dtype));
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0", "0", input_dtype));
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst, input_dtype));
            break;
        case BinaryOpType::LDEXP:
            defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGADDEXP2:
            defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0"));
            defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(get_defines(UnaryOpType::LOG2, std::nullopt, "0", idst));
            break;
        default: TT_THROW("Undefined op type {}", op_type);
    }

    if (input_dtype.has_value() && output_dtype.has_value() && is_typecast(*input_dtype, *output_dtype)) {
        TT_ASSERT(defines.count("SFPU_OP_CHAIN_0") == 0, "SFPU_OP_CHAIN_0 already defined");

        auto in_dataformat = (uint32_t)datatype_to_dataformat_converter(input_dtype.value());
        auto out_dataformat = (uint32_t)datatype_to_dataformat_converter(output_dtype.value());
        defines.insert(
            {"SFPU_OP_CHAIN_0",
             fmt::format("typecast_tile_init(); typecast_tile<{0}u, {1}u>(i);", in_dataformat, out_dataformat)});
        defines.insert({"SFPU_OP_TYPECAST_INCLUDE", "1"});
    }

    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_TYPE"] = op_binary_type.c_str();
    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations->size() == 1 and
            fused_activations->at(0).op_type == UnaryOpType::RELU and not input_tensor_a_activation.has_value()) {
            defines["PACK_RELU"] = "1";
        } else {
            defines.merge(ttnn::operations::unary::utils::get_block_defines(*fused_activations, "0", idst));
        }
    }

    if (input_tensor_a_activation.has_value()) {
        defines.merge(ttnn::operations::unary::utils::get_defines(
            input_tensor_a_activation.value().op_type, std::nullopt, "PRE_IN0_0", idst, input_dtype));
    }

    return defines;
}

std::map<std::string, std::string> get_defines_fp32(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_a_dtype,
    const std::optional<tt::tt_metal::DataType> input_b_dtype,
    const std::optional<std::vector<UnaryWithParam>>& fused_activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    std::map<std::string, std::string> new_defines;
    std::string op_name = "sub_binary_tile";
    std::string idst1 = "i*2";    // tile index for input A in dst and final output
    std::string idst2 = "i*2+1";  // tile index for input B in dst
    std::string idst = "i";       // tile index for input prescaling

    using ttnn::operations::unary::utils::get_defines;
    switch (op_type) {
        case BinaryOpType::ADD:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                new_defines.insert({"ADD_INT_INIT", fmt::format("add_int_tile_init();")});
                op_name = "add_int32_tile";
            } else if (input_a_dtype == DataType::UINT32 && input_b_dtype == DataType::UINT32) {
                new_defines.insert({"ADD_INT_INIT", fmt::format("add_int_tile_init();")});
                op_name = "add_uint32_tile";
            } else if (input_a_dtype == DataType::UINT16 && input_b_dtype == DataType::UINT16) {
                new_defines.insert({"ADD_INT_INIT", fmt::format("add_int_tile_init();")});
                op_name = "add_uint16_tile";
            } else {
                new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
                op_name = "add_binary_tile";
            }
            break;
        case BinaryOpType::SUB:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                new_defines.insert({"SUB_INT_INIT", fmt::format("sub_int_tile_init();")});
                op_name = "sub_int32_tile";
            } else if (input_a_dtype == DataType::UINT16 && input_b_dtype == DataType::UINT16) {
                new_defines.insert({"SUB_INT_INIT", fmt::format("sub_int_tile_init();")});
                op_name = "sub_uint16_tile";
            } else {
                new_defines.insert({"BINOP_INIT", "sub_binary_tile_init();"});
                op_name = "sub_binary_tile";
            }
            break;
        case BinaryOpType::MUL:
            if (input_a_dtype == DataType::UINT16 && input_b_dtype == DataType::UINT16) {
                new_defines.insert({"MUL_INT_INIT", fmt::format("mul_int_tile_init();")});
                op_name = "mul_uint16_tile";
            } else {
                new_defines.insert({"BINOP_INIT", fmt::format("mul_binary_tile_init();")});
                op_name = "mul_binary_tile";
            }
            break;
        case BinaryOpType::RSUB:
            new_defines.insert({"BINOP_INIT", fmt::format("rsub_binary_tile_init();")});
            op_name = "rsub_binary_tile";
            break;
        case BinaryOpType::POWER:
            new_defines.insert({"BINOP_INIT", fmt::format("power_binary_tile_init();")});
            op_name = "power_binary_tile";
            break;
        case BinaryOpType::DIV:
            new_defines.insert({"BINOP_INIT", fmt::format("div_binary_tile_init();")});
            op_name = "div_binary_tile";
            break;
        case BinaryOpType::BITWISE_AND:
            if (input_a_dtype == DataType::UINT16 && input_b_dtype == DataType::UINT16) {
                new_defines.insert({"BITWISE_UINT16_INIT", fmt::format("binary_bitwise_tile_init();")});
                op_name = "bitwise_and_uint16_binary_tile";
            } else {
                new_defines.insert({"BITWISE_INIT", fmt::format("binary_bitwise_tile_init();")});
                op_name = "bitwise_and_binary_tile";
            }
            break;
        case BinaryOpType::BITWISE_OR:
            if (input_a_dtype == DataType::UINT16 && input_b_dtype == DataType::UINT16) {
                new_defines.insert({"BITWISE_UINT16_INIT", fmt::format("binary_bitwise_tile_init();")});
                op_name = "bitwise_or_uint16_binary_tile";
            } else {
                new_defines.insert({"BITWISE_INIT", fmt::format("binary_bitwise_tile_init();")});
                op_name = "bitwise_or_binary_tile";
            }
            break;
        case BinaryOpType::BITWISE_XOR:
            if (input_a_dtype == DataType::UINT16 && input_b_dtype == DataType::UINT16) {
                new_defines.insert({"BITWISE_UINT16_INIT", fmt::format("binary_bitwise_tile_init();")});
                op_name = "bitwise_xor_uint16_binary_tile";
            } else {
                new_defines.insert({"BITWISE_INIT", fmt::format("binary_bitwise_tile_init();")});
                op_name = "bitwise_xor_binary_tile";
            }
            break;
        case BinaryOpType::LEFT_SHIFT:
            new_defines.insert({"SHIFT_INIT", fmt::format("binary_shift_tile_init();")});
            op_name = "binary_left_shift_tile";
            break;
        case BinaryOpType::RIGHT_SHIFT:
            new_defines.insert({"SHIFT_INIT", fmt::format("binary_shift_tile_init();")});
            op_name = "binary_right_shift_tile";
            break;
        case BinaryOpType::LOGICAL_RIGHT_SHIFT:
            new_defines.insert({"SHIFT_INIT", fmt::format("binary_shift_tile_init();")});
            op_name = "binary_logical_right_shift_tile";
            break;
        case BinaryOpType::MAXIMUM:
            new_defines.insert({"BINOP_INIT", fmt::format("binary_max_tile_init();")});
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "binary_max_int32_tile";
            } else {
                op_name = "binary_max_tile";
            }
            break;
        case BinaryOpType::MINIMUM:
            new_defines.insert({"BINOP_INIT", fmt::format("binary_min_tile_init();")});
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "binary_min_int32_tile";
            } else {
                op_name = "binary_min_tile";
            }
            break;
        case BinaryOpType::GCD:
            new_defines.insert({"BINOP_INIT", fmt::format("gcd_tile_init();")});
            op_name = "gcd_tile";
            break;
        case BinaryOpType::LCM:
            new_defines.insert({"BINOP_INIT", fmt::format("lcm_tile_init();")});
            op_name = "lcm_tile";
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            new_defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0"));
            new_defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0"));
            new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
            op_name = "add_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::LOG, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LOGADDEXP2:
            new_defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0"));
            new_defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
            op_name = "add_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::LOG2, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LDEXP:
            new_defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_binary_tile";
            break;
        case BinaryOpType::SQUARED_DIFFERENCE:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::BIAS_GELU:
            new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
            op_name = "add_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::GELU, std::vector<float>{0}, "0", idst1));
            break;
        case BinaryOpType::LOGICAL_OR:
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0", "0", input_a_dtype));
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0", "0", input_b_dtype));
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                new_defines.insert({"ADD_INT32_INIT", fmt::format("add_int32_tile_init();")});
                op_name = "add_int32_tile";
            } else {
                new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
                op_name = "add_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::LOGICAL_XOR:
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0", "0", input_a_dtype));
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0", "0", input_b_dtype));
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        // applied on A-B
        case BinaryOpType::GT:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::LT:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::LTZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::GE:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::GEZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::LE:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::LEZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::EQ:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::EQZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        case BinaryOpType::NE:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                op_name = "sub_int32_tile";
            } else {
                op_name = "sub_binary_tile";
            }
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst1, input_a_dtype));
            break;
        default:
            log_debug(tt::LogOp, "Undefined op type {}", op_type);
            TT_FATAL(false, "Undefined op type for binary sfpu operation {}", op_type);
    }

    new_defines.insert({"BINARY_SFPU_OP", fmt::format("{}({}, {});", op_name, idst1, idst2)});

    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations.value().size() == 1 and
            fused_activations.value().at(0).op_type == UnaryOpType::RELU) {
            new_defines["PACK_RELU"] = "1";
        } else {
            new_defines.merge(ttnn::operations::unary::utils::get_block_defines(fused_activations.value(), "0", idst1));
        }
    }

    if (input_tensor_a_activation.has_value()) {
        new_defines.merge(ttnn::operations::unary::utils::get_defines(
            input_tensor_a_activation.value().op_type, std::nullopt, "PRE_IN0_0", idst));
    }

    return new_defines;
}

}  // namespace ttnn::operations::binary::utils
