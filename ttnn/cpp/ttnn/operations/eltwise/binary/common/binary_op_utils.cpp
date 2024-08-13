// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op_utils.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::binary::utils {

using ttnn::operations::unary::UnaryWithParam;
using ttnn::operations::unary::UnaryOpType;


std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_dtype,
    const std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<std::vector<UnaryWithParam>> fused_activations,
    const std::optional<unary::UnaryWithParam> input_tensor_a_activation) {

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
            defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LT:
            defines.merge(get_defines(UnaryOpType::LTZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::GTE:
            defines.merge(get_defines(UnaryOpType::GEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LTE:
            defines.merge(get_defines(UnaryOpType::LEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::EQ:
            defines.merge(get_defines(UnaryOpType::EQZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::NE:
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::SQUARED_DIFFERENCE:
            defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst));
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
        case BinaryOpType::DIV_FAST:
            // Divide by a non-zero tensor
            defines.merge(get_defines(UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGICAL_OR:
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0"));
            defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst));
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
        default: TT_ASSERT(false && "Undefined op type");
    }

    using DataType = tt::tt_metal::DataType;
    if(input_dtype.has_value() && output_dtype.has_value() &&
        ((input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::UINT16) ||
        (input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::INT32) ||
        (input_dtype.value() == DataType::UINT16 && output_dtype.value() == DataType::BFLOAT16) ||
        (input_dtype.value() == DataType::INT32 && output_dtype.value() == DataType::BFLOAT16) ||
        (input_dtype.value() == DataType::FLOAT32 && output_dtype.value() == DataType::BFLOAT16) ||
        (input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::FLOAT32) ||
        (input_dtype.value() == DataType::FLOAT32 && output_dtype.value() == DataType::UINT16) ||
        (input_dtype.value() == DataType::UINT16 && output_dtype.value() == DataType::FLOAT32) ||
        (input_dtype.value() == DataType::FLOAT32 && output_dtype.value() == DataType::INT32) ||
        (input_dtype.value() == DataType::INT32 && output_dtype.value() == DataType::FLOAT32) ||
        (input_dtype.value() == DataType::BFLOAT8_B && output_dtype.value() == DataType::UINT16) ||
        (input_dtype.value() == DataType::UINT16 && output_dtype.value() == DataType::BFLOAT8_B) ||
        (input_dtype.value() == DataType::BFLOAT8_B && output_dtype.value() == DataType::INT32) ||
        (input_dtype.value() == DataType::INT32 && output_dtype.value() == DataType::BFLOAT8_B) ||
        (input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::UINT32) ||
        (input_dtype.value() == DataType::UINT32 && output_dtype.value() == DataType::BFLOAT16) ||
        (input_dtype.value() == DataType::FLOAT32 && output_dtype.value() == DataType::UINT32) ||
        (input_dtype.value() == DataType::UINT32 && output_dtype.value() == DataType::FLOAT32) ||
        (input_dtype.value() == DataType::BFLOAT8_B && output_dtype.value() == DataType::UINT32) ||
        (input_dtype.value() == DataType::UINT32 && output_dtype.value() == DataType::BFLOAT8_B) ||
        (input_dtype.value() == DataType::UINT16 && output_dtype.value() == DataType::UINT32))){
        TT_ASSERT(defines.count("SFPU_OP_CHAIN_0") == 0 && "SFPU_OP_CHAIN_0 already defined");

        auto in_dataformat = std::to_string((uint32_t)datatype_to_dataformat_converter(input_dtype.value()));
        auto out_dataformat = std::to_string((uint32_t)datatype_to_dataformat_converter(output_dtype.value()));
        defines.insert(
            {"SFPU_OP_CHAIN_0",
             fmt::format("typecast_tile_init(); typecast_tile<{0}u, {1}u>(i);", in_dataformat, out_dataformat)});
        defines.insert({"SFPU_OP_TYPECAST_INCLUDE", "1"});
    }

    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_TYPE"] = op_binary_type.c_str();
    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations.value().size() == 1 and
            fused_activations.value().at(0).op_type == UnaryOpType::RELU) {
            defines["PACK_RELU"] = "1";
        } else {
            defines.merge(ttnn::operations::unary::utils::get_block_defines(fused_activations.value(), "0", idst));
        }
    }

    if (input_tensor_a_activation.has_value() ) {
        defines.merge(ttnn::operations::unary::utils::get_defines(input_tensor_a_activation.value().op_type, std::nullopt, "PRE_IN0_0", idst));
    }

    return defines;
}

}
