// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt_stl/assert.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <enchantum/enchantum.hpp>

namespace ttnn::operations::binary_ng {

struct Lowercase {
    std::string_view view;
};

}  // namespace ttnn::operations::binary_ng

template <>
struct fmt::formatter<ttnn::operations::binary_ng::Lowercase> : fmt::formatter<std::string_view> {
    auto format(const ttnn::operations::binary_ng::Lowercase& value, fmt::format_context& ctx) const {
        auto out = ctx.out();
        for (char c : value.view) {
            *out++ = std::tolower(static_cast<unsigned char>(c));
        }
        return out;
    }
};

namespace ttnn::operations::binary_ng {

BinaryNgKernelConfig::BinaryNgKernelConfig(SubtileBroadcastType subtile_broadcast_type) {
    switch (subtile_broadcast_type) {
        case SubtileBroadcastType::NONE:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterScalar;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::SCALAR_A:;
            compute_kernel = KernelName::ComputeBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::SCALAR_B:
            compute_kernel = KernelName::ComputeBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A:
        case SubtileBroadcastType::ROW_B:
            compute_kernel = KernelName::ComputeNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::COL_A:
        case SubtileBroadcastType::ROW_B_COL_A:
            compute_kernel = KernelName::ComputeBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::COL_B:
        case SubtileBroadcastType::ROW_A_COL_B:
            compute_kernel = KernelName::ComputeBcast;
            bcast_input = 1;
            break;
    }
}

std::string BinaryNgKernelConfig::bcast_input_str() const {
    if (bcast_input.has_value()) {
        return std::to_string(*bcast_input);
    }
    return "";
}

std::string get_kernel_file_path(KernelName kernel_name, bool is_sfpu, bool is_where_op) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels";
    constexpr std::string_view root_ng = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcastNg: return fmt::format(dataflow, root_ng, "reader_interleaved_no_bcast.cpp");
        case KernelName::ReaderRowBcastNg: return fmt::format(dataflow, root_ng, "reader_interleaved_row_bcast.cpp");
        case KernelName::ReaderColBcastNg: return fmt::format(dataflow, root_ng, "reader_interleaved_col_bcast.cpp");
        case KernelName::ReaderRowBColABcastNg:
            return fmt::format(dataflow, root_ng, "reader_interleaved_row_col_mixed_bcast.cpp");
        case KernelName::ReaderScalarBcastNg:
            return fmt::format(dataflow, root_ng, "reader_interleaved_scalar_bcast.cpp");
        case KernelName::WriterNoBcastNg: return fmt::format(dataflow, root_ng, "writer_interleaved_no_bcast.cpp");
        case KernelName::ReaderNoBcast: return fmt::format(dataflow, root, "reader_interleaved_no_bcast.cpp");
        case KernelName::WriterScalar: return fmt::format(dataflow, root, "writer_interleaved_scalar.cpp");
        case KernelName::ComputeNoBcast:
            return fmt::format(
                compute,
                root,
                is_where_op ? "eltwise_where_no_bcast.cpp"
                            : (is_sfpu ? "eltwise_binary_sfpu_no_bcast.cpp" : "eltwise_binary_no_bcast.cpp"));
        case KernelName::ComputeBcast:
            return fmt::format(
                compute,
                root,
                is_where_op ? "eltwise_where_sfpu.cpp" : (is_sfpu ? "eltwise_binary_sfpu.cpp" : "eltwise_binary.cpp"));
        case KernelName::ComputeScalar:
            return fmt::format(
                compute,
                root,
                is_where_op ? "eltwise_where_sfpu_scalar"
                            : (is_sfpu ? "eltwise_binary_sfpu_scalar.cpp" : "eltwise_binary_scalar.cpp"));
        case KernelName::ComputeRowBcastNg:
            return fmt::format(
                compute,
                root_ng,
                is_where_op ? "eltwise_where_sfpu_row_bcast.cpp"
                            : (is_sfpu ? "eltwise_binary_sfpu_row_bcast.cpp" : "eltwise_binary_row_bcast.cpp"));
        case KernelName::ComputeRowColBcastNg:
            return fmt::format(
                compute,
                root_ng,
                is_where_op ? "eltwise_where_sfpu_row_col_bcast.cpp"
                            : (is_sfpu ? "eltwise_binary_sfpu_row_col_bcast.cpp" : "eltwise_binary_row_col_bcast.cpp"));
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

//  EnumT can either be FpuBinaryOp or SfpuBinaryOp
template <class EnumT>
OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<EnumT>, std::optional<DataType> dtype) :
    binary_op(EnumT::SUB) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: binary_op = EnumT::ADD; break;
        case BinaryOpType::SUB: break;
        case BinaryOpType::MUL: binary_op = EnumT::MUL; break;
        case BinaryOpType::DIV:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::DIV;
            } else {
                process_rhs = unary::UnaryOpType::RECIP;
                binary_op = FpuBinaryOp::MUL;
            }
            break;
        case BinaryOpType::DIV_FLOOR: binary_op = SfpuBinaryOp::DIV_FLOOR; break;
        case BinaryOpType::DIV_TRUNC: binary_op = SfpuBinaryOp::DIV_TRUNC; break;
        // b - a
        case BinaryOpType::RSUB:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::RSUB;
            } else {
                process_lhs = unary::UnaryOpType::NEG;
                binary_op = FpuBinaryOp::ADD;
            }
            break;
        case BinaryOpType::LT:
            if (dtype != DataType::INT32) {
                postprocess = unary::UnaryOpType::LTZ;
            } else {
                binary_op = SfpuBinaryOp::LT;
            }
            break;
        case BinaryOpType::GT:
            if (dtype != DataType::INT32) {
                postprocess = unary::UnaryOpType::GTZ;
            } else {
                binary_op = SfpuBinaryOp::GT;
            }
            break;
        case BinaryOpType::GE:
            if (dtype != DataType::INT32) {
                postprocess = unary::UnaryOpType::GEZ;
            } else {
                binary_op = SfpuBinaryOp::GE;
            }
            break;
        case BinaryOpType::LE:
            if (dtype != DataType::INT32) {
                postprocess = unary::UnaryOpType::LEZ;
            } else {
                binary_op = SfpuBinaryOp::LE;
            }
            break;
        case BinaryOpType::EQ: postprocess = unary::UnaryOpType::EQZ; break;
        case BinaryOpType::NE: postprocess = unary::UnaryOpType::NEZ; break;
        // (a-b)**2
        case BinaryOpType::SQUARED_DIFFERENCE: postprocess = unary::UnaryOpType::SQUARE; break;
        // gelu(a+b)
        case BinaryOpType::BIAS_GELU:
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::GELU;
            break;
        case BinaryOpType::LOGICAL_AND:
            process_lhs = unary::UnaryOpType::NEZ;
            process_rhs = unary::UnaryOpType::NEZ;
            binary_op = EnumT::MUL;
            postprocess = unary::UnaryOpType::NEZ;
            break;
        case BinaryOpType::LOGICAL_OR:
            process_lhs = unary::UnaryOpType::NEZ;
            process_rhs = unary::UnaryOpType::NEZ;
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::NEZ;
            break;
        case BinaryOpType::LOGICAL_XOR:
            process_lhs = unary::UnaryOpType::NEZ;
            process_rhs = unary::UnaryOpType::NEZ;
            postprocess = unary::UnaryOpType::NEZ;
            break;
        // a * (2**b)
        case BinaryOpType::LDEXP:
            process_rhs = unary::UnaryOpType::EXP2;
            binary_op = EnumT::MUL;
            break;
        // log( exp(a) + exp(b) )
        case BinaryOpType::LOGADDEXP:
            process_lhs = unary::UnaryOpType::EXP;
            process_rhs = unary::UnaryOpType::EXP;
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::LOG;
            break;
        // log2( 2**a + 2**b )
        case BinaryOpType::LOGADDEXP2:
            process_lhs = unary::UnaryOpType::EXP2;
            process_rhs = unary::UnaryOpType::EXP2;
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::LOG2;
            break;
        case BinaryOpType::BITWISE_AND:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::BITWISE_AND;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::BITWISE_OR:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::BITWISE_OR;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::BITWISE_XOR:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::BITWISE_XOR;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::LEFT_SHIFT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::LEFT_SHIFT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::RIGHT_SHIFT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::RIGHT_SHIFT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::LOGICAL_RIGHT_SHIFT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::LOGICAL_RIGHT_SHIFT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::POWER:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::POWER;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::QUANT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::QUANT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::REQUANT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::REQUANT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::DEQUANT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::DEQUANT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::MAXIMUM:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::MAXIMUM;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::MINIMUM:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::MINIMUM;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::GCD:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::GCD;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::LCM:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::LCM;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::XLOGY:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::XLOGY;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::WHERE_TTS:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::WHERE;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::WHERE_TST:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::WHERE;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        // sqrt(a^2 + b^2)
        case BinaryOpType::HYPOT:
            process_lhs = unary::UnaryOpType::SQUARE;
            process_rhs = unary::UnaryOpType::SQUARE;
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::SQRT;
            break;
        default: TT_THROW("Unsupported binary op {}", binary_op_type);
    }
}

std::pair<std::string, std::string> get_sfpu_init_fn(OpConfig::SfpuBinaryOp sfpu_binary_op, DataType dtype) {
    using enum OpConfig::SfpuBinaryOp;
    switch (sfpu_binary_op) {
        case ADD:
            if (dtype == DataType::INT32) {
                return {"add_int_tile_init();", "add_int32_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"add_int_tile_init();", "add_uint32_tile"};
            } else if (dtype == DataType::UINT16) {
                return {"add_int_tile_init();", "add_uint16_tile"};
            } else {
                return {"add_binary_tile_init();", "add_binary_tile"};
            }
        case SUB:
            if (dtype == DataType::INT32) {
                return {"sub_int_tile_init();", "sub_int32_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"sub_int_tile_init();", "sub_uint32_tile"};
            } else if (dtype == DataType::UINT16) {
                return {"sub_int_tile_init();", "sub_uint16_tile"};
            } else {
                return {"sub_binary_tile_init();", "sub_binary_tile"};
            }
        case MUL:
            if (dtype == DataType::UINT16) {
                return {"mul_int_tile_init();", "mul_uint16_tile"};
            } else if (dtype == DataType::INT32) {
                return {"mul_int32_tile_init();", "mul_int32_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"mul_int32_tile_init();", "mul_uint32_tile"};
            } else {
                return {"mul_binary_tile_init();", "mul_binary_tile"};
            }
        case DIV:
            if (dtype == DataType::INT32) {
                return {"div_int32_tile_init();", "div_int32_tile"};
            } else {
                return {"div_binary_tile_init();", "div_binary_tile"};
            }
        case DIV_FLOOR: return {"div_int32_floor_tile_init();", "div_int32_floor_tile"};
        case DIV_TRUNC: return {"div_int32_trunc_tile_init();", "div_int32_trunc_tile"};
        case POWER: return {"power_binary_tile_init();", "power_binary_tile"};
        case RSUB:
            if (dtype == DataType::INT32) {
                return {"rsub_int_tile_init();", "rsub_int32_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"rsub_int_tile_init();", "rsub_uint32_tile"};
            } else if (dtype == DataType::UINT16) {
                return {"rsub_int_tile_init();", "rsub_uint16_tile"};
            } else {
                return {"rsub_binary_tile_init();", "rsub_binary_tile"};
            }
        case GCD: return {"gcd_tile_init();", "gcd_tile"};
        case LCM: return {"lcm_tile_init();", "lcm_tile"};
        case LEFT_SHIFT:
            if (dtype == DataType::UINT32) {
                return {"binary_shift_tile_init();", "binary_left_shift_uint32_tile"};
            } else if (dtype == DataType::INT32) {
                return {"binary_shift_tile_init();", "binary_left_shift_int32_tile"};
            } else {
                return {"binary_shift_tile_init();", "binary_left_shift_tile"};
            }
        case RIGHT_SHIFT:
            if (dtype == DataType::UINT32) {
                return {"binary_shift_tile_init();", "binary_right_shift_uint32_tile"};
            } else if (dtype == DataType::INT32) {
                return {"binary_shift_tile_init();", "binary_right_shift_int32_tile"};
            } else {
                return {"binary_shift_tile_init();", "binary_right_shift_tile"};
            }
        case LOGICAL_RIGHT_SHIFT:
            if (dtype == DataType::UINT32) {
                return {"binary_shift_tile_init();", "binary_logical_right_shift_uint32_tile"};
            } else if (dtype == DataType::INT32) {
                return {"binary_shift_tile_init();", "binary_logical_right_shift_int32_tile"};
            } else {
                return {"binary_shift_tile_init();", "binary_logical_right_shift_tile"};
            }
        case BITWISE_AND:
            if (dtype == DataType::UINT16) {
                return {"binary_bitwise_tile_init();", "bitwise_and_uint16_binary_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"binary_bitwise_tile_init();", "bitwise_and_uint32_binary_tile"};
            } else {
                return {"binary_bitwise_tile_init();", "bitwise_and_binary_tile"};
            }
        case BITWISE_OR:
            if (dtype == DataType::UINT16) {
                return {"binary_bitwise_tile_init();", "bitwise_or_uint16_binary_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"binary_bitwise_tile_init();", "bitwise_or_uint32_binary_tile"};
            } else {
                return {"binary_bitwise_tile_init();", "bitwise_or_binary_tile"};
            }
        case BITWISE_XOR:
            if (dtype == DataType::UINT16) {
                return {"binary_bitwise_tile_init();", "bitwise_xor_uint16_binary_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"binary_bitwise_tile_init();", "bitwise_xor_uint32_binary_tile"};
            } else {
                return {"binary_bitwise_tile_init();", "bitwise_xor_binary_tile"};
            }
        case MAXIMUM:
            if (dtype == DataType::INT32) {
                return {"binary_max_tile_init();", "binary_max_int32_tile"};
            } else {
                return {"binary_max_tile_init();", "binary_max_tile"};
            }
        case MINIMUM:
            if (dtype == DataType::INT32) {
                return {"binary_min_tile_init();", "binary_min_int32_tile"};
            } else {
                return {"binary_min_tile_init();", "binary_min_tile"};
            }
        case QUANT: return {"quant_tile_init(get_arg_val<uint32_t>(QUANT_ZERO_POINT_RT_ARGS_IDX));", "quant_tile"};
        case REQUANT:
            return {"requant_tile_init(get_arg_val<uint32_t>(QUANT_ZERO_POINT_RT_ARGS_IDX));", "requant_tile"};
        case DEQUANT:
            return {"dequant_tile_init(get_arg_val<uint32_t>(QUANT_ZERO_POINT_RT_ARGS_IDX));", "dequant_tile"};
        case XLOGY: return {"xlogy_binary_tile_init();", "xlogy_binary_tile"};
        case LT: return {"lt_int32_tile_init();", "lt_int32_tile"};
        case GT: return {"gt_int32_tile_init();", "gt_int32_tile"};
        case GE: return {"ge_int32_tile_init();", "ge_int32_tile"};
        case LE: return {"le_int32_tile_init();", "le_int32_tile"};
        case WHERE:
            if (dtype == DataType::INT32) {
                return {"where_tile_init();", "where_int32_tile"};
            } else if (dtype == DataType::UINT32) {
                return {"where_tile_init();", "where_uint32_tile"};
            } else if (dtype == DataType::FLOAT32) {
                return {"where_tile_init();", "where_fp32_tile"};
            } else {
                return {"where_tile_init();", "where_tile"};
            }
        default: TT_THROW("Unsupported sfpu binary op {}", sfpu_binary_op);
    }
}

std::map<std::string, std::string> OpConfig::as_defines(DataType dtype) const {
    std::map<std::string, std::string> defines;

    if (!is_sfpu_op()) {
        auto fpu_binary_op = std::get<FpuBinaryOp>(binary_op);
        auto binary_op_str = enchantum::to_string(fpu_binary_op);
        defines["BINARY_OP"] = fmt::format("{}_tiles", Lowercase{binary_op_str});
        defines["BINARY_OP_TYPE"] = fmt::format("EltwiseBinaryType::ELW{}", binary_op_str);
        return defines;
    }
    auto&& [tile_init, tile_fn] = get_sfpu_init_fn(std::get<SfpuBinaryOp>(binary_op), dtype);
    defines["BINARY_SFPU_INIT"] = std::move(tile_init);
    defines["BINARY_SFPU_OP"] = std::move(tile_fn);
    return defines;
}

void add_activation_defines(
    std::map<std::string, std::string>& defines,
    tt::stl::Span<const unary::EltwiseUnaryWithParam> activations,
    std::string_view operand,
    std::optional<DataType> dtype) {
    defines[fmt::format("PROCESS_{}_ACTIVATIONS(i)", operand)] = std::accumulate(
        activations.begin(),
        activations.end(),
        std::string{},
        [&](std::string&& process, const unary::EltwiseUnaryWithParam& a) {
            const auto& [op_init, op_func] = std::visit(
                [&](auto params) { return unary::utils::get_op_init_and_func(a.type(), params, "i", dtype); },
                a.get_params());
            process += op_init;
            process += op_func;
            unary::utils::update_macro_defines(a.type(), defines);
            return std::move(process);
        });
}

std::map<std::string, std::string> make_dataflow_defines(
    const DataType dtype, const std::optional<DataType> b_dtype_opt) {
    std::map<std::string, std::string> defines;
    const auto b_dtype = b_dtype_opt.value_or(dtype);
    // to maintain backward compatibility, we need to support both dtype and b_dtype
    if (dtype == DataType::FLOAT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<float>";
        defines["FILL_WITH_VALUE_FLOAT"] = "fill_with_val<1024, float>";
    } else if (dtype == DataType::INT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<int32_t>";
        defines["FILL_WITH_VALUE"] = "fill_with_val<1024, int32_t>";
    } else if (dtype == DataType::UINT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element<uint32_t>";
        defines["FILL_WITH_VALUE"] = "fill_with_val<1024, uint32_t>";
    } else {
        defines["FILL_TILE_WITH_FIRST_COLUMN"] = "fill_tile_with_first_column_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ROW"] = "fill_tile_with_first_row_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ELEMENT"] = "fill_tile_with_first_element_bfloat16";
        defines["FILL_WITH_VALUE"] = "fill_with_val_bfloat16";
    }

    if (b_dtype == DataType::FLOAT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element<float>";
        defines["FILL_WITH_VALUE_FLOAT_B"] = "fill_with_val<1024, float>";
    } else if (b_dtype == DataType::INT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element<int32_t>";
        defines["FILL_WITH_VALUE_B"] = "fill_with_val<1024, int32_t>";
    } else if (b_dtype == DataType::UINT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element<uint32_t>";
        defines["FILL_WITH_VALUE_B"] = "fill_with_val<1024, uint32_t>";
    } else {
        defines["FILL_TILE_WITH_FIRST_COLUMN_B"] = "fill_tile_with_first_column_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ROW_B"] = "fill_tile_with_first_row_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_B"] = "fill_tile_with_first_element_bfloat16";
        defines["FILL_WITH_VALUE_B"] = "fill_with_val_bfloat16";
    }
    return defines;
}

bool OpConfig::is_sfpu_op() const { return std::holds_alternative<SfpuBinaryOp>(binary_op); }

uint32_t pack_scalar_runtime_arg(const unary::ScalarVariant scalar, const DataType dtype, const bool is_quant_op) {
    // std::visit([&](auto v) {
    //     std::cout << "pack_scalar_runtime_arg: " << v << std::endl;
    // }, scalar);
    return std::visit(
        [&](auto v) -> uint32_t {
            // Always pass the more accurate fp32 when the quantization scale is passed as a scalar
            if ((dtype == DataType::FLOAT32) || is_quant_op) {
                return std::bit_cast<uint32_t>(static_cast<float>(v));
            }
            if (dtype == DataType::INT32) {
                return std::bit_cast<uint32_t>(static_cast<int32_t>(v));
            }
            if (dtype == DataType::UINT32) {
                return static_cast<uint32_t>(v);
            }
            // TODO: #27672: Truncation should be removed once we figure a root cause of regression without it
            auto scalar_bf16 = bfloat16::truncate(static_cast<float>(v));
            return pack_two_bfloat16_into_uint32({scalar_bf16, scalar_bf16});
        },
        scalar);
}

template OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<FpuBinaryOp>, std::optional<DataType>);
template OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<SfpuBinaryOp>, std::optional<DataType>);

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    auto ret = shard_spec;

    // Calculate volume of all dimensions EXCEPT the last (width)
    // This is the "collapsed height" for sharding purposes
    uint32_t from_volume_except_width = 1;
    uint32_t to_volume_except_width = 1;

    const int rank = std::max(from_shape.rank(), to_shape.rank());

    // Accumulate all dimensions except the last
    for (int i = 0; i < rank - 1; ++i) {
        uint32_t from_dim = (i < from_shape.rank()) ? from_shape[i] : 1;
        uint32_t to_dim = (i < to_shape.rank()) ? to_shape[i] : 1;
        from_volume_except_width *= from_dim;
        to_volume_except_width *= to_dim;
    }

    // Get width dimensions
    uint32_t from_width = from_shape[-1];
    uint32_t to_width = to_shape[-1];

    // Adjust shard shape based on full volume ratios
    TT_FATAL(from_volume_except_width > 0, "Invalid from_shape: volume is zero");
    TT_FATAL(from_width > 0, "Invalid from_shape: width dimension is zero");
    ret.shape[0] = std::max((ret.shape[0] * to_volume_except_width) / from_volume_except_width, 32u);
    ret.shape[1] = std::max((ret.shape[1] * to_width) / from_width, 32u);
    return ret;
}

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

inline auto is_uneven(const TensorSpec& t) {
    if (not t.memory_config().is_sharded()) {
        return false;
    }

    const auto& shape = t.padded_shape();
    const auto& shard = get_shard_spec(t)->shape;
    const auto rank = shape.rank();

    // Compute product of all dimensions except the last
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }

    return (volume_except_last % shard[0]) != 0 or (shape[-1] % shard[1]) != 0;
}

bool is_native_L1_sharding(const TensorSpec& a, const std::optional<TensorSpec>& b, const MemoryConfig& c) {
    // scalar value treated as interleaved
    if (!b.has_value()) {
        return false;
    }

    // does not work for width and block sharding, pcc error,
    // maybe support later to improve performance
    // if (!b.has_value() && a.memory_config().is_sharded()) {
    //     return !is_uneven(a);
    // }

    if (!c.is_sharded()) {
        return false;
    }

    // a and b identical shape, no broadcast on any dimension
    if (b.has_value() && (a.logical_shape() == b->logical_shape()) && (a.memory_config() == b->memory_config())) {
        if (is_uneven(a) || is_uneven(*b)) {
            return false;
        }
        if (a.memory_config().buffer_type() == BufferType::DRAM ||
            b->memory_config().buffer_type() == BufferType::DRAM || c.buffer_type() == BufferType::DRAM) {
            return false;
        }

        // Check if output grid differs from input grids - if so, cannot use native sharding
        // This will force resharding through interleaved path
        if (c.is_sharded() && c.shard_spec().has_value()) {
            const auto& c_grid = c.shard_spec()->grid;
            if (a.memory_config().is_sharded() && a.memory_config().shard_spec().has_value()) {
                const auto& a_grid = a.memory_config().shard_spec()->grid;
                if (a_grid != c_grid) {
                    // Different grids require resharding - treat as interleaved
                    return false;
                }
            }
            if (b->memory_config().is_sharded() && b->memory_config().shard_spec().has_value()) {
                const auto& b_grid = b->memory_config().shard_spec()->grid;
                if (b_grid != c_grid) {
                    // Different grids require resharding - treat as interleaved
                    return false;
                }
            }
        }

        if ((a.memory_config().is_sharded() && a.memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if (b->memory_config().is_sharded() && b->memory_config().buffer_type() == BufferType::L1) {
            return true;
        }
        if (c.is_sharded() && c.buffer_type() == BufferType::L1) {
            return true;
        }
    }

    return false;
}

ttnn::Shape compute_broadcasted_output(const ttnn::Shape& shape_a, const ttnn::Shape& shape_b) {
    // Broadcasting Rules Overview:
    // - If the two tensors have different ranks, we virtually pad the smaller-rank tensor's shape
    //   with ones on the left (i.e., higher-order dimensions) until both shapes have the same length.
    // - For each dimension (starting from the rightmost), the sizes are compatible if:
    //     - They are equal, or
    //     - One of them is 1 (the dimension can be broadcast to match the other size).

    const int rank_a = shape_a.rank();
    const int rank_b = shape_b.rank();
    const int larger_rank = std::max(rank_a, rank_b);
    SmallVector<uint32_t> output_shape(larger_rank, 1);
    for (int i = -1; i >= -larger_rank; --i) {
        auto dim_a = (i >= -rank_a) ? shape_a[i] : 1;
        auto dim_b = (i >= -rank_b) ? shape_b[i] : 1;
        if (dim_a != 1 && dim_b != 1) {
            output_shape[i + larger_rank] = dim_a;
        } else {
            output_shape[i + larger_rank] = dim_a + dim_b - 1;
        }
    }
    return ttnn::Shape(output_shape);
}

MemoryConfig compute_mem_config_actual(const ttnn::Tensor& input_tensor_a, const ttnn::Shape& shape_b) {
    // Compute adjusted shard spec for output shape
    const auto& padded_a_shape = input_tensor_a.padded_shape();
    const auto& logical_out_shape =
        operations::binary_ng::compute_broadcasted_output(input_tensor_a.logical_shape(), shape_b);
    const auto& padded_out_shape = input_tensor_a.tensor_spec().tensor_layout().compute_padded_shape(logical_out_shape);

    auto adjusted_shard_spec = ttnn::operations::binary_ng::adjust_to_shape(
        *input_tensor_a.memory_config().shard_spec(), padded_a_shape, padded_out_shape);

    return MemoryConfig(
        input_tensor_a.memory_config().memory_layout(),
        input_tensor_a.memory_config().buffer_type(),
        adjusted_shard_spec);
}
}  // namespace ttnn::operations::binary_ng
