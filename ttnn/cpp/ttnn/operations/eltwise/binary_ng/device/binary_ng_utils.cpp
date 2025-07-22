// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/assert.hpp>

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
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::SCALAR_A:
            reader_kernel = KernelName::ReaderScalarBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::SCALAR_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterScalarBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A:
            reader_kernel = KernelName::ReaderRowBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::ROW_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterRowBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::COL_A:
            reader_kernel = KernelName::ReaderColBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::COL_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterColBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A_COL_B:
            reader_kernel = KernelName::ReaderRowBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterColBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_B_COL_A:
            reader_kernel = KernelName::ReaderColBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterRowBcast;
            bcast_input = 0;
            break;
    }
}

std::string BinaryNgKernelConfig::bcast_input_str() const {
    if (bcast_input.has_value()) {
        return std::to_string(*bcast_input);
    }
    return "";
}

std::string get_kernel_file_path(KernelName kernel_name, bool is_sfpu) {
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
        case KernelName::ReaderRowBcast: return fmt::format(dataflow, root, "reader_interleaved_row_bcast.cpp");
        case KernelName::ReaderColBcast: return fmt::format(dataflow, root, "reader_interleaved_col_bcast.cpp");
        case KernelName::ReaderScalarBcast: return fmt::format(dataflow, root, "reader_interleaved_scalar_bcast.cpp");
        case KernelName::WriterNoBcast: return fmt::format(dataflow, root, "writer_interleaved_no_bcast.cpp");
        case KernelName::WriterRowBcast: return fmt::format(dataflow, root, "writer_interleaved_row_bcast.cpp");
        case KernelName::WriterColBcast: return fmt::format(dataflow, root, "writer_interleaved_col_bcast.cpp");
        case KernelName::WriterScalarBcast: return fmt::format(dataflow, root, "writer_interleaved_scalar_bcast.cpp");
        case KernelName::WriterScalar: return fmt::format(dataflow, root, "writer_interleaved_scalar.cpp");
        case KernelName::ComputeNoBcast:
            return fmt::format(
                compute, root, is_sfpu ? "eltwise_binary_sfpu_no_bcast.cpp" : "eltwise_binary_no_bcast.cpp");
        case KernelName::ComputeBcast:
            return fmt::format(compute, root, is_sfpu ? "eltwise_binary_sfpu.cpp" : "eltwise_binary.cpp");
        case KernelName::ComputeScalar:
            return fmt::format(compute, root, is_sfpu ? "eltwise_binary_sfpu_scalar.cpp" : "eltwise_binary_scalar.cpp");
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

//  EnumT can either be FpuBinaryOp or SfpuBinaryOp
template <class EnumT>
OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<EnumT>) : binary_op(EnumT::SUB) {
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
        // b - a
        case BinaryOpType::RSUB:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::RSUB;
            } else {
                process_lhs = unary::UnaryOpType::NEG;
                binary_op = FpuBinaryOp::ADD;
            }
            break;
        case BinaryOpType::GT: postprocess = unary::UnaryOpType::GTZ; break;
        case BinaryOpType::LT: postprocess = unary::UnaryOpType::LTZ; break;
        case BinaryOpType::GE: postprocess = unary::UnaryOpType::GEZ; break;
        case BinaryOpType::LE: postprocess = unary::UnaryOpType::LEZ; break;
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
            binary_op = EnumT::MUL;
            postprocess = unary::UnaryOpType::NEZ;
            break;
        case BinaryOpType::LOGICAL_OR:
            process_lhs = unary::UnaryOpType::NEZ;
            process_rhs = unary::UnaryOpType::NEZ;
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::GTZ;
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
            } else if (dtype == DataType::UINT16) {
                return {"sub_int_tile_init();", "sub_uint16_tile"};
            } else {
                return {"sub_binary_tile_init();", "sub_binary_tile"};
            }
        case MUL:
            if (dtype == DataType::UINT16) {
                return {"mul_int_tile_init();", "mul_uint16_tile"};
            } else {
                return {"mul_binary_tile_init();", "mul_binary_tile"};
            }
        case DIV: return {"div_binary_tile_init();", "div_binary_tile"};
        case POWER: return {"power_binary_tile_init();", "power_binary_tile"};
        case RSUB: return {"rsub_binary_tile_init();", "rsub_binary_tile"};
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
            } else {
                return {"binary_bitwise_tile_init();", "bitwise_and_binary_tile"};
            }
        case BITWISE_OR:
            if (dtype == DataType::UINT16) {
                return {"binary_bitwise_tile_init();", "bitwise_or_uint16_binary_tile"};
            } else {
                return {"binary_bitwise_tile_init();", "bitwise_or_binary_tile"};
            }
        case BITWISE_XOR:
            if (dtype == DataType::UINT16) {
                return {"binary_bitwise_tile_init();", "bitwise_xor_uint16_binary_tile"};
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
    } else {
        auto&& [tile_init, tile_fn] = get_sfpu_init_fn(std::get<SfpuBinaryOp>(binary_op), dtype);
        defines["BINARY_SFPU_INIT"] = std::move(tile_init);
        defines["BINARY_SFPU_OP"] = std::move(tile_fn);
        return defines;
    }
}

void add_activation_defines(
    std::map<std::string, std::string>& defines,
    tt::stl::Span<const unary::UnaryWithParam> activations,
    std::string_view operand,
    std::optional<DataType> dtype) {
    defines[fmt::format("PROCESS_{}_ACTIVATIONS(i)", operand)] = std::accumulate(
        activations.begin(),
        activations.end(),
        std::string{},
        [&](std::string&& process, const unary::UnaryWithParam& a) {
            const auto& [op_init, op_func] = unary::utils::get_op_init_and_func(a.op_type, a.params, "i", dtype);
            process += op_init;
            process += op_func;
            unary::utils::update_macro_defines(a.op_type, defines);
            return std::move(process);
        });
}

std::map<std::string, std::string> make_dataflow_defines(const DataType dtype, const DataType b_dtype) {
    std::map<std::string, std::string> defines;
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

uint32_t pack_scalar_runtime_arg(const float scalar, const DataType dtype, const bool is_quant_op) {
    // Always pass the more accurate fp32 when the quantization scale is passed as a scalar
    if ((dtype == DataType::FLOAT32) || is_quant_op) {
        return std::bit_cast<uint32_t>(scalar);
    }
    if (dtype == DataType::INT32) {
        return std::bit_cast<uint32_t>(static_cast<int32_t>(scalar));
    }
    if (dtype == DataType::UINT32) {
        return std::bit_cast<uint32_t>(scalar);
    }
    return pack_two_bfloat16_into_uint32({scalar, scalar});
}

template OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<FpuBinaryOp>);
template OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<SfpuBinaryOp>);

}  // namespace ttnn::operations::binary_ng
