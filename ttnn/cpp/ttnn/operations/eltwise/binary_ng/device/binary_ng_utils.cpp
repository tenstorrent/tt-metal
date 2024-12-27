// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

template <>
struct fmt::formatter<ttnn::operations::binary_ng::Lowercase> : fmt::formatter<std::string_view> {
    auto format(ttnn::operations::binary_ng::Lowercase const& value, fmt::format_context& ctx) const {
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

std::string get_kernel_file_path(KernelName kernel_name) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcast: return fmt::format(dataflow, root, "reader_interleaved_no_bcast.cpp");
        case KernelName::ReaderRowBcast: return fmt::format(dataflow, root, "reader_interleaved_row_bcast.cpp");
        case KernelName::ReaderColBcast: return fmt::format(dataflow, root, "reader_interleaved_col_bcast.cpp");
        case KernelName::ReaderScalarBcast: return fmt::format(dataflow, root, "reader_interleaved_scalar_bcast.cpp");
        case KernelName::WriterNoBcast: return fmt::format(dataflow, root, "writer_interleaved_no_bcast.cpp");
        case KernelName::WriterRowBcast: return fmt::format(dataflow, root, "writer_interleaved_row_bcast.cpp");
        case KernelName::WriterColBcast: return fmt::format(dataflow, root, "writer_interleaved_col_bcast.cpp");
        case KernelName::WriterScalarBcast: return fmt::format(dataflow, root, "writer_interleaved_scalar_bcast.cpp");
        case KernelName::WriterScalar: return fmt::format(dataflow, root, "writer_interleaved_scalar.cpp");
        case KernelName::ComputeNoBcast: return fmt::format(compute, root, "eltwise_binary_no_bcast.cpp");
        case KernelName::ComputeBcast: return fmt::format(compute, root, "eltwise_binary.cpp");
        case KernelName::ComputeScalar: return fmt::format(compute, root, "eltwise_binary_scalar.cpp");
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

std::string get_sfpu_kernel_file_path(KernelName kernel_name) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcast: return fmt::format(dataflow, root, "reader_interleaved_no_bcast.cpp");
        case KernelName::ReaderRowBcast: return fmt::format(dataflow, root, "reader_interleaved_row_bcast.cpp");
        case KernelName::ReaderColBcast: return fmt::format(dataflow, root, "reader_interleaved_col_bcast.cpp");
        case KernelName::ReaderScalarBcast: return fmt::format(dataflow, root, "reader_interleaved_scalar_bcast.cpp");
        case KernelName::WriterNoBcast: return fmt::format(dataflow, root, "writer_interleaved_no_bcast.cpp");
        case KernelName::WriterRowBcast: return fmt::format(dataflow, root, "writer_interleaved_row_bcast.cpp");
        case KernelName::WriterColBcast: return fmt::format(dataflow, root, "writer_interleaved_col_bcast.cpp");
        case KernelName::WriterScalarBcast: return fmt::format(dataflow, root, "writer_interleaved_scalar_bcast.cpp");
        case KernelName::WriterScalar: return fmt::format(dataflow, root, "writer_interleaved_scalar.cpp");
        case KernelName::ComputeNoBcast: return fmt::format(compute, root, "eltwise_binary_sfpu_no_bcast.cpp");
        case KernelName::ComputeBcast: return fmt::format(compute, root, "eltwise_binary_sfpu.cpp");
        case KernelName::ComputeScalar: return fmt::format(compute, root, "eltwise_binary_sfpu_scalar.cpp");
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

constexpr OpConfig::SfpuConfig NezConfig("nez_tile_init", "nez_tile(i)");
constexpr OpConfig::SfpuConfig GtzConfig("gtz_tile_init", "gtz_tile(i)");

OpConfig::OpConfig(BinaryOpType binary_op_type) {
    fpu_binary_op = FpuBinaryOp::SUB;
    switch (binary_op_type) {
        case BinaryOpType::ADD: fpu_binary_op = FpuBinaryOp::ADD; break;
        case BinaryOpType::SUB: break;
        case BinaryOpType::RSUB:
            preprocess_a =
                SfpuConfig("negative_tile_init", "negative_tile(i)", "compute_kernel_api/eltwise_unary/negative.h");
            fpu_binary_op = FpuBinaryOp::ADD;
            break;
        case BinaryOpType::MUL: fpu_binary_op = FpuBinaryOp::MUL; break;
        case BinaryOpType::DIV:
            preprocess_b = SfpuConfig("recip_tile_init", "recip_tile(i)", "compute_kernel_api/eltwise_unary/recip.h");
            fpu_binary_op = FpuBinaryOp::MUL;
            break;
        case BinaryOpType::GT: postprocess = GtzConfig; break;
        case BinaryOpType::LT: postprocess = SfpuConfig("ltz_tile_init", "ltz_tile(i)"); break;
        case BinaryOpType::GTE: postprocess = SfpuConfig("gez_tile_init", "gez_tile(i)"); break;
        case BinaryOpType::LTE: postprocess = SfpuConfig("lez_tile_init", "lez_tile(i)"); break;
        case BinaryOpType::EQ: postprocess = SfpuConfig("eqz_tile_init", "eqz_tile(i)"); break;
        case BinaryOpType::NE: postprocess = NezConfig; break;
        case BinaryOpType::SQUARED_DIFFERENCE: postprocess = SfpuConfig("square_tile_init", "square_tile(i)"); break;
        case BinaryOpType::BIAS_GELU:
            fpu_binary_op = FpuBinaryOp::ADD;
            preprocess_a =
                SfpuConfig("gelu_tile_init<false>", "gelu_tile<false>(i)", "compute_kernel_api/eltwise_unary/gelu.h");
            break;
        case BinaryOpType::LOGICAL_AND:
            fpu_binary_op = FpuBinaryOp::MUL;
            postprocess = NezConfig;
            break;
        case BinaryOpType::LOGICAL_OR:
            fpu_binary_op = FpuBinaryOp::ADD;
            preprocess_a = NezConfig;
            preprocess_b = NezConfig;
            postprocess = GtzConfig;
            break;
        case BinaryOpType::LOGICAL_XOR:
            preprocess_a = NezConfig;
            preprocess_b = NezConfig;
            postprocess = NezConfig;
            break;
        case BinaryOpType::LDEXP:
            fpu_binary_op = FpuBinaryOp::MUL;
            preprocess_b = SfpuConfig("exp2_tile_init", "exp2_tile(i)");
            break;
        case BinaryOpType::LOGADDEXP:
            fpu_binary_op = FpuBinaryOp::ADD;
            preprocess_a =
                SfpuConfig("exp_tile_init<false>", "exp_tile<false>(i)", "compute_kernel_api/eltwise_unary/exp.h");
            preprocess_b = preprocess_a;
            postprocess = SfpuConfig("log_tile_init", "log_tile(i)");
            break;
        case BinaryOpType::LOGADDEXP2:
            fpu_binary_op = FpuBinaryOp::ADD;
            preprocess_a = SfpuConfig("exp2_tile_init", "exp2_tile(i)");
            preprocess_b = preprocess_a;
            postprocess = SfpuConfig("log_with_base_tile_init", "log_with_base_tile(i, 0x3dc5u);");
            break;
        default: __builtin_unreachable();
    }
}

std::map<std::string, std::string> OpConfig::SfpuConfig::as_defines(std::string_view prefix) const {
    if (init.empty()) {
        return {};
    }

    std::map<std::string, std::string> defines;
    defines[fmt::format("{}_INIT", prefix)] = init;
    defines[fmt::format("{}_APPLY(i)", prefix)] = apply;
    defines[fmt::format("{}_INCLUDE", prefix)] = include;
    return defines;
}

std::map<std::string, std::string> OpConfig::as_defines() const {
    std::map<std::string, std::string> defines;
    defines.merge(preprocess_a.as_defines("PREPROCESS_A"));
    defines.merge(preprocess_b.as_defines("PREPROCESS_B"));
    defines.merge(postprocess.as_defines("POSTPROCESS"));

    auto binary_op_str = magic_enum::enum_name(fpu_binary_op);
    defines["BINARY_OP"] = fmt::format("{}_tiles", Lowercase{binary_op_str});
    defines["BINARY_OP_TYPE"] = fmt::format("EltwiseBinaryType::ELW{}", binary_op_str);

    for (const auto& pair : defines) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    return defines;
}

std::map<std::string, std::string> get_defines_fp32(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_a_dtype,
    const std::optional<tt::tt_metal::DataType> input_b_dtype,
    const std::optional<std::vector<ttnn::operations::unary::UnaryWithParam>>& fused_activations,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& input_tensor_a_activation) {
    std::map<std::string, std::string> new_defines;
    std::string op_name = "sub_binary_tile";
    std::string idst1 = "i*2";    // tile index for input A in dst and final output
    std::string idst2 = "i*2+1";  // tile index for input B in dst
    std::string idst = "i";       // tile index for input prescaling

    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::utils::get_defines;

    switch (op_type) {
        case BinaryOpType::ADD:
            if (input_a_dtype == DataType::INT32 && input_b_dtype == DataType::INT32) {
                new_defines.insert({"ADD_INT32_INIT", fmt::format("add_int32_tile_init();")});
                op_name = "add_int32_tile";
            } else {
                op_name = "add_binary_tile";
            }
            break;
        case BinaryOpType::SUB: op_name = "sub_binary_tile"; break;
        case BinaryOpType::MUL: op_name = "mul_binary_tile"; break;
        case BinaryOpType::RSUB: op_name = "rsub_binary_tile"; break;
        case BinaryOpType::POWER: op_name = "power_binary_tile"; break;
        case BinaryOpType::DIV: op_name = "div_binary_tile"; break;
        case BinaryOpType::BITWISE_AND:
            new_defines.insert({"BITWISE_INIT", fmt::format("binary_bitwise_tile_init();")});
            op_name = "and_binary_tile";
            break;
        case BinaryOpType::BITWISE_OR:
            new_defines.insert({"BITWISE_INIT", fmt::format("binary_bitwise_tile_init();")});
            op_name = "or_binary_tile";
            break;
        case BinaryOpType::BITWISE_XOR:
            new_defines.insert({"BITWISE_INIT", fmt::format("binary_bitwise_tile_init();")});
            op_name = "xor_binary_tile";
            break;
        case BinaryOpType::LEFT_SHIFT:
            new_defines.insert({"SHIFT_INIT", fmt::format("binary_shift_tile_init();")});
            op_name = "binary_left_shift_tile";
            break;
        case BinaryOpType::RIGHT_SHIFT:
            new_defines.insert({"SHIFT_INIT", fmt::format("binary_shift_tile_init();")});
            op_name = "binary_right_shift_tile";
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            new_defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0"));
            new_defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0"));
            op_name = "add_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::LOG, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LOGADDEXP2:
            new_defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0"));
            new_defines.merge(get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
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
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::BIAS_GELU:
            op_name = "add_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::GELU, std::vector<float>{0}, "0", idst1));
            break;
        case BinaryOpType::LOGICAL_OR:
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0"));
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0"));
            op_name = "add_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LOGICAL_XOR:
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0"));
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0"));
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst1));
            break;
        // applied on A-B
        case BinaryOpType::GT:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LT:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::LTZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::GTE:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::GEZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::LTE:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::LEZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::EQ:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::EQZ, std::nullopt, "0", idst1));
            break;
        case BinaryOpType::NE:
            op_name = "sub_binary_tile";
            new_defines.merge(get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst1));
            break;
        default:
            tt::log_debug(tt::LogOp, "Undefined op type {}", op_type);
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

    for (const auto& pair : new_defines) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return new_defines;
}

}  // namespace ttnn::operations::binary_ng
