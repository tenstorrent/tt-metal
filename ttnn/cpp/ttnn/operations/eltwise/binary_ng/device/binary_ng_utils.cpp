// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"

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

constexpr OpConfig::SfpuConfig NezConfig("nez_tile_init", "nez_tile(i)");
constexpr OpConfig::SfpuConfig GtzConfig("gtz_tile_init", "gtz_tile(i)");

OpConfig::OpConfig(BinaryOpType binary_op_type) {
    fpu_binary_op = FpuBinaryOp::SUB;
    switch (binary_op_type) {
        case BinaryOpType::ADD: fpu_binary_op = FpuBinaryOp::ADD; break;
        case BinaryOpType::SUB: break;
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

    return defines;
}

}  // namespace ttnn::operations::binary_ng
