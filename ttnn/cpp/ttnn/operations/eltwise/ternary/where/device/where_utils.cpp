// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_utils.hpp"
#include <tt-metalium/assert.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

namespace ttnn::operations::ternary {

WhereKernelConfig::WhereKernelConfig(WhereVariant where_variant) {
    switch (where_variant) {
        case WhereVariant::TTT:
            reader_kernel = KernelName::ReaderNoBcastTTT;
            compute_kernel = KernelName::ComputeNoBcastTTT;
            writer_kernel = KernelName::WriterNoBcastTTT;
            break;

        case WhereVariant::TTS:
            reader_kernel = KernelName::ReaderNoBcastTTS;
            compute_kernel = KernelName::ComputeNoBcastTTS;
            writer_kernel = KernelName::WriterNoBcastTTS;
            break;

        case WhereVariant::TST:
            reader_kernel = KernelName::ReaderNoBcastTST;
            compute_kernel = KernelName::ComputeNoBcastTST;
            writer_kernel = KernelName::WriterNoBcastTST;
            break;

        case WhereVariant::TSS:
            reader_kernel = KernelName::ReaderNoBcastTSS;
            compute_kernel = KernelName::ComputeNoBcastTSS;
            writer_kernel = KernelName::WriterNoBcastTSS;
            break;
    }
}

std::string get_kernel_file_path(KernelName kernel_name) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcastTTT: return fmt::format(dataflow, root, "ternary_reader_nobcast_ttt.cpp");
        case KernelName::ReaderNoBcastTST: return fmt::format(dataflow, root, "ternary_reader_nobcast_tst_tts.cpp");
        case KernelName::ReaderNoBcastTTS: return fmt::format(dataflow, root, "ternary_reader_nobcast_tst_tts.cpp");
        case KernelName::ReaderNoBcastTSS: return fmt::format(dataflow, root, "ternary_reader_nobcast_tss.cpp");

        case KernelName::WriterNoBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                   "writer_unary_interleaved_start_id.cpp";
        case KernelName::WriterNoBcastTST:
            return "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                   "writer_unary_interleaved_start_id.cpp";
        case KernelName::WriterNoBcastTTS:
            return "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                   "writer_unary_interleaved_start_id.cpp";
        case KernelName::WriterNoBcastTSS:
            return "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                   "writer_unary_interleaved_start_id.cpp";

        case KernelName::ComputeNoBcastTTT: return fmt::format(compute, root, "where_sfpu_no_bcast_ttt.cpp");
        case KernelName::ComputeNoBcastTST: return fmt::format(compute, root, "where_sfpu_no_bcast_tst.cpp");
        case KernelName::ComputeNoBcastTTS: return fmt::format(compute, root, "where_sfpu_no_bcast_tts.cpp");
        case KernelName::ComputeNoBcastTSS: return fmt::format(compute, root, "where_sfpu_no_bcast_tss.cpp");
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

uint32_t pack_scalar_runtime_arg(const float scalar, const DataType dtype) {
    if (dtype == DataType::INT32) {
        return std::bit_cast<uint32_t>(static_cast<int32_t>(scalar));
    }
    return std::bit_cast<uint32_t>(scalar);
}

}  // namespace ttnn::operations::ternary
