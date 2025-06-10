// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_utils.hpp"
#include <tt-metalium/assert.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

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
        case KernelName::ReaderNoBcastTST: return fmt::format(dataflow, root, "ternary_reader_nobcast_tst.cpp");
        case KernelName::ReaderNoBcastTTS: return fmt::format(dataflow, root, "ternary_reader_nobcast_tts.cpp");
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

}  // namespace ttnn::operations::ternary
