// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_utils.hpp"
#include <tt-metalium/assert.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

namespace ttnn::operations::ternary {

WhereKernelConfig::WhereKernelConfig(WhereVariant where_variant, WhereBroadcastType broadcast_type) {
    switch (where_variant) {
        case WhereVariant::TTT:
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                reader_kernel = KernelName::ReaderColBcastTTT;
                compute_kernel = KernelName::ComputeColBcastTTT;
                writer_kernel = KernelName::WriterColBcastTTT;  // Use binary_ng compatible writer
            } else if (broadcast_type == WhereBroadcastType::OUTER_BCAST) {
                reader_kernel = KernelName::ReaderOuterBcastTTT;
                compute_kernel = KernelName::ComputeNoBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (broadcast_type == WhereBroadcastType::ROW_BCAST) {
                reader_kernel = KernelName::ReaderRowBcastTTT;
                compute_kernel = KernelName::ComputeNoBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            } else {
                reader_kernel = KernelName::ReaderNoBcastTTT;
                compute_kernel = KernelName::ComputeNoBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            }
            break;

        case WhereVariant::TTS:
            if (broadcast_type == WhereBroadcastType::OUTER_BCAST) {
                reader_kernel = KernelName::ReaderOuterBcastTTS;
                compute_kernel = KernelName::ComputeNoBcastTTS;
                writer_kernel = KernelName::WriterNoBcast;
            } else {
                reader_kernel = KernelName::ReaderNoBcastTTS;
                compute_kernel = KernelName::ComputeNoBcastTTS;
                writer_kernel = KernelName::WriterNoBcast;
            }
            break;

        case WhereVariant::TST:
            if (broadcast_type == WhereBroadcastType::OUTER_BCAST) {
                reader_kernel = KernelName::ReaderOuterBcastTST;
                compute_kernel = KernelName::ComputeNoBcastTST;
                writer_kernel = KernelName::WriterNoBcast;
            } else {
                reader_kernel = KernelName::ReaderNoBcastTST;
                compute_kernel = KernelName::ComputeNoBcastTST;
                writer_kernel = KernelName::WriterNoBcast;
            }
            break;

        case WhereVariant::TSS:
            reader_kernel = KernelName::ReaderNoBcastTSS;
            compute_kernel = KernelName::ComputeNoBcastTSS;
            writer_kernel = KernelName::WriterNoBcast;
            break;
    }
}

std::string get_kernel_file_path(KernelName kernel_name) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcastTTT: return fmt::format(dataflow, root, "ternary_reader_nobcast_ttt.cpp");
        case KernelName::ReaderOuterBcastTTT: return fmt::format(dataflow, root, "ternary_reader_outerbcast_ttt.cpp");
        case KernelName::ReaderNoBcastTST: return fmt::format(dataflow, root, "ternary_reader_nobcast_tst_tts.cpp");
        case KernelName::ReaderNoBcastTTS: return fmt::format(dataflow, root, "ternary_reader_nobcast_tst_tts.cpp");
        case KernelName::ReaderOuterBcastTTS:
            return fmt::format(dataflow, root, "ternary_reader_outerbcast_tst_tts.cpp");
        case KernelName::ReaderOuterBcastTST:
            return fmt::format(dataflow, root, "ternary_reader_outerbcast_tst_tts.cpp");
        case KernelName::ReaderNoBcastTSS: return fmt::format(dataflow, root, "ternary_reader_nobcast_tss.cpp");
        case KernelName::ReaderColBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/"
                   "ternary_reader_colbcast_ttt.cpp";
        case KernelName::ReaderRowBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/"
                   "ternary_reader_rowbcast_ttt.cpp";

        case KernelName::WriterNoBcast:
            return "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                   "writer_unary_interleaved_start_id.cpp";
        case KernelName::WriterColBcastTTT:
            // Use unary writer (simple and works with 3 args: dst_addr, num_tiles, start_id)
            return "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                   "writer_unary_interleaved_start_id.cpp";

        case KernelName::ComputeNoBcastTTT: return fmt::format(compute, root, "where_sfpu_no_bcast_ttt.cpp");
        case KernelName::ComputeNoBcastTST: return fmt::format(compute, root, "where_sfpu_no_bcast_tst.cpp");
        case KernelName::ComputeNoBcastTTS: return fmt::format(compute, root, "where_sfpu_no_bcast_tts.cpp");
        case KernelName::ComputeNoBcastTSS: return fmt::format(compute, root, "where_sfpu_no_bcast_tss.cpp");
        case KernelName::ComputeColBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/compute/where_sfpu_col_bcast_ttt.cpp";
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

uint32_t pack_scalar_runtime_arg(const float scalar, const DataType dtype) {
    if (dtype == DataType::INT32) {
        return std::bit_cast<uint32_t>(static_cast<int32_t>(scalar));
    }
    return std::bit_cast<uint32_t>(scalar);
}

std::map<std::string, std::string> make_dataflow_defines(
    const DataType dtype, const DataType b_dtype, const DataType c_dtype) {
    std::map<std::string, std::string> defines;
    // Exact copy of binary_ng make_dataflow_defines for compatibility
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

    // Add defines for second tensor (true tensor)
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

    // Add defines for third tensor (false tensor)
    if (c_dtype == DataType::FLOAT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element<float>";
        defines["FILL_WITH_VALUE_FLOAT_C"] = "fill_with_val<1024, float>";
    } else if (c_dtype == DataType::INT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element<int32_t>";
        defines["FILL_WITH_VALUE_C"] = "fill_with_val<1024, int32_t>";
    } else if (c_dtype == DataType::UINT32) {
        defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column";
        defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element<uint32_t>";
        defines["FILL_WITH_VALUE_C"] = "fill_with_val<1024, uint32_t>";
    } else {
        defines["FILL_TILE_WITH_FIRST_COLUMN_C"] = "fill_tile_with_first_column_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ROW_C"] = "fill_tile_with_first_row_bfloat16";
        defines["FILL_TILE_WITH_FIRST_ELEMENT_C"] = "fill_tile_with_first_element_bfloat16";
        defines["FILL_WITH_VALUE_C"] = "fill_with_val_bfloat16";
    }

    return defines;
}

WhereBroadcastType get_broadcast_type(
    const ttnn::Shape& predicate_shape, const ttnn::Shape& true_shape, const ttnn::Shape& false_shape) {
    // Check for column broadcast pattern:
    // Examples: (1,1,32,32), (1,1,32,1), (1,1,32,32) or (1,1,32,1), (1,1,32,1), (1,1,32,32)
    // Column broadcast means one or more tensors have last dimension = 1 while at least one has full width
    if ((predicate_shape == true_shape) && (predicate_shape == false_shape)) {
        return WhereBroadcastType::NONE;
    }

    if ((predicate_shape[-1] == true_shape[-1]) && (predicate_shape[-1] == false_shape[-1]) &&
        (predicate_shape[-2] == true_shape[-2]) && (predicate_shape[-2] == false_shape[-2])) {
        return WhereBroadcastType::OUTER_BCAST;
    }

    bool same_width = (predicate_shape[-1] == true_shape[-1]) && (predicate_shape[-1] == false_shape[-1]);
    bool same_height = (predicate_shape[-2] == true_shape[-2]) && (predicate_shape[-2] == false_shape[-2]);

    // Multi-dimensional ROW and COL broadcast is not supported for now
    if (!same_height && !same_width) {
        return WhereBroadcastType::INVALID_BCAST;
    }

    // Get last dimension sizes
    auto pred_w = predicate_shape[-1];
    auto true_w = true_shape[-1];
    auto false_w = false_shape[-1];

    auto pred_h = predicate_shape[-2];  // height (second-to-last dimension)
    auto true_h = true_shape[-2];
    auto false_h = false_shape[-2];

    if (!same_height) {
        // Check for row broadcast patterns first (height dimension differs)
        auto max_h = std::max({pred_h, true_h, false_h});
        bool pred_row_broadcasted = (pred_h == 1 && max_h > 1);
        bool true_row_broadcasted = (true_h == 1 && max_h > 1);
        bool false_row_broadcasted = (false_h == 1 && max_h > 1);

        // Row broadcast case: at least one tensor is broadcasting in height and widths are same
        if ((pred_row_broadcasted || true_row_broadcasted || false_row_broadcasted) &&
            (pred_w == true_w && pred_w == false_w)) {
            return WhereBroadcastType::ROW_BCAST;
        }
    }

    if (!same_width) {
        // Check for column broadcast patterns (width dimension differs, heights same)
        auto max_w = std::max({pred_w, true_w, false_w});
        bool pred_col_broadcasted = (pred_w == 1 && max_w > 1);
        bool true_col_broadcasted = (true_w == 1 && max_w > 1);
        bool false_col_broadcasted = (false_w == 1 && max_w > 1);

        // Column broadcast case: at least one tensor is broadcasting in width and heights are same
        if ((pred_col_broadcasted || true_col_broadcasted || false_col_broadcasted) &&
            (pred_h == true_h && pred_h == false_h)) {
            return WhereBroadcastType::COL_BCAST;
        }
    }

    return WhereBroadcastType::INVALID_BCAST;
}

WhereBroadcastType get_broadcast_type(const ttnn::Shape& predicate_shape, const ttnn::Shape& b_shape) {
    if ((predicate_shape == b_shape)) {
        return WhereBroadcastType::NONE;
    }

    if ((predicate_shape[-1] == b_shape[-1]) && (predicate_shape[-2] == b_shape[-2])) {
        return WhereBroadcastType::OUTER_BCAST;
    }

    return WhereBroadcastType::INVALID_BCAST;
}

}  // namespace ttnn::operations::ternary
