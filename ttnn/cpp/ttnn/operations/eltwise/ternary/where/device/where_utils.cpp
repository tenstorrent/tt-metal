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
                compute_kernel = KernelName::ComputeBcastTTT;
                writer_kernel = KernelName::WriterColBcastTTT;  // Use binary_ng compatible writer
            } else if (broadcast_type == WhereBroadcastType::OUTER_BCAST) {
                reader_kernel = KernelName::ReaderOuterBcastTTT;
                compute_kernel = KernelName::ComputeNoBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (broadcast_type == WhereBroadcastType::ROW_BCAST) {
                reader_kernel = KernelName::ReaderRowBcastTTT;
                compute_kernel = KernelName::ComputeNoBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (broadcast_type == WhereBroadcastType::SCALAR_BCAST) {
                reader_kernel = KernelName::ReaderScalarBcastTTT;
                compute_kernel = KernelName::ComputeBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            } else {
                reader_kernel = KernelName::ReaderNoBcastTTT;
                compute_kernel = KernelName::ComputeNoBcastTTT;
                writer_kernel = KernelName::WriterNoBcast;
            }
            break;

        case WhereVariant::TTS:
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                reader_kernel = KernelName::ReaderColBcastTTS;
                compute_kernel = KernelName::ComputeColBcastTTS;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (broadcast_type == WhereBroadcastType::OUTER_BCAST) {
                reader_kernel = KernelName::ReaderOuterBcastTTS;
                compute_kernel = KernelName::ComputeNoBcastTTS;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (
            broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
            broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) {
                reader_kernel = KernelName::ReaderScalarBcastTTS;
                compute_kernel = KernelName::ComputeScalarBcastTTS;
                writer_kernel = KernelName::WriterNoBcast;
            } else {
                reader_kernel = KernelName::ReaderNoBcastTTS;
                compute_kernel = KernelName::ComputeNoBcastTTS;
                writer_kernel = KernelName::WriterNoBcast;
            }
            break;

        case WhereVariant::TST:
            if (broadcast_type == WhereBroadcastType::COL_BCAST) {
                reader_kernel = KernelName::ReaderColBcastTST;
                compute_kernel = KernelName::ComputeColBcastTST;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (broadcast_type == WhereBroadcastType::OUTER_BCAST) {
                reader_kernel = KernelName::ReaderOuterBcastTST;
                compute_kernel = KernelName::ComputeNoBcastTST;
                writer_kernel = KernelName::WriterNoBcast;
            } else if (
            broadcast_type == WhereBroadcastType::SCALAR_A_BCAST ||
            broadcast_type == WhereBroadcastType::SCALAR_B_BCAST) {
                reader_kernel = KernelName::ReaderScalarBcastTST;
                compute_kernel = KernelName::ComputeScalarBcastTST;
                writer_kernel = KernelName::WriterNoBcast;
            } else {
                reader_kernel = KernelName::ReaderNoBcastTST;
                compute_kernel = KernelName::ComputeNoBcastTST;
                writer_kernel = KernelName::WriterNoBcast;
            }
            break;
            // TODO: Move TSS impl from Unary infra once sharding support is added

        case WhereVariant::TSS: TT_THROW("TSS variant is yet to be moved into Where Device Operation");

        default: TT_THROW("Invalid where variant");
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
        case KernelName::ReaderOuterBcastTTS: return fmt::format(dataflow, root, "tst_tts_reader_outer_bcast.cpp");
        case KernelName::ReaderOuterBcastTST: return fmt::format(dataflow, root, "tst_tts_reader_outer_bcast.cpp");
        case KernelName::ReaderScalarBcastTTS: return fmt::format(dataflow, root, "tst_tts_reader_scalar_bcast.cpp");
        case KernelName::ReaderScalarBcastTST: return fmt::format(dataflow, root, "tst_tts_reader_scalar_bcast.cpp");
        case KernelName::ReaderScalarBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/"
                   "ternary_reader_scalar_ttt.cpp";
        case KernelName::ReaderColBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/"
                   "ternary_reader_colbcast_ttt.cpp";
        case KernelName::ReaderColBcastTTS:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/"
                   "tts_tst_reader_col_bcast.cpp";
        case KernelName::ReaderColBcastTST:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/"
                   "tts_tst_reader_col_bcast.cpp";
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
        case KernelName::ComputeScalarBcastTST: return fmt::format(compute, root, "where_sfpu_scalar_bcast_tst.cpp");
        case KernelName::ComputeScalarBcastTTS: return fmt::format(compute, root, "where_sfpu_scalar_bcast_tts.cpp");
        case KernelName::ComputeBcastTTT:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/compute/"
                   "where_sfpu_col_scalar_bcast_ttt.cpp";
        case KernelName::ComputeColBcastTTS:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/compute/where_sfpu_col_bcast_tts.cpp";
        case KernelName::ComputeColBcastTST:
            return "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/compute/where_sfpu_col_bcast_tst.cpp";
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
    const DataType dtype, const DataType b_dtype, std::optional<DataType> c_dtype) {
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
    // For TTT variant, Add defines for third tensor (false tensor)
    if (c_dtype.has_value()) {
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
    }

    return defines;
}

// 2-tensor broadcast compatibility (used by both TTS and TST variants)
// For TTS: checks predicate vs true tensor (false is scalar)
// For TST: checks predicate vs false tensor (true is scalar)
WhereBroadcastType get_broadcast_type(const ttnn::Shape& predicate_shape, const ttnn::Shape& tensor_shape) {
    // Check for exact match
    if (predicate_shape == tensor_shape) {
        return WhereBroadcastType::NONE;
    }

    bool same_width = (predicate_shape[-1] == tensor_shape[-1]);
    bool same_height = (predicate_shape[-2] == tensor_shape[-2]);

    log_debug(
        tt::LogOp,
        "2-tensor broadcast detection - predicate shape: {}, tensor shape: {}",
        predicate_shape,
        tensor_shape);
    log_debug(tt::LogOp, "same_width: {}, same_height: {}", same_width, same_height);

    // Check for outer broadcast first: if last two dimensions match exactly,
    // it's outer broadcast (broadcasting in dimensions beyond -2)
    if (same_height && same_width) {
        log_debug(tt::LogOp, "Detected OUTER_BCAST for 2-tensor case");
        return WhereBroadcastType::OUTER_BCAST;
    }

    // Multi-dimensional ROW and COL broadcast is not supported for now
    if (!same_height && !same_width) {
        // Get last dimension sizes
        auto pred_w = predicate_shape[-1];
        auto b_w = tensor_shape[-1];

        auto pred_h = predicate_shape[-2];  // height (second-to-last dimension)
        auto b_h = tensor_shape[-2];

        auto max_h = std::max({pred_h, b_h});
        auto max_w = std::max({pred_w, b_w});
        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool b_row_bcast = (b_h == 1 && max_h > 1);
        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool b_col_bcast = (b_w == 1 && max_w > 1);

        if (b_row_bcast && b_col_bcast) {
            return WhereBroadcastType::SCALAR_B_BCAST;
        }

        if (pred_row_bcast && pred_col_bcast) {
            return WhereBroadcastType::SCALAR_A_BCAST;
        }

        return WhereBroadcastType::INVALID_BCAST;
    }

    // Get dimension sizes
    auto pred_w = predicate_shape[-1];
    auto tensor_w = tensor_shape[-1];
    auto pred_h = predicate_shape[-2];
    auto tensor_h = tensor_shape[-2];

    if (!same_width) {
        // Check for column broadcast patterns (width dimension differs, heights same)
        auto max_w = std::max(pred_w, tensor_w);
        bool pred_col_broadcasted = (pred_w == 1 && max_w > 1);
        bool tensor_col_broadcasted = (tensor_w == 1 && max_w > 1);

        // Column broadcast case: at least one tensor is broadcasting in width and heights are same
        if ((pred_col_broadcasted || tensor_col_broadcasted) && same_height) {
            // TTS and TST support column broadcast
            log_debug(tt::LogOp, "2-tensor case detected column broadcast");
            return WhereBroadcastType::COL_BCAST;
        }
    }

    // Check for row broadcast patterns (height dimension differs)
    if (!same_height) {
        // Check for row broadcast patterns: at least one tensor is broadcasting in height and widths are same
        auto max_h = std::max(pred_h, tensor_h);
        bool pred_row_broadcasted = (pred_h == 1 && max_h > 1);
        bool tensor_row_broadcasted = (tensor_h == 1 && max_h > 1);

        log_debug(
            tt::LogOp,
            "2-tensor row broadcast check: pred_h={}, tensor_h={}, same_width={}, pred_row_broadcasted={}, "
            "tensor_row_broadcasted={}",
            pred_h,
            tensor_h,
            same_width,
            pred_row_broadcasted,
            tensor_row_broadcasted);

        // Row broadcast case: at least one tensor is broadcasting in height and widths are same
        if ((pred_row_broadcasted || tensor_row_broadcasted) && same_width) {
            // TTS and TST do not support row broadcast - fallback to legacy
            log_debug(tt::LogOp, "2-tensor row broadcast not supported, falling back to legacy");
            return WhereBroadcastType::INVALID_BCAST;
        }
    }

    // If we reach here, no valid broadcast pattern was found
    return WhereBroadcastType::INVALID_BCAST;
}

WhereBroadcastType get_broadcast_type(
    const ttnn::Shape& predicate_shape, const ttnn::Shape& true_shape, const ttnn::Shape& false_shape) {
    // Check for column broadcast pattern:
    // Examples: (1,1,32,32), (1,1,32,1), (1,1,32,32) or (1,1,32,1), (1,1,32,1), (1,1,32,32)
    // Column broadcast means one or more tensors have last dimension = 1 while at least one has full width
    if ((predicate_shape == true_shape) && (predicate_shape == false_shape)) {
        return WhereBroadcastType::NONE;
    }

    bool same_width = (predicate_shape[-1] == true_shape[-1]) && (predicate_shape[-1] == false_shape[-1]);
    bool same_height = (predicate_shape[-2] == true_shape[-2]) && (predicate_shape[-2] == false_shape[-2]);

    // Check for outer broadcast: same height and width
    if (same_height && same_width) {
        return WhereBroadcastType::OUTER_BCAST;
    }

    // Multi-dimensional mixed ROW and COL broadcast (i.e., cases where both height and width require broadcasting in
    // different tensors)
    if (!same_height && !same_width) {
        // Get last dimension sizes
        auto pred_w = predicate_shape[-1];
        auto true_w = true_shape[-1];
        auto false_w = false_shape[-1];

        auto pred_h = predicate_shape[-2];
        auto true_h = true_shape[-2];
        auto false_h = false_shape[-2];

        auto max_h = std::max({pred_h, true_h, false_h});
        auto max_w = std::max({pred_w, true_w, false_w});

        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool true_row_bcast = (true_h == 1 && max_h > 1);
        bool false_row_bcast = (false_h == 1 && max_h > 1);

        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool true_col_bcast = (true_w == 1 && max_w > 1);
        bool false_col_bcast = (false_w == 1 && max_w > 1);

        bool is_pred_scalar = (pred_row_bcast && pred_col_bcast);
        bool is_true_scalar = (true_row_bcast && true_col_bcast);
        bool is_false_scalar = (false_row_bcast && false_col_bcast);

        bool is_pred_non_bcast = (pred_h == max_h && pred_w == max_w);
        bool is_true_non_bcast = (true_h == max_h && true_w == max_w);
        bool is_false_non_bcast = (false_h == max_h && false_w == max_w);

        if ((is_pred_scalar || is_pred_non_bcast) && (is_true_scalar || is_true_non_bcast) &&
            (is_false_scalar || is_false_non_bcast)) {
            return WhereBroadcastType::SCALAR_BCAST;
        }

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

}  // namespace ttnn::operations::ternary
