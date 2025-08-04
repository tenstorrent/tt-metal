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
            } else {
                reader_kernel = KernelName::ReaderNoBcastTTT;
            }
            compute_kernel = KernelName::ComputeNoBcastTTT;  // Same compute kernel for both
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
        case KernelName::ReaderColBcastTTT: return fmt::format(dataflow, root, "ternary_reader_colbcast_ttt.cpp");

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

std::map<std::string, std::string> make_ternary_dataflow_defines(
    const DataType predicate_dtype, const DataType value_true_dtype, const DataType value_false_dtype) {
    auto get_fill_function = [](DataType dtype) -> std::string {
        return (dtype == DataType::FLOAT32 || dtype == DataType::INT32 || dtype == DataType::UINT32)
                   ? "fill_tile_with_first_column"
                   : "fill_tile_with_first_column_bfloat16";
    };

    return {
        {"FILL_TILE_WITH_FIRST_COLUMN", get_fill_function(predicate_dtype)},
        {"FILL_TILE_WITH_FIRST_COLUMN_B", get_fill_function(value_true_dtype)},
        {"FILL_TILE_WITH_FIRST_COLUMN_C", get_fill_function(value_false_dtype)}};
}

// Check if two tensors are broadcastable for general validation (permissive)
bool are_tensors_broadcastable_general(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.logical_shape();
    const auto& b_shape = b.logical_shape();

    // Same shape is always valid
    if (a_shape == b_shape) {
        return true;
    }

    // Check for general broadcasting compatibility (similar to PyTorch/NumPy)
    // This includes height broadcasting, width broadcasting, and scalar broadcasting
    if (a_shape.rank() != b_shape.rank() || a_shape.rank() < 2) {
        return false;  // For now, require same rank and at least 2D
    }

    auto rank = a_shape.rank();
    auto a_w = a_shape[-1];
    auto a_h = a_shape[-2];
    auto b_w = b_shape[-1];
    auto b_h = b_shape[-2];

    // Allow height broadcasting: one height can be 1, other > 1, or both same
    bool height_compatible = (a_h == b_h) || (a_h == 1) || (b_h == 1);

    // Allow width broadcasting: one width can be 1, other > 1, or both same
    bool width_compatible = (a_w == b_w) || (a_w == 1) || (b_w == 1);

    if (!height_compatible || !width_compatible) {
        return false;
    }

    // Check that all other dimensions are broadcastable
    for (int i = 0; i < rank - 2; ++i) {
        if (a_shape[i] != b_shape[i] && a_shape[i] != 1 && b_shape[i] != 1) {
            return false;
        }
    }

    return true;
}

// Check if two tensors are broadcastable for LLK path (restrictive - only column broadcast)
bool are_tensors_broadcastable_llk(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.logical_shape();
    const auto& b_shape = b.logical_shape();

    // Same shape is always valid
    if (a_shape == b_shape) {
        return true;
    }

    // Check for column broadcast: tensors have same rank and all dimensions match except width
    if (a_shape.rank() != b_shape.rank() || a_shape.rank() < 2) {
        return false;
    }

    auto rank = a_shape.rank();
    auto a_w = a_shape[-1];
    auto a_h = a_shape[-2];
    auto b_w = b_shape[-1];
    auto b_h = b_shape[-2];

    // Heights must match exactly (no height broadcasting supported in LLK kernel)
    if (a_h != b_h) {
        return false;
    }

    // For column broadcast: one width must be 1, other > 1 (pure column broadcast)
    // OR both widths must be same (same shape)
    bool same_width = (a_w == b_w);
    bool column_broadcast = (a_w == 1 && b_w > 1) || (a_w > 1 && b_w == 1);

    if (!(same_width || column_broadcast)) {
        return false;
    }

    // All other dimensions must match exactly (no broadcasting on other dims)
    for (int i = 0; i < rank - 2; ++i) {
        if (a_shape[i] != b_shape[i]) {
            return false;
        }
    }

    return true;
}

// Comprehensive broadcast type detection for WHERE TTT operation
WhereBroadcastInfo get_where_broadcast_info(
    const Tensor& predicate, const Tensor& value_true, const Tensor& value_false) {
    WhereBroadcastInfo info;

    const auto& pred_shape = predicate.logical_shape();
    const auto& true_shape = value_true.logical_shape();
    const auto& false_shape = value_false.logical_shape();

    // Check if all shapes are the same (no broadcast needed)
    if (pred_shape == true_shape && pred_shape == false_shape) {
        info.type = WhereBroadcastType::NONE;
        return info;
    }

    // For now, only support column broadcast where all tensors have same rank
    if (pred_shape.rank() != true_shape.rank() || pred_shape.rank() != false_shape.rank()) {
        // Return NONE for unsupported cases, but mark that LLK should not be used
        info.type = WhereBroadcastType::NONE;
        info.predicate_broadcast = false;
        info.value_true_broadcast = false;
        info.value_false_broadcast = false;
        return info;
    }

    // Check for column broadcast: one of the tensors has width 1, others have same width > 1
    auto rank = pred_shape.rank();
    if (rank >= 2) {
        auto pred_w = pred_shape[-1];
        auto pred_h = pred_shape[-2];
        auto true_w = true_shape[-1];
        auto true_h = true_shape[-2];
        auto false_w = false_shape[-1];
        auto false_h = false_shape[-2];

        // Check if all heights match
        if (pred_h == true_h && pred_h == false_h) {
            // Determine which tensors are broadcasted
            info.predicate_broadcast = (pred_w == 1);
            info.value_true_broadcast = (true_w == 1);
            info.value_false_broadcast = (false_w == 1);

            // Count how many tensors have width=1
            int broadcast_count = info.predicate_broadcast + info.value_true_broadcast + info.value_false_broadcast;

            // Valid column broadcast: exactly one tensor has width=1, others have same width>1
            if (broadcast_count == 1) {
                // Get the target width (from non-broadcasted tensors)
                uint32_t target_w = info.predicate_broadcast ? true_w : pred_w;

                // Verify the pattern is valid
                if (target_w > 1 && (!info.predicate_broadcast || pred_w == 1) &&
                    (!info.value_true_broadcast || true_w == 1) && (!info.value_false_broadcast || false_w == 1) &&
                    (info.predicate_broadcast || pred_w == target_w) &&
                    (info.value_true_broadcast || true_w == target_w) &&
                    (info.value_false_broadcast || false_w == target_w)) {
                    // Check that all other dimensions match
                    bool other_dims_match = true;
                    for (int i = 0; i < rank - 2; ++i) {
                        if (pred_shape[i] != true_shape[i] || pred_shape[i] != false_shape[i]) {
                            other_dims_match = false;
                            break;
                        }
                    }

                    if (other_dims_match) {
                        info.type = WhereBroadcastType::COL_BCAST;
                        return info;
                    }
                }
            }
        }
    }

    // Return NONE for unsupported cases (will fall back to legacy)
    // Mark as unsupported by setting a broadcast flag to indicate this is not true same shape
    info.type = WhereBroadcastType::NONE;
    info.predicate_broadcast = true;  // Use this as a flag to indicate "unsupported broadcast"
    return info;
}

// Check if LLK can be used with the given broadcast pattern
bool can_use_llk_with_broadcast(const WhereBroadcastInfo& broadcast_info) {
    // Only allow LLK for:
    // 1. True same shape (NONE with no broadcast flags set)
    // 2. Valid column broadcast (COL_BCAST)
    if (broadcast_info.type == WhereBroadcastType::COL_BCAST) {
        return true;
    }

    if (broadcast_info.type == WhereBroadcastType::NONE) {
        // Only allow NONE if no broadcast flags are set (true same shape)
        return !broadcast_info.predicate_broadcast && !broadcast_info.value_true_broadcast &&
               !broadcast_info.value_false_broadcast;
    }

    return false;
}

}  // namespace ttnn::operations::ternary
