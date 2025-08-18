// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_to_utils.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

template <>
struct fmt::formatter<ttnn::operations::experimental::broadcast_to::Lowercase> : fmt::formatter<std::string_view> {
    auto format(const ttnn::operations::experimental::broadcast_to::Lowercase& value, fmt::format_context& ctx) const {
        auto out = ctx.out();
        for (char c : value.view) {
            *out++ = std::tolower(static_cast<unsigned char>(c));
        }
        return out;
    }
};

namespace ttnn::operations::experimental::broadcast_to {

BcastToKernelConfig::BcastToKernelConfig(SubtileBroadcastType subtile_broadcast_type) {
    switch (subtile_broadcast_type) {
        case SubtileBroadcastType::NONE:
            reader_kernel = KernelName::ReaderNoBcast;
            writer_kernel = KernelName::WriterNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            break;

        case SubtileBroadcastType::SCALAR:
            reader_kernel = KernelName::ReaderScalarBcast;
            writer_kernel = KernelName::WriterScalarBcast;
            compute_kernel = KernelName::ComputeScalarBcast;
            break;

        case SubtileBroadcastType::ROW:
            reader_kernel = KernelName::ReaderRowBcast;
            writer_kernel = KernelName::WriterRowBcast;
            compute_kernel = KernelName::ComputeRowBcast;
            break;

        case SubtileBroadcastType::COL:
            reader_kernel = KernelName::ReaderColBcast;
            writer_kernel = KernelName::WriterColBcast;
            compute_kernel = KernelName::ComputeColBcast;
            break;
    }
}

std::string get_kernel_file_path(KernelName kernel_name) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/experimental/bcast_to/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcast: return fmt::format(dataflow, root, "reader_interleaved_no_bcast_to.cpp");
        case KernelName::ReaderRowBcast: return fmt::format(dataflow, root, "reader_interleaved_row_bcast_to.cpp");
        case KernelName::ReaderColBcast: return fmt::format(dataflow, root, "reader_interleaved_col_bcast_to.cpp");
        case KernelName::ReaderScalarBcast:
            return fmt::format(dataflow, root, "reader_interleaved_scalar_bcast_to.cpp");
        case KernelName::WriterNoBcast: return fmt::format(dataflow, root, "writer_interleaved_no_bcast_to.cpp");
        case KernelName::WriterRowBcast: return fmt::format(dataflow, root, "writer_interleaved_row_bcast_to.cpp");
        case KernelName::WriterColBcast: return fmt::format(dataflow, root, "writer_interleaved_col_bcast_to.cpp");
        case KernelName::WriterScalarBcast:
            return fmt::format(dataflow, root, "writer_interleaved_scalar_bcast_to.cpp");
        case KernelName::ComputeNoBcast: return fmt::format(compute, root, "compute_interleaved_no_bcast_to.cpp");
        case KernelName::ComputeRowBcast: return fmt::format(compute, root, "compute_interleaved_row_bcast_to.cpp");
        case KernelName::ComputeColBcast: return fmt::format(compute, root, "compute_interleaved_col_bcast_to.cpp");
        case KernelName::ComputeScalarBcast:
            return fmt::format(compute, root, "compute_interleaved_scalar_bcast_to.cpp");
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

}  // namespace ttnn::operations::experimental::broadcast_to
