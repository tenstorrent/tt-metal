// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bcast_to_device_operation.hpp"

#include <optional>
#include <string>

namespace ttnn::operations::experimental::broadcast_to {

enum class KernelName {
    ReaderNoBcast,
    ReaderRowBcast,
    ReaderColBcast,
    ReaderScalarBcast,
    WriterNoBcast,
    WriterRowBcast,
    WriterColBcast,
    WriterScalarBcast,
    ComputeNoBcast,
    ComputeRowBcast,
    ComputeColBcast,
    ComputeScalarBcast,
};

struct BcastToKernelConfig {
    BcastToKernelConfig(SubtileBroadcastType subtile_broadcast_type);
    KernelName reader_kernel;
    KernelName writer_kernel;
    KernelName compute_kernel;
};

std::string get_kernel_file_path(KernelName kernel_name);
struct Lowercase {
    std::string_view view;
};

}  // namespace ttnn::operations::experimental::broadcast_to
