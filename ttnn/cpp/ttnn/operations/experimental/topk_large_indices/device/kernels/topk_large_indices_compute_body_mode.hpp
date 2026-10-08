// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::topk_large_indices::program {

// Shared host/device encoding for the compile-time row-reduction body.
enum class ComputeBodyMode : uint32_t {
    FusedEndToEnd = 1,
    FusedSegmented = 2,
    // K 1024 chunks folded per column, each of the 16 columns keeping its own top 64: k <= 64 only.
    ColumnSegmented = 3,
};

// The fused index stamp carries the chunk id in five bits, so a fused row or segment holds at most 32 chunks.
constexpr uint32_t max_fused_chunks = 32;

}  // namespace ttnn::operations::experimental::topk_large_indices::program
