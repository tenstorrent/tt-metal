// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::topk_large_indices::program {

// Shared host/device encoding for the compile-time row-reduction body.
enum class ComputeBodyMode : uint32_t {
    Classic = 0,
    FusedEndToEnd = 1,
    FusedSegmented = 2,
};

}  // namespace ttnn::operations::experimental::topk_large_indices::program
