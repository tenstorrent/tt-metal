// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::reduction {

/**
 * Operation types for the tilize_untilize template operation.
 *
 * This enum is shared between:
 * - Program factory (host code) - to pass OpType as compile-time arg
 * - Compute kernel (device code) - to select Phase 2 operation via if constexpr
 * - Reader kernel (device code) - to conditionally generate auxiliary CBs
 *
 * The enum value is passed as a compile-time argument to both reader and compute kernels,
 * enabling compile-time branching with if constexpr for zero runtime overhead.
 */
enum class OpType : uint32_t {
    IDENTITY = 0,  // Pass-through: tilize -> untilize (no Phase 2 operation)
    // Future operations:
    // REDUCE_W_SUM = 1,   // Sum reduction along width
    // REDUCE_W_MAX = 2,   // Max reduction along width
    // REDUCE_W_AVG = 3,   // Average reduction along width
    // RELU = 4,           // ReLU activation
};

}  // namespace ttnn::operations::reduction
