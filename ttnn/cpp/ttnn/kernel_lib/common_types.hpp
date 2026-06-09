// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file common_types.hpp
 * @brief Common types shared between kernel library helpers (reduce, binary_op, etc.)
 *
 * This file contains type definitions that are used by multiple kernel helper libraries
 * to avoid duplication and ensure consistency.
 */

namespace compute_kernel_lib {

/**
 * @brief Tag type indicating no accumulation (zero overhead)
 *
 * When this type is passed to reduce() or binary_op(), all accumulation code is
 * eliminated at compile-time via `if constexpr`.
 */
struct NoAccumulation {};

/**
 * @brief Default no-op functor for post operation parameter
 *
 * When no custom post operation is needed, this empty functor is used.
 * It compiles away completely due to inlining.
 */
struct NoOp {
    ALWI void operator()(uint32_t = 0) const {}
};

}  // namespace compute_kernel_lib
