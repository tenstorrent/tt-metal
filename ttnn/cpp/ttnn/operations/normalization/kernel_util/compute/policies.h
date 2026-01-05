// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file policies.h
 * @brief Policies to control various functionality in kernels
 */

#pragma once

namespace norm::kernel_util::compute::policies {

/**
 * @brief Control whether to cb_wait_front at the end
 * of a function
 */
enum class WaitAtEndPolicy { WAIT, NO_WAIT };

/**
 * @brief Policies to control how handle data fed
 * into input CBs for a compute operation
 */
struct PartialBlockWithPopPolicy {
    static constexpr bool pop = true;
    static constexpr bool sync_full_block = false;
};
struct PartialBlockWithoutPopPolicy {
    static constexpr bool pop = false;
    static constexpr bool sync_full_block = false;
};

struct FullBlockWithPopPolicy {
    static constexpr bool pop = true;
    static constexpr bool sync_full_block = true;
};

struct FullBlockWithoutPopPolicy {
    static constexpr bool pop = false;
    static constexpr bool sync_full_block = true;
};

}  // namespace norm::kernel_util::compute::policies
