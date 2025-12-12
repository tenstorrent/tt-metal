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
 * @brief Control whether to pop an input CB after
 * processing it in a function and how to pop it
 */
struct PopInputWithRemainderPolicy {
    static constexpr bool pop = true;
    static constexpr bool pop_remainder = true;
};

struct PopInputWithoutRemainderPolicy {
    static constexpr bool pop = true;
    static constexpr bool pop_remainder = false;
};

struct NoPopInputPolicy {
    static constexpr bool pop = false;
    static constexpr bool pop_remainder = false;
};

}  // namespace norm::kernel_util::compute::policies
