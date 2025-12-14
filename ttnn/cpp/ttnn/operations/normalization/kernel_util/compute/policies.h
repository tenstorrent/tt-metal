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
 * processing it in a function
 */
enum class PopInputPolicy { POP, NO_POP };

}  // namespace norm::kernel_util::compute::policies
