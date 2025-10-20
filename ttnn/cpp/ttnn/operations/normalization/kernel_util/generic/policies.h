// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace norm::kernel_util::generic::policies {

/**
 * @brief Policy to specify if an operation should wait
 * for the tiles after it pushes them.
 */
enum class CBWaitPolicy { Wait, NoWait };

}  // namespace norm::kernel_util::generic::policies
