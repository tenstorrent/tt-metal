// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_assert.h"

namespace hal::cfg::detail
{

/**
 * @brief Report an invalid descriptor-table index and trap without returning.
 *
 * LLK_ASSERT supplies the diagnostic when enabled. The unconditional trap also
 * prevents invalid indices from being accepted during constant evaluation.
 *
 * @tparam T: Result type of the surrounding lookup, allowing use in its conditional expression.
 */
template <typename T>
[[noreturn]] inline T invalid_index()
{
    LLK_ASSERT(false, "CFG descriptor index out of range");
    __builtin_trap();
}

} // namespace hal::cfg::detail
