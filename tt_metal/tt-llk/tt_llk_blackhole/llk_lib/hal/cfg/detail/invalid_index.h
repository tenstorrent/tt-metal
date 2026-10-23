// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_assert.h"

namespace hal::cfg::detail
{

template <typename T>
[[noreturn]] inline T invalid_index()
{
    LLK_ASSERT(false, "CFG descriptor index out of range");
    __builtin_trap();
}

} // namespace hal::cfg::detail
