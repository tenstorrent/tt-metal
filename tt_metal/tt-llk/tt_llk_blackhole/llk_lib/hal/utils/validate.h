// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_assert.h"

namespace hal::detail
{

/**
 * @brief Report a violated runtime requirement through LLK_ASSERT.
 *
 * Not constexpr: LLK_ASSERT may expand to asm, which C++17 forbids in a constexpr body.
 * Cold keeps the assertion out of line, so the valid path falls through without a taken branch.
 *
 * @param message: Description of the violated constraint.
 */
__attribute__((cold, always_inline)) inline void fail_assert([[maybe_unused]] const char* message)
{
    LLK_ASSERT(false, message);
}

/**
 * @brief Fail constant evaluation, or LLK_ASSERT at runtime, when a condition does not hold.
 *
 * A failed constant evaluation shows the message in the compiler's constexpr expansion trace.
 *
 * @param valid: Condition required by the operand or instruction encoding.
 * @param message: Description of the violated constraint.
 * @note With LLK assertions disabled the runtime check is omitted; callers must still satisfy the condition.
 */
inline constexpr __attribute__((always_inline)) void require(const bool valid, const char* message)
{
    // Test validity first: valid constants then fold away even in non-constant contexts,
    // so TTI_INSN still receives an immediate.
    if (!valid)
    {
        if (__builtin_is_constant_evaluated())
        {
            __builtin_trap();
        }
        else
        {
            fail_assert(message);
        }
    }
}

} // namespace hal::detail
