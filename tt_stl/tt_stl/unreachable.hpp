// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Try to surface feature-test macros as early as possible
#if defined(__has_include)
#if __has_include(<version>)
#include <version>
#endif
#endif
#include <utility>  // ensures __cpp_lib_unreachable is visible on some libcs

// We may use abort() in debug-fallbacks
#include <cstdlib>

namespace ttsl {

// Prefer the C++23 library facility when available
#if defined(__cpp_lib_unreachable)
using std::unreachable;

#else

[[noreturn]] inline void unreachable() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(false);
#else
    // Portable fallback: still ensure no return
    std::abort();
#endif
}

#endif  // !defined(__cpp_lib_unreachable)

}  // namespace ttsl
