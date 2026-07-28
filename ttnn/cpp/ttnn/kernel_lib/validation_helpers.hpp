// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Kernel-side validation helpers.
//
// INVALID_VALUE_PLACEHOLDER_ARG(name) declares `const uint32_t name` holding a deliberately invalid
// placeholder value. The local is marked [[deprecated]], so any READ of it raises a compiler
// diagnostic, and [[maybe_unused]], so leaving it unread raises none. Use it whenever a variable must
// be declared for structural reasons but carries no valid value in the current context and must never
// be read: an accidental read then surfaces at compile time instead of silently using a garbage value.
//
// By default such a read is only a WARNING, because the kernel build passes
// -Wno-error=deprecated-declarations. A kernel that wants a hard COMPILE ERROR opts in around the
// relevant region with:
//
//     #pragma GCC diagnostic push
//     #pragma GCC diagnostic error "-Wdeprecated-declarations"
//     ...
//     INVALID_VALUE_PLACEHOLDER_ARG(x);
//     INVALID_VALUE_PLACEHOLDER_ARG(y);
//     ... code that must not read variables x or y ...
//     #pragma GCC diagnostic pop
//
// The push/pop matters because a kernel is compiled inside a firmware-wrapper translation unit that
// includes more headers after the kernel source; without the pop the error state would leak into them.
//
// Example use case: one kernel source shared across several compile-time paths whose runtime-arg
// schema is the union of every path's arg names. On a given compiled path only some names are
// meaningful; the rest are declared with this macro, so a wrong-path read is caught at compile time
// instead of silently reading the placeholder value the host passed for that unused name.
#define INVALID_VALUE_PLACEHOLDER_ARG(name)                                                          \
    [[maybe_unused, deprecated("'" #name "' holds an invalid placeholder value here; do not read")]] \
    const uint32_t name = 0
