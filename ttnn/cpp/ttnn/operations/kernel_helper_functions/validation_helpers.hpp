// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Kernel-side validation helpers.
//
// INVALID_VALUE_PLACEHOLDER names a value that must exist in a scope for structural reasons but
// carries no valid value there and must never be used. Declare such placeholders as
// `extern INVALID_VALUE_PLACEHOLDER`: that is a declaration, not a definition, so no object is
// constructed and no storage is reserved. Every special member is deleted, so an
// INVALID_VALUE_PLACEHOLDER can never be constructed, copied, or assigned; it exists only to occupy a
// name.
//
// Any use is a hard error. If "extern INVALID_VALUE_PLACEHOLDER name;" is present in a scope, then
//   * reading the value (e.g. `uint32_t x = name;`) fails to compile, because the type has no
//     conversion and no operators;
//   * anything that slips past that (e.g. `&name`) fails to link, because the extern declaration
//     has no definition anywhere.
// Leaving a placeholder unread is fine: an unused extern declaration reserves nothing and draws no
// -Wunused-variable diagnostic.
//
// Example use case: one kernel source shared across several compile-time paths whose runtime-arg
// schema is the union of every path's arg names. On a given compiled path only some names are
// meaningful; declare the rest as `extern INVALID_VALUE_PLACEHOLDER` so that reading one is rejected
// at build time instead of silently using the placeholder value the host passed for that unused name.
struct INVALID_VALUE_PLACEHOLDER {
    INVALID_VALUE_PLACEHOLDER() = delete;
    INVALID_VALUE_PLACEHOLDER(const INVALID_VALUE_PLACEHOLDER&) = delete;
    INVALID_VALUE_PLACEHOLDER& operator=(const INVALID_VALUE_PLACEHOLDER&) = delete;
};
