// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include <zoo/FunctionPolicy.h>

namespace ttsl {

namespace detail {

// Matches std::function's inline buffer on libstdc++, so sizeof is unchanged when one replaces the
// other. Captures larger than this are heap allocated.
inline constexpr std::size_t kMoveOnlyFunctionInlinePointers = 2;

}  // namespace detail

// A move-only type-erased callable: std::function without the copyability that forces move-only
// captures, such as std::unique_ptr, through a shared_ptr.
//
// Calling an empty instance throws, as std::function does.
//
// Requires RTTI. The implementation does not compile under -fno-rtti.
//
// TODO(#57444): becomes std::move_only_function once tt-metal moves to C++23. That is not
// behaviour-neutral: an empty call becomes undefined and the inline capacity becomes
// implementation-defined.
template <typename Signature>
using move_only_function = zoo::Function<
    // zoo::RTTI is load-bearing, not introspection: operator bool is unreliable without it (#57444).
    zoo::AnyContainer<
        zoo::Policy<void * [detail::kMoveOnlyFunctionInlinePointers], zoo::Destroy, zoo::Move, zoo::RTTI>>,
    Signature>;

}  // namespace ttsl
