// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <zoo/FunctionPolicy.h>

namespace ttsl {

namespace detail {

// Matches std::function's inline buffer on libstdc++, so sizeof is unchanged when one replaces the
// other. Captures larger than this are heap allocated.
inline constexpr std::size_t kMoveOnlyFunctionInlinePointers = 2;

template <typename Signature>
using MoveOnlyFunctionBase = zoo::Function<
    // zoo::RTTI is load-bearing, not introspection: operator bool is unreliable without it (#57444).
    zoo::AnyContainer<zoo::Policy<void* [kMoveOnlyFunctionInlinePointers], zoo::Destroy, zoo::Move, zoo::RTTI>>,
    Signature>;

// std::function and std::move_only_function both treat a null function pointer as empty. The
// backing type stores it as an ordinary target instead, which would report engaged and then call
// through null, so it is filtered on the way in.
template <typename T>
constexpr bool is_null_callable(const T& f) noexcept {
    if constexpr (std::is_pointer_v<std::decay_t<T>> || std::is_member_pointer_v<std::decay_t<T>>) {
        return f == nullptr;
    } else {
        return false;
    }
}

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
class move_only_function : public detail::MoveOnlyFunctionBase<Signature> {
    using Base = detail::MoveOnlyFunctionBase<Signature>;

public:
    move_only_function() noexcept = default;
    move_only_function(std::nullptr_t) noexcept {}

    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, move_only_function> && std::is_constructible_v<Base, F &&>)
    move_only_function(F&& f) : Base() {
        if (!detail::is_null_callable(f)) {
            Base::operator=(Base{std::forward<F>(f)});
        }
    }

    move_only_function(move_only_function&&) noexcept = default;
    move_only_function& operator=(move_only_function&&) noexcept = default;
};

}  // namespace ttsl
