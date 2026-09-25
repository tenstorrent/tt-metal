// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
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

template <typename T>
struct is_std_function : std::false_type {};
template <typename R, typename... A>
struct is_std_function<std::function<R(A...)>> : std::true_type {};

// std::move_only_function yields an empty wrapper for a null pointer or an empty std::function;
// the backing type stores them as ordinary engaged targets. Only those two -- not a general
// emptiness probe; a new source type needs a case here rather than falling through as engaged.
template <typename T>
bool is_empty_callable(const T& f) noexcept {
    using D = std::decay_t<T>;
    if constexpr (std::is_pointer_v<D> || std::is_member_pointer_v<D>) {
        return f == nullptr;
    } else if constexpr (is_std_function<D>::value) {
        return !static_cast<bool>(f);
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
// This is a thin class rather than an alias because the backing type deviates from
// std::move_only_function in six ways that would otherwise be inherited; each override below says
// which. Measured free: identical sizeof and within noise of the bare alias on the job path.
//
// TODO(#57444): becomes std::move_only_function once tt-metal moves to C++23. That is not
// behaviour-neutral: an empty call becomes undefined and the inline capacity becomes
// implementation-defined.
template <typename Signature>
class move_only_function;

template <typename R, typename... Args>
class move_only_function<R(Args...)> : public detail::MoveOnlyFunctionBase<R(Args...)> {
    using Base = detail::MoveOnlyFunctionBase<R(Args...)>;

public:
    move_only_function() noexcept = default;
    move_only_function(std::nullptr_t) noexcept {}

    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, move_only_function> && std::is_constructible_v<Base, F &&>)
    move_only_function(F&& f) : Base() {
        if (!detail::is_empty_callable(f)) {
            Base::operator=(Base{std::forward<F>(f)});
        }
    }

    // The defaulted move leaves the source engaged, so operator bool lies about it and calling it
    // dereferences null.
    move_only_function(move_only_function&& other) noexcept : Base(static_cast<Base&&>(other)) { other.Base::reset(); }

    // The self-check is required, not defensive: without it a self-move corrupts a heap-stored
    // target and the next call segfaults.
    move_only_function& operator=(move_only_function&& other) noexcept {
        if (this != &other) {
            Base::operator=(static_cast<Base&&>(other));
            other.Base::reset();
        }
        return *this;
    }

    move_only_function(const move_only_function&) = delete;
    move_only_function& operator=(const move_only_function&) = delete;

    // Hides the base's non-explicit conversion, which let `int n = f;` compile.
    explicit operator bool() const noexcept { return this->Base::has_value(); }
    bool operator==(std::nullptr_t) const noexcept { return !static_cast<bool>(*this); }

    // Hides the base's unconstrained variadic call operator, which made std::is_invocable_v accept
    // any argument list and so broke code constrained on std::invocable.
    R operator()(Args... args) const { return Base::operator()(std::forward<Args>(args)...); }
};

}  // namespace ttsl
