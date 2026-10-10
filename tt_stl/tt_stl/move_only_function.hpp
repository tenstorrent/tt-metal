// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include <zoo/FunctionPolicy.h>

namespace ttsl {

namespace detail {

// Inline buffer plus one vtable pointer makes sizeof 32, the same as std::function on libstdc++.
// Captures larger than this are heap allocated.
inline constexpr std::size_t kMoveOnlyFunctionInlinePointers = 3;

// The invoker lives in the per-type vtable rather than in each object, which is what leaves room
// for a third pointer of buffer at that size.
template <typename Signature>
using MoveOnlyFunctionBase = zoo::VTableFunction<kMoveOnlyFunctionInlinePointers, Signature>;

template <typename T>
struct is_in_place_type : std::false_type {};
template <typename T>
struct is_in_place_type<std::in_place_type_t<T>> : std::true_type {};

template <typename T>
struct is_std_function : std::false_type {};
template <typename R, typename... A>
struct is_std_function<std::function<R(A...)>> : std::true_type {};

template <typename T>
inline constexpr bool is_nullable_callable_v =
    std::is_pointer_v<T> || std::is_member_pointer_v<T> || is_std_function<T>::value;

template <typename F, typename Self, typename R, typename... Args>
concept MoveOnlyFunctionTarget =
    !std::is_same_v<std::remove_cvref_t<F>, Self> && !is_in_place_type<std::remove_cvref_t<F>>::value &&
    std::is_invocable_r_v<R, std::decay_t<F>, Args...> && std::is_invocable_r_v<R, std::decay_t<F>&, Args...>;

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

// A polyfill for C++23 std::move_only_function: a type-erased wrapper for any callable that is
// move-constructible, including ones that cannot be copied, such as a lambda capturing a
// std::unique_ptr.
//
// It follows std::move_only_function except that:
//   - calling an empty instance throws std::bad_function_call, as std::function does, rather than
//     being undefined;
//   - only the unqualified R(Args...) signature is provided, not the const, reference or noexcept
//     qualified forms;
//   - it does not compile under -fno-rtti.
//
// TODO(#57444): replace with std::move_only_function once every supported standard library ships
// it; libc++ does not as of LLVM 20. Calling an empty instance then becomes undefined.
template <typename Signature>
class move_only_function;

template <typename R, typename... Args>
class move_only_function<R(Args...)> : private detail::MoveOnlyFunctionBase<R(Args...)> {
    using Base = detail::MoveOnlyFunctionBase<R(Args...)>;

public:
    using result_type = R;

    move_only_function() noexcept = default;
    move_only_function(std::nullptr_t) noexcept {}

    move_only_function(move_only_function&& other) noexcept : Base(static_cast<Base&&>(other)) { clear(other); }

    template <typename F>
        requires(
            detail::MoveOnlyFunctionTarget<F, move_only_function, R, Args...> &&
            !detail::is_nullable_callable_v<std::decay_t<F>>)
    move_only_function(F&& f) : Base(std::forward<F>(f)) {}

    template <typename F>
        requires(
            detail::MoveOnlyFunctionTarget<F, move_only_function, R, Args...> &&
            detail::is_nullable_callable_v<std::decay_t<F>>)
    move_only_function(F&& f) : Base() {
        if (!detail::is_empty_callable(f)) {
            Base::operator=(Base{std::forward<F>(f)});
        }
    }

    template <typename T, typename... CArgs>
        requires(std::is_constructible_v<T, CArgs...> && std::is_invocable_r_v<R, T&, Args...>)
    explicit move_only_function(std::in_place_type_t<T>, CArgs&&... args) : Base(T(std::forward<CArgs>(args)...)) {}

    template <typename T, typename U, typename... CArgs>
        requires(
            std::is_constructible_v<T, std::initializer_list<U>&, CArgs...> && std::is_invocable_r_v<R, T&, Args...>)
    explicit move_only_function(std::in_place_type_t<T>, std::initializer_list<U> il, CArgs&&... args) :
        Base(T(il, std::forward<CArgs>(args)...)) {}

    move_only_function(const move_only_function&) = delete;
    move_only_function& operator=(const move_only_function&) = delete;

    // The self-check is required: a self-move corrupts a heap-stored target.
    move_only_function& operator=(move_only_function&& other) noexcept {
        if (this != &other) {
            Base::operator=(static_cast<Base&&>(other));
            clear(other);
        }
        return *this;
    }

    move_only_function& operator=(std::nullptr_t) noexcept {
        clear(*this);
        return *this;
    }

    template <typename F>
        requires std::is_constructible_v<move_only_function, F>
    move_only_function& operator=(F&& f) {
        move_only_function(std::forward<F>(f)).swap(*this);
        return *this;
    }

    ~move_only_function() = default;

    // Compares the invoker with the empty state's thrower rather than using zoo's isDefault(), whose
    // destroy-pointer comparison identical-code folding can defeat.
    explicit operator bool() const noexcept {
        return Base::container()->template vTable<Callable>()->executor_ != &Callable::throwStdBadFunctionCall;
    }

    R operator()(Args... args) { return Base::operator()(std::forward<Args>(args)...); }

    void swap(move_only_function& other) noexcept { Base::swap(static_cast<Base&>(other)); }

    friend void swap(move_only_function& a, move_only_function& b) noexcept { a.swap(b); }

    friend bool operator==(const move_only_function& f, std::nullptr_t) noexcept { return !f; }

private:
    using Callable = zoo::CallableViaVTable<R(Args...)>;

    // The base move leaves the source's vtable in place, so it still looks engaged and calling it
    // runs the moved-out callable. reset() installs the empty vtable, invoker included.
    static void clear(move_only_function& f) noexcept { f.Base::reset(); }
};

}  // namespace ttsl
