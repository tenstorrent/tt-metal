// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <zoo/Any/VTablePolicy.h>
#include <zoo/AnyContainer.h>

namespace ttsl {

template <typename Signature>
class move_only_function;

namespace detail {

// Matches std::function's inline buffer on libstdc++, so sizeof is unchanged when one replaces the
// other. Captures larger than this are heap allocated.
inline constexpr std::size_t kMoveOnlyFunctionInlinePointers = 2;

// RTTI distinguishes empty storage from trivially destructible targets even under code folding.
using MoveOnlyFunctionStorage =
    zoo::AnyContainer<zoo::Policy<void* [kMoveOnlyFunctionInlinePointers], zoo::Destroy, zoo::Move, zoo::RTTI>>;

template <typename T>
inline constexpr bool is_move_only_function_v = false;

template <typename Signature>
inline constexpr bool is_move_only_function_v<move_only_function<Signature>> = true;

// Match std::move_only_function: null pointers and empty wrappers of this type are empty.
// An empty std::function is still a target; invoking it throws std::bad_function_call.
template <typename T>
constexpr bool is_null_callable(const T& f) noexcept {
    if constexpr (std::is_pointer_v<std::decay_t<T>> || std::is_member_pointer_v<std::decay_t<T>>) {
        return f == nullptr;
    } else if constexpr (is_move_only_function_v<std::decay_t<T>>) {
        return !f;
    } else {
        return false;
    }
}

}  // namespace detail

// A move-only type-erased callable: std::function without the copyability that forces move-only
// captures, such as std::unique_ptr, through a shared_ptr.
//
// Calling an empty instance throws, as std::function does. Moves leave the source empty;
// self-move assignment preserves the target. Const instances can invoke mutable targets,
// as std::function does. Only unqualified R(Args...) signatures are supported.
//
// Requires RTTI. The implementation does not compile under -fno-rtti.
//
// TODO(#57444): becomes std::move_only_function once tt-metal moves to C++23. That is not
// behaviour-neutral: an empty call becomes undefined, const invocation requires a const-qualified
// signature, and the inline capacity and moved-from state become implementation-defined/unspecified.
template <typename R, typename... Args>
class move_only_function<R(Args...)> {
    using Storage = detail::MoveOnlyFunctionStorage;

    template <typename F>
    static Storage make_storage(F&& f) {
        if (detail::is_null_callable(f)) {
            return Storage{};
        }
        // Construct the final storage directly, without moving through a temporary wrapper.
        return Storage{std::in_place_type<std::decay_t<F>>, std::forward<F>(f)};
    }

    template <typename F>
    static R invoke(Storage& storage, Args&&... args) {
        if constexpr (std::is_void_v<R>) {
            std::invoke(*storage.template state<F>(), std::forward<Args>(args)...);
        } else {
            return std::invoke(*storage.template state<F>(), std::forward<Args>(args)...);
        }
    }

    static R call_empty(Storage&, Args&&...) { throw std::bad_function_call{}; }

    mutable Storage storage_;
    R (*executor_)(Storage&, Args&&...) = call_empty;

public:
    using result_type = R;

    move_only_function() noexcept = default;
    move_only_function(std::nullptr_t) noexcept {}

    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, move_only_function> &&
                 std::is_constructible_v<std::decay_t<F>, F &&> && std::is_invocable_r_v<R, std::decay_t<F>&, Args...>)
    move_only_function(F&& f) :
        storage_(make_storage(std::forward<F>(f))),
        executor_(storage_.has_value() ? invoke<std::decay_t<F>> : call_empty) {}

    move_only_function(const move_only_function&) = delete;
    move_only_function& operator=(const move_only_function&) = delete;

    move_only_function(move_only_function&& other) noexcept :
        storage_(std::move(other.storage_)), executor_(other.executor_) {
        other = nullptr;
    }

    move_only_function& operator=(move_only_function&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            executor_ = other.executor_;
            other = nullptr;
        }
        return *this;
    }

    move_only_function& operator=(std::nullptr_t) noexcept {
        storage_ = Storage{};
        executor_ = call_empty;
        return *this;
    }

    explicit operator bool() const noexcept { return has_value(); }
    bool has_value() const noexcept { return storage_.has_value(); }
    bool operator==(std::nullptr_t) const noexcept { return !has_value(); }

    R operator()(Args... args) const { return executor_(storage_, std::forward<Args>(args)...); }

    void swap(move_only_function& other) noexcept {
        if (this != &other) {
            move_only_function temporary{std::move(other)};
            other = std::move(*this);
            *this = std::move(temporary);
        }
    }

    friend void swap(move_only_function& lhs, move_only_function& rhs) noexcept { lhs.swap(rhs); }
};

}  // namespace ttsl
