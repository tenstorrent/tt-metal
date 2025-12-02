// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

/**
 * @file indirect.hpp
 * @brief Implementation of C++26 compliant std::indirect without allocator support.
 *
 * This is an implementation of C++26's std::indirect that provides value semantics
 * for dynamically allocated objects. When migrating to C++26, replace with:
 *
 * @code{.cpp}
 * template<typename T>
 * using ttsl::indirect = std::indirect<T>;
 * @endcode
 *
 * For detailed documentation, refer to:
 * - https://en.cppreference.com/w/cpp/memory/indirect
 * - https://eel.is/c++draft/indirect
 */

/**
 * @namespace ttsl
 * @brief Tenstorrent Standard Library namespace containing utility types and functions.
 */
namespace ttsl {

/**
 * @brief Implementation of std::indirect without allocator support.
 *
 * @tparam T The type of object to manage indirectly.
 *
 * `indirect` manages the lifetime of an owned object (like `unique_ptr`) while providing value semantics.
 *
 * ## Value Semantics
 * Unlike `unique_ptr`, which has deleted copy operations, `indirect` provides copy construction and
 * copy assignment that behave like the underlying type T. This means copying an `indirect<T>` creates
 * a deep copy of the managed object.
 *
 * ## Const Propagation
 * `indirect` correctly propagates const-qualification to the managed object. For example, dereferencing
 * a `const indirect<T>` returns a `const T&`. This differs from pointer semantics where the pointer
 * and pointed-to object can have independent const-qualification.
 *
 * ## Use Cases
 * This class is particularly useful for the PImpl (Pointer to Implementation) idiom, where the wrapper
 * class needs to preserve value semantics (copy, assignment) and const-correctness of the internal class.
 *
 * ## Important Notes
 * - `indirect` has a move-from state similar to `unique_ptr`. After being moved from, the `indirect`
 *   becomes valueless (essentially a null pointer). Check with `valueless_after_move()`.
 * - The managed type T can be an incomplete type (as required by PImpl pattern).
 * - Dereferencing an rvalue `indirect` is safe; the managed object will be moved out.
 *
 * @see https://en.cppreference.com/w/cpp/memory/indirect
 */
template <typename T>
class indirect {
public:
    using value_type = T;
    using pointer = /* allocator_traits<Allocator>::pointer */ T*;
    using const_pointer = /* allocator_traits<Allocator>::const_pointer */ const T*;

    /**
     * @brief Constructs an owned object using default initialization.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/indirect
     */
    constexpr explicit indirect() : p(std::make_unique<T>()) {}

    /**
     * Construct an owned object that is the copy of *other.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/indirect
     */
    constexpr indirect(const indirect& other) {
        if (other.p) {
            this->p = std::make_unique<T>(*other);
        }
    }

    /**
     * Transfer the ownership of other's owned object. The underlying object will not be moved and other will enter
     * valueless state. Similar to move constructing a unique_ptr.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/indirect
     */
    constexpr indirect(indirect&& other) noexcept {
        std::swap(this->p, other.p);
        other.p.reset();
    }

    /**
     * Construct an owned object with std::forward<U>(u).
     *
     * This is often helpful with conversion operators.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/indirect
     */
    template <class U = T>
    constexpr explicit indirect(U&& u)
        requires(
            !std::is_same_v<std::remove_cvref_t<U>, indirect> &&         //
            !std::is_same_v<std::remove_cvref_t<U>, std::in_place_t> &&  //
            std::is_constructible_v<T, U>)
    {
        this->p = std::make_unique<T>(std::forward<U>(u));
    }

    /**
     * In-place construct an owned object using us..., similar to std::make_unique.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/indirect
     */
    template <class... Us>
    constexpr explicit indirect(std::in_place_t, Us&&... us)
        requires(std::is_constructible_v<T, Us...>)
    {
        this->p = std::make_unique<T>(std::forward<Us>(us)...);
    }

    /**
     * In-place construct an owned object using initializer list (and other us... args).
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/indirect
     */
    template <class I, class... Us>
    constexpr explicit indirect(std::in_place_t, std::initializer_list<I> ilist, Us&&... us)
        requires(std::is_constructible_v<T, std::initializer_list<I>, Us...>)
    {
        this->p = std::make_unique<T>(std::move(ilist), std::forward<Us>(us)...);
    }

    // [indirect.dtor], destructor
    constexpr ~indirect() = default;

    // [indirect.assign], assignment
    /**
     * Copy assigns the internally owned object with other.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/operator=.html
     */
    constexpr indirect& operator=(const indirect& other) {
        if (std::addressof(other) == this) {
            return *this;
        }

        if (other.valueless_after_move()) {
            this->p.reset();
        } else if (this->valueless_after_move()) {
            this->p = std::make_unique<T>(*other);
        } else {
            **this = *other;
        }
        return *this;
    }

    /**
     * Transfers the ownership of the owned object from other to this. The underlying object will not be move assigned,
     * and other will be in a valueless state.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/operator=.html
     */
    constexpr indirect& operator=(indirect&& other) noexcept {
        if (std::addressof(other) == this) {
            return *this;
        }

        this->p = std::move(other.p);
        return *this;
    }

    /**
     * Move assigns the underlying object with u.
     *
     * @see https://en.cppreference.com/w/cpp/memory/indirect/operator=.html
     */
    template <class U = T>
    constexpr indirect& operator=(U&& u)
        requires(
            !std::is_same_v<std::remove_cvref_t<U>, indirect> &&  //
            std::is_constructible_v<T, U> &&                      //
            std::is_assignable_v<T&, U>)
    {
        if (this->valueless_after_move()) {
            this->p = std::make_unique<T>(std::forward<U>(u));
        } else {
            **this = std::forward<U>(u);
        }

        return *this;
    }

    // [indirect.obs], observers
    /* Obtain a const reference to the owned object. */
    constexpr const T& operator*() const& noexcept { return *p; }
    /* Obtain a (mutable) reference to the owned object. */
    constexpr T& operator*() & noexcept { return *p; }
    /* Obtain a const rvalue reference to the owned object. */
    constexpr const T&& operator*() const&& noexcept { return std::move(*p); }
    /* Obtain a mutable rvalue reference to the owned object. */
    constexpr T&& operator*() && noexcept { return std::move(*p); }
    /* Obtain a const pointer to the owned object. */
    constexpr const_pointer operator->() const noexcept { return p.get(); }
    /* Obtain a mutable pointer to the owned object. */
    constexpr pointer operator->() noexcept { return p.get(); }
    /* Checks if this is in a moved-from state (can happen after being used to move construct/assign other indirects).
     */
    constexpr bool valueless_after_move() const noexcept { return !bool(p); }

    // [indirect.swap], swap
    constexpr void swap(indirect& other) noexcept { std::swap(this->p, other.p); }

    friend constexpr void swap(indirect& lhs, indirect& rhs) noexcept { lhs.swap(rhs); }

    // [indirect.relops], relational operators
    template <class U>
    friend constexpr bool operator==(const indirect& lhs, const indirect<U>& rhs) noexcept(noexcept(*lhs == *rhs)) {
        if (lhs.valueless_after_move() || rhs.valueless_after_move()) {
            return lhs.valueless_after_move() && rhs.valueless_after_move();
        }
        return *lhs == *rhs;
    }

    template <class U>
    friend constexpr auto operator<=>(const indirect& lhs, const indirect<U>& rhs) {
        if (lhs.valueless_after_move() || rhs.valueless_after_move()) {
            return !lhs.valueless_after_move() <=> !rhs.valueless_after_move();
        }
        return *lhs <=> *rhs;
    }

    // [indirect.comp.with.t], comparison with T
    template <class U>
    friend constexpr bool operator==(const indirect& lhs, const U& rhs) noexcept(noexcept(*lhs == rhs)) {
        return *lhs == rhs;
    }

    template <class U>
    friend constexpr auto operator<=>(const indirect& lhs, const U& rhs) -> decltype(*lhs <=> rhs) {
        if (lhs.valueless_after_move()) {
            return std::strong_ordering::less;
        }

        return *lhs <=> rhs;
    }

private:
    std::unique_ptr<T> p;
};

template <class Value>
indirect(Value) -> indirect<Value>;

}  // namespace ttsl
