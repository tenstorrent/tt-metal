// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

/**
 * This is an implementation of C++26 compliant std::indirect without allocator support.
 * When we switch in C++26, please replace and alias this class as
 *
 * ```C++
 * template<typename T>
 * using ttsl::indirect = std::indirect<T>;
 * ```
 *
 * Please refer to cppreference/ eel.is for documentation.
 */

namespace ttsl {

/**
 * Implementation of std::indirect without allocator support.
 *
 * Indirect manages the lifetime of owned object (like a unique_ptr) while retaining value semantic.
 *
 * By value semantic, it means that The behavior of special functions like copy assignment/ construction of indirect is
 * the same as it's underlying type (unlike that of unique_ptr, which is deleted).
 *
 * Indirect also handles const propogation correctly as if it's the underlying value. e.g. a dereference to a const
 * indirect returns a const reference. Unlike a pointer semantic where the pointer and the pointed-to type can exhibit
 * different const properties.
 *
 * This class is very helpful for pimpl pattern, where the wrapper class needs to act like the internal class on
 * capabilities like copy, assignment and const correctness.
 *
 * Note that
 * - indirect have a move-from state similar to unique_ptr, where the indirect will become valueless and essentially a
 * null pointer.
 * - The owned type can be an incomplete type (as would be required by pimpl).
 * - Dereference a R-value indirect is safe, as the managed object will be moved out.
 */
template <typename T>
class indirect {
public:
    using value_type = T;
    using pointer = /* allocator_traits<Allocator>::pointer */ T*;
    using const_pointer = /* allocator_traits<Allocator>::const_pointer */ const T*;

    /**
     * Construct an owned object with the default initializer
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/indirect.html
     */
    constexpr explicit indirect() : p(std::make_unique<T>()) {}

    /**
     * Construct an owned object that is the copy of *other
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/indirect.html
     */
    constexpr indirect(const indirect& other) {
        if (other.p) {
            this->p = std::make_unique<T>(*other);
        }
    }

    /**
     * Transfer the ownership of other's owned object. The underlying object will not be moved and other will enter
     * valueless state.  Similiar move constructing a unique_ptr.
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/indirect.html
     */
    constexpr indirect(indirect&& other) noexcept {
        std::swap(this->p, other.p);
        other.p.release();
    }

    /*
     * Construct an owned object with std::forward<U>(u).
     *
     * This is often helpful to do with conversion operators.
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/indirect.html
     */
    template <class U = T>
    constexpr explicit indirect(U&& u)
        requires(
            !std::is_same_v<std::remove_cvref_t<U>, indirect> &&         //
            !std::is_same_v<std::remove_cvref_t<U>, std::in_place_t> &&  //
            std::is_constructible_v<T, U>)
    {
        this->p = std::make_unique<T>(std::forward<T>(u));
    }

    /**
     * Inplace construct an owned object using us... similar to std::make_unique
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/indirect.html
     */
    template <class... Us>
    constexpr explicit indirect(std::in_place_t, Us&&... us)
        requires(std::is_constructible_v<T, Us...>)
    {
        this->p = std::make_unique<T>(std::forward<Us>(us)...);
    }

    /**
     * Inplace construct an owned object using initializer list (and other (us...) args)
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/indirect.html
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
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/operator=.html
     */
    constexpr indirect& operator=(const indirect& other) {
        if (std::addressof(other) == this) {
            return *this;
        }

        if (other.valueless_after_move()) {
            this->p.release();
        } else if (this->valueless_after_move()) {
            this->p = std::make_unique<T>(*other);
        } else {
            **this = *other;
        }
        return *this;
    }

    /**
     * Transfers the ownership of owned object from other to this. The underlying object will not be move assigned, and
     * other will be in a valueless state.
     *
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/operator=.html
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
     * See also: https://en.cppreference.com/w/cpp/memory/indirect/operator=.html
     */
    template <class U = T>
    constexpr indirect& operator=(U&& u)
        requires(
            !std::is_same_v<std::remove_cvref_t<U>, indirect> &&  //
            std::is_constructible_v<T, U> &&                      //
            std::is_assignable_v<T&, U>)
    {
        if (this->valueless_after_move()) {
            this->p = std::make_unique<U>(std::forward<U>(u));
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
    /* Obtain a const R-value reference to the owned object. */
    constexpr const T&& operator*() const&& noexcept { return std::move(*p); }
    /* Obtain a mutable R-value reference to the owned object. */
    constexpr T&& operator*() && noexcept { return std::move(*p); }
    /* Obtain a const pointer to the owned object */
    constexpr const_pointer operator->() const noexcept { return p.get(); }
    /* Obtain a mutable pointer to the owned object */
    constexpr pointer operator->() noexcept { return p.get(); }
    /* Checks if this class is in a moved from state, this can happen if this object has been used to move construct/
     * move assign other indirects, think unique_ptr */
    constexpr bool valueless_after_move() const noexcept { return !bool(p); }

    // [indirect.swap], swap
    constexpr void swap(indirect& other) noexcept { std::swap(this->p, other.p); }

    friend constexpr void swap(indirect& lhs, indirect& rhs) noexcept { std::swap(lhs.p, rhs.p); }

    // [indirect.relops], relational operators
    template <class U>
    friend constexpr bool operator==(const indirect& lhs, const indirect<U>& rhs) noexcept(noexcept(*lhs == *rhs)) {
        return *lhs == *rhs;
    }

    template <class U>
    friend constexpr auto operator<=>(const indirect& lhs, const indirect<U>& rhs) {
        return *lhs <=> *rhs;
    }

    // [indirect.comp.with.t], comparison with T
    template <class U>
    friend constexpr bool operator==(const indirect& lhs, const U& rhs) noexcept(noexcept(*lhs == rhs)) {
        return *lhs == rhs;
    }

    template <class U>
    friend constexpr auto operator<=>(const indirect& lhs, const U& rhs) {
        return *lhs <=> rhs;
    }

private:
    std::unique_ptr<T> p;
};

template <class Value>
indirect(Value) -> indirect<Value>;

}  // namespace ttsl
