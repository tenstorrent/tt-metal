// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>

/**
 * @file pimpl.hpp
 * @brief This header provides two helper utilities specifically for pimpling:
 *   1. ttsl::PimplBase<Impl> -- a base class that implements the pimpl idiom with minimal boilerplate.
 *   2. ttsl::indirect<T> -- a value-semantic wrapper (a checked, no-allocator subset of C++26 std::indirect)
 *      that gives pimpl'd classes consistent move, null-Impl, and copy behavior.
 *
 * Pimpl ("pointer to implementation") lets a class type-erase its internal implementation object by
 * introducing a level of indirection, so a library maintainer can change the class layout of the
 * implementation without updating the public declaration. This is an often-used pattern in this codebase,
 * commonly implemented with a bare `std::unique_ptr<Impl>` member. The two utilities above alleviate the
 * common pain points of that hand-rolled `unique_ptr` approach, each of which centers on the `Impl` pointer
 * going null:
 *
 *   1. CONSISTENT MOVE BEHAVIOR. A pimpl'd class's move should always use pointer move semantics -- transfer
 *      the `Impl` pointer and leave the source null -- because the move ctor/assignment should always be
 *      `noexcept`, and only moving the pointer (never touching `Impl` itself) can guarantee that. These
 *      utilities enforce that convention (and handle self-move / ctor-assignment consistency) once.
 *
 *   2. CONSISTENT NULL-IMPL HANDLING. Because the `Impl` pointer is nullable (a moved-from source is null),
 *      accessing the `Impl` needs to be guarded with special care. These utilities centrally guard that
 *      null behavior to TT_FATAL instead of being UB: the common case accesses the `Impl` by reference
 *      through a null-checked `impl()`, and a consistent `valueless_after_move()` observer lets callers
 *      check for valuelessness up front.
 *
 *   3. VALUE-SEMANTIC COPY. A copyable pimpl'd class usually wants copy to deep-copy the `Impl` (bubbling up
 *      `Impl`'s own copy semantics), not the pointer semantics of `unique_ptr` (which is non-copyable). With
 *      these utilities, a derived class that declares its copy ctor/assignment and `= default`s them gets
 *      that deep copy for free -- no hand-written copy logic.
 *
 * See the individual class documentation below for more details.
 */

namespace ttsl {

/*
 * `indirect<T>` is a wrapper holding a uniquely-owned, dynamically-allocated T with value-like semantics:
 *   1. Constructing an indirect<T> allocates and owns a T, so a live indirect is never null -- it only
 *      becomes valueless (null) after being moved from.
 *   2. Copying an indirect<T> propagates as a deep copy -- the copy owns a fresh T copy-constructed from the
 *      source's T (as if `indirect{std::in_place, *old}`); only requires T to be copy-constructible.
 *   3. Moving an indirect<T> transfers ownership of the T (no copy); the source is left valueless (null).
 *   4. Const propagates to the owned T: a const indirect<T> hands out only const access to T.
 *
 * Access is always guarded: operator* / operator-> TT_FATAL if you attempt to dereference a valueless object.
 * T may be INCOMPLETE where indirect<T> is declared; it must be complete when the
 * copy/move/destroy members are instantiated.
 *
 * This implementation is a trimmed subset of C++26's std::indirect tailored to our codebase:
 *   - Omitted allocator support.
 *   - Defined valueless access: observers TT_FATAL on a valueless dereference instead of being UB.
 * For background on std::indirect, see:
 * - https://en.cppreference.com/w/cpp/memory/indirect
 * - https://eel.is/c++draft/indirect
 */
template <typename T>
class indirect {
public:
    using value_type = T;

    /*
     * In-place construction (the pimpl entry point): owns make_unique<T>(forward(us)...). The
     * leading std::in_place_t tag disambiguates this constructor from the copy/move constructors
     * below (and from any other single/zero-argument constructor T might have), preventing it from
     * accidentally participating in overload resolution as a generic converting constructor.
     */
    template <class... Us>
        requires(std::is_constructible_v<T, Us...>)
    explicit indirect(std::in_place_t, Us&&... us) :  // NOLINT(readability-named-parameter)
        p(std::make_unique<T>(std::forward<Us>(us)...)) {}

    /*
     * Copy constructor: copy-constructs a new indirect<T> by allocating a new T and copying the contents of
     * `other` (or propagates valuelessness). Requires T to be copy-constructible.
     */
    indirect(const indirect& other) : p(other.p ? std::make_unique<T>(*other.p) : nullptr) {}

    /*
     * Copy assignment: replaces the contents by allocating a new T and copying the contents of `other` (or
     * propagates valuelessness). Requires T to be copy-constructible.
     */
    indirect& operator=(const indirect& other) {
        if (this == &other) {
            return *this;
        }
        if (other.p) {
            p = std::make_unique<T>(*other.p);
        } else {
            p.reset();
        }
        return *this;
    }

    /* Move constructor: transfers ownership of the T to *this; other is left valueless. */
    indirect(indirect&& other) noexcept : p(std::move(other.p)) {}

    /* Move assignment: transfers ownership of the T to *this; other is left valueless. */
    indirect& operator=(indirect&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        p = std::move(other.p);
        return *this;
    }

    /* Destructor: releases ownership of the owned T (destroys it, if any). */
    ~indirect() = default;

    /* Observers. */
    T& operator*() & {
        TT_FATAL(p != nullptr, "The object is in a moved-from state and does not hold value. Cannot dereference.");
        return *p;
    }
    const T& operator*() const& {
        TT_FATAL(p != nullptr, "The object is in a moved-from state and does not hold value. Cannot dereference.");
        return *p;
    }
    T* operator->() {
        TT_FATAL(p != nullptr, "The object is in a moved-from state and does not hold value. Cannot dereference.");
        return p.get();
    }
    const T* operator->() const {
        TT_FATAL(p != nullptr, "The object is in a moved-from state and does not hold value. Cannot dereference.");
        return p.get();
    }

    /* Returns true if the object is in a valueless state. The only observer safe to call when valueless. */
    bool valueless_after_move() const noexcept { return p == nullptr; }

private:
    std::unique_ptr<T> p;
};

/*
 * `PimplBase<Impl>` is a base class that implements the pimpl idiom for a derived class. It holds the
 * implementation object in a value-semantic ttsl::indirect<Impl> and exposes it to the derived class through
 * the public impl() accessor.
 *
 * Impl needs to be COMPLETE only in the translation unit that instantiates PimplBase's special members
 * (indirect<Impl>'s copy/move/destroy). It may stay INCOMPLETE (forward-declared) in the header and in every
 * consumer that only includes the header -- which is the whole point of the pimpl idiom.
 *
 * USAGE (a copyable Widget; for a move-only type, delete the copy members instead):
 *
 *   // widget.hpp -- WidgetImpl is only forward-declared, so it stays INCOMPLETE for consumers.
 *   class WidgetImpl;
 *   class Widget : public ttsl::PimplBase<WidgetImpl> {
 *   public:
 *       explicit Widget(int value);
 *       // Declared here, defined out-of-line in widget.cpp:
 *       ~Widget();
 *       Widget(const Widget&);
 *       Widget& operator=(const Widget&);
 *       Widget(Widget&&) noexcept;
 *       Widget& operator=(Widget&&) noexcept;
 *       int value() const;
 *   };
 *
 *   // widget.cpp -- WidgetImpl is COMPLETE here, so the defaulted members can be instantiated.
 *   class WidgetImpl { public: int value; explicit WidgetImpl(int v) : value(v) {} };
 *   Widget::Widget(int value) : PimplBase(std::in_place, value) {}  // forwards to WidgetImpl(value)
 *   Widget::~Widget() = default;
 *   Widget::Widget(const Widget&) = default;                 // deep-copies WidgetImpl
 *   Widget& Widget::operator=(const Widget&) = default;
 *   Widget::Widget(Widget&&) noexcept = default;             // transfers Impl; source left valueless
 *   Widget& Widget::operator=(Widget&&) noexcept = default;
 *   int Widget::value() const { return impl().value; }       // impl() TT_FATALs if valueless
 *
 */
template <typename Impl>
class PimplBase {
public:
    /*
     * Returns true if this class is in a valueless state after a move. A valueless instance may only be
     * assigned to; all other access will TT_FATAL.
     */
    bool valueless_after_move() const noexcept { return impl_.valueless_after_move(); }

    /*
     * Access the internal Impl object by reference. TT_FATALs if the object is in a valueless (moved-from) state.
     */
    Impl& impl() { return *impl_; }

    /*
     * Access the internal Impl object by reference. TT_FATALs if the object is in a valueless (moved-from) state.
     */
    const Impl& impl() const { return *impl_; }

protected:
    /* Forwarding constructor: build the Impl in place (forwards to indirect's in_place ctor). */
    template <typename... Args>
    explicit PimplBase(std::in_place_t, Args&&... args) :  // NOLINT(readability-named-parameter)
        impl_(std::in_place, std::forward<Args>(args)...) {}

    /*
     * Copy constructor: propagates the copy behavior of Impl (deep-copies the Impl), or propagates valuelessness if
     * other is valueless.
     *
     * Note: If Impl is on a different linker boundary than the declaration, declare this function in the header
     * and give it a default implementation (e.g. `Derived::Derived(const Derived&) = default;`) in the .cpp,
     * where Impl is complete.
     */
    PimplBase(const PimplBase&) = default;

    /*
     * Copy assignment: propagates the copy behavior of Impl (deep-copies the Impl), or propagates valuelessness if
     * other is valueless.
     *
     * Note: If Impl is on a different linker boundary than the declaration, declare this function in the header
     * and give it a default implementation (e.g. `Derived& Derived::operator=(const Derived&) = default;`) in
     * the .cpp, where Impl is complete.
     */
    PimplBase& operator=(const PimplBase&) = default;

    /* Move constructor: transfers ownership of the Impl to *this; other is left valueless. */
    PimplBase(PimplBase&&) noexcept = default;

    /* Move assignment: transfers ownership of the Impl to *this; other is left valueless. */
    PimplBase& operator=(PimplBase&&) noexcept = default;

    /*
     * Destructor: destroys the owned Impl (if any).
     *
     * Note: If Impl is on a different linker boundary than the declaration, declare this function in the header
     * and give it a default implementation (e.g. `Derived::~Derived() = default;`) in the .cpp, where Impl is
     * complete. This is the most common case: even a move-only pimpl'd type needs its destructor defined
     * out-of-line, since destroying the Impl requires it to be complete.
     */
    ~PimplBase() = default;

private:
    ttsl::indirect<Impl> impl_;
};

}  // namespace ttsl
