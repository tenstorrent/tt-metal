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
 * @brief ttsl::indirect<T> (a checked, no-allocator subset of C++26 std::indirect) and
 * ttsl::PimplBase<Impl> (a base class that uses it to implement the pimpl idiom).
 *
 * `indirect<T>` originates from a no-allocator implementation of C++26's std::indirect authored for this
 * codebase; the subset kept here is what the pimpl idiom needs. Unlike std::indirect (UB on valueless
 * access), `indirect<T>`'s observers are CHECKED: they TT_FATAL instead of dereferencing a null pointer.
 * Because of that behavioral difference, this is intentionally NOT a drop-in alias for std::indirect --
 * a future C++26 migration should keep this wrapper rather than `using indirect = std::indirect<T>;`.
 *
 * For background on std::indirect, see:
 * - https://en.cppreference.com/w/cpp/memory/indirect
 * - https://eel.is/c++draft/indirect
 */

namespace ttsl {

// No-allocator, trimmed subset of C++26 std::indirect, sufficient for the pimpl idiom.
// Value semantics: copy deep-copies the owned T; move leaves the source valueless.
// Access is CHECKED: operator*/operator-> TT_FATAL if valueless, so we never dereference a
// null unique_ptr. This deliberately deviates from std::indirect (which is UB on valueless).
// T may be INCOMPLETE where indirect<T> is declared; it must be complete only where the
// copy/move/destroy members are instantiated (the enclosing class's .cpp).
template <typename T>
class indirect {
public:
    using value_type = T;

    // In-place construction (the pimpl entry point): owns make_unique<T>(forward(us)...). The
    // leading std::in_place_t tag disambiguates this constructor from the copy/move constructors
    // below (and from any other single/zero-argument constructor T might have), preventing it from
    // accidentally participating in overload resolution as a generic converting constructor.
    template <class... Us>
        requires(std::is_constructible_v<T, Us...>)
    explicit indirect(std::in_place_t, Us&&... us) :  // NOLINT(readability-named-parameter)
        p(std::make_unique<T>(std::forward<Us>(us)...)) {}

    // Deep copy: allocates a new T from *other (nullptr-safe -- a valueless other yields a valueless copy).
    // Only requires T to be copy-constructible.
    indirect(const indirect& other) {
        if (other.p) {
            p = std::make_unique<T>(*other.p);
        }
    }

    // Deep copy assignment: reallocates from *other.p (nullptr-safe). Deliberately reallocates rather than
    // reusing **this = *other (as the upstream std::indirect implementation does), so this only requires T
    // to be copy-constructible, not copy-assignable.
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

    // Transfers ownership; other becomes valueless. The pointee itself is not moved.
    indirect(indirect&& other) noexcept : p(std::move(other.p)) {}

    indirect& operator=(indirect&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        p = std::move(other.p);
        return *this;
    }

    ~indirect() = default;

    // Observers (CHECKED: TT_FATAL if valueless). Not noexcept, since TT_FATAL throws.
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

    // The only observer safe to call in the valueless state.
    bool valueless_after_move() const noexcept { return p == nullptr; }

private:
    std::unique_ptr<T> p;
};

// Base class for the pimpl idiom. Holds a value-semantic ttsl::indirect<Impl> and exposes it to
// the derived class via protected impl(). Impl may be INCOMPLETE in the header
// (see the contract below); it only needs to be complete in the .cpp that defines the derived
// class's special members.
//
// Contract for the DERIVED class: declare all used special members in the header and define them
// (= default) out-of-line in the .cpp; never = default or implicitly generate them in the header.
// Violating this instantiates Impl-completeness-requiring code (indirect<Impl>'s special members)
// in consumer translation units, where Impl is only forward-declared -- this fails to compile
// rather than silently leaking, but avoid it by following the contract.
template <typename Impl>
class PimplBase {
public:
    // The only part of the pimpl machinery that is intentionally part of the derived class's PUBLIC
    // API: derived types inherit this name directly (the shared convention), so e.g. a new pimpl
    // class exposes valueless_after_move() with no wrapper and no redundant alias.
    bool valueless_after_move() const noexcept { return impl_.valueless_after_move(); }

protected:
    // The rest is derived-class-only machinery (protected), so it does not leak into the public API.
    // Derived special members can still reach these (protected base ctor/SMFs are accessible to the
    // derived class); the base ctor is usable from the derived initializer list.

    // Forwarding constructor: build the Impl in place (forwards to indirect's in_place ctor).
    template <typename... Args>
    explicit PimplBase(std::in_place_t, Args&&... args) :  // NOLINT(readability-named-parameter)
        impl_(std::in_place, std::forward<Args>(args)...) {}

    // Value semantics come from the indirect member, so these are defaulted (still instantiated
    // lazily, in the derived class's .cpp).
    PimplBase(const PimplBase&) = default;
    PimplBase& operator=(const PimplBase&) = default;
    PimplBase(PimplBase&&) noexcept = default;
    PimplBase& operator=(PimplBase&&) noexcept = default;
    ~PimplBase() = default;  // non-virtual: not a polymorphic base (protected prevents delete-through-base)

    // Reference access; indirect::operator* already TT_FATALs on the valueless state. Check
    // valueless_after_move() first if a non-fataling presence check is needed.
    Impl& impl() { return *impl_; }
    const Impl& impl() const { return *impl_; }

private:
    ttsl::indirect<Impl> impl_;
};

}  // namespace ttsl
