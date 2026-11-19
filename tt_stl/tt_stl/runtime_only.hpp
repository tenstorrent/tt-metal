// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <ostream>
#include <type_traits>
#include <utility>

namespace ttsl {

// Marks a device-operation attribute that never reaches the compiled program: its value only feeds
// runtime arguments, and the program factory's override path re-applies it on every cache hit.
//
// The default program hash (ttsl::hash::hash_object) and the exact canonical cache key
// (ttsl::hash::canonical_key) both skip a runtime_only field, so changing its value does not
// trigger a recompile or create a new program-cache entry. Every other field is hashed. That
// makes "hash everything" the default and every exclusion an explicit, reviewable annotation,
// instead of a hand-written compute_program_hash that has to list each field it keeps.
//
//     struct operation_attributes_t {
//         ttsl::runtime_only<float> lr = 0.001f;  // patched in override_runtime_arguments
//         float beta1 = 0.9f;                     // compiled in -> hashed
//     };
//
// The wrapper is transparent to read and write: it converts implicitly from and to T, so
// designated initializers and plain reads keep working. It is printed and serialized as the bare
// value, so logs and traces are unchanged.
//
// Only mark a field runtime_only when the factory's override path really does re-apply it.
// A field that is wrongly marked gives a stale cache hit: the program built for an earlier value
// is reused. A field that is wrongly left unmarked only costs an extra compile.
template <typename T>
class runtime_only {
public:
    using value_type = T;

    constexpr runtime_only()
        requires std::default_initializable<T>
        : value_{} {}
    // Implicit on purpose: `.lr = lr` in a designated initializer must keep compiling.
    constexpr runtime_only(T value) noexcept(std::is_nothrow_move_constructible_v<T>) : value_(std::move(value)) {}
    // Also implicit from anything T is constructible from, so e.g. `.scalar = std::nullopt` or
    // `.scalar = some_variant` works for runtime_only<std::optional<V>> (one conversion, not two).
    template <typename U>
        requires(
            !std::is_same_v<std::remove_cvref_t<U>, runtime_only> && !std::is_same_v<std::remove_cvref_t<U>, T> &&
            std::is_constructible_v<T, U &&>)
    constexpr runtime_only(U&& value) : value_(std::forward<U>(value)) {}

    constexpr runtime_only(const runtime_only&) = default;
    constexpr runtime_only(runtime_only&&) noexcept = default;
    constexpr runtime_only& operator=(const runtime_only&) = default;
    constexpr runtime_only& operator=(runtime_only&&) noexcept = default;
    constexpr runtime_only& operator=(T value) {
        value_ = std::move(value);
        return *this;
    }

    // Implicit on purpose: `uint32_t step = attrs.step;` must keep compiling.
    constexpr operator const T&() const noexcept { return value_; }
    constexpr const T& operator*() const noexcept { return value_; }
    constexpr const T* operator->() const noexcept { return &value_; }
    constexpr const T& get() const noexcept { return value_; }

    // Arithmetic T compares through the implicit conversion above (a member operator== would make
    // `field == 0` ambiguous against the built-in comparison). Class types get an explicit one so an
    // enclosing struct can still default its operator==.
    friend constexpr bool operator==(const runtime_only& a, const runtime_only& b)
        requires(!std::is_arithmetic_v<T> && std::equality_comparable<T>)
    {
        return a.value_ == b.value_;
    }

private:
    T value_;
};

template <typename T>
struct is_runtime_only : std::false_type {};
template <typename T>
struct is_runtime_only<runtime_only<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_runtime_only_v = is_runtime_only<std::remove_cvref_t<T>>::value;

template <typename T>
    requires requires(std::ostream& os, const T& v) { os << v; }
std::ostream& operator<<(std::ostream& os, const runtime_only<T>& value) {
    return os << *value;
}

}  // namespace ttsl
