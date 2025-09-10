// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <optional>

namespace ttsl {

// `optional_reference` is a trivially copyable wrapper around a pointer to a value.
// This is useful for creating functions that accept optional parameters without the overhead of copying values.
// The main use case is to allow convenient and efficient passing of optional values to functions:
// 1. `optional_reference<T>` binds to `T&`, `optional<T>&`
// 2. `optional_reference<const T>` binds to `const T&`, `const optional<T>&`, and temporaries.
//
// Be aware retaining `optional_reference<T>` is dangerous, as the ownership of the value is not guaranteed!
// Copy the value if you need to retain it.
//
//
// Example usage:
//
// void process_config(optional_reference<const Config> config) {
//     if (config) {
//         apply_settings(config->get_settings());
//     } else {
//         apply_default_settings();
//     }
// }
//
// // Can be called with various types:
// Config my_config;
// process_config(my_config);                    // Pass a reference
// process_config(Config());                     // Pass a temporary
//
// std::optional<Config> maybe_config = load_config();
// process_config(maybe_config);                 // Pass an optional
// process_config(load_config());                // Pass a temporary optional
//
// process_config(std::nullopt);                 // Pass nullopt
//
template <typename T>
class optional_reference {
public:
    optional_reference() : value_(nullptr) {}
    optional_reference(std::nullopt_t) : value_(nullptr) {}

    // Moveable and copyable.
    optional_reference(const optional_reference& other) = default;
    optional_reference(optional_reference&& other) = default;
    optional_reference& operator=(const optional_reference& other) = default;
    optional_reference& operator=(optional_reference&& other) = default;

    // Constructors from lvalues.
    optional_reference(T& value) noexcept : value_(std::addressof(value)) {}
    optional_reference(std::optional<T>& value) {
        if (value.has_value()) {
            value_ = std::addressof(value.value());
        }
    }

    // Constructors that may bind to temporaries.
    template <typename U>
        requires std::is_const_v<T> && std::same_as<std::remove_const_t<T>, U>
    optional_reference(const U& value) noexcept : value_(std::addressof(value)) {}

    template <typename U>
        requires std::is_const_v<T> && std::same_as<std::remove_const_t<T>, U>
    optional_reference(const std::optional<U>& value) {
        if (value.has_value()) {
            value_ = std::addressof(value.value());
        }
    }

    bool has_value() const { return value_ != nullptr; }
    operator bool() const { return has_value(); }

    void reset() { value_ = nullptr; }

    T* operator->() const { return value_; }
    T& operator*() const { return *value_; }
    T& value() const { return *value_; }

    friend bool operator==(const optional_reference& lhs, const optional_reference& rhs) {
        return lhs.value_ == rhs.value_;
    }
    friend bool operator!=(const optional_reference& lhs, const optional_reference& rhs) { return !(lhs == rhs); }

private:
    T* value_ = nullptr;
};

}  // namespace ttsl
