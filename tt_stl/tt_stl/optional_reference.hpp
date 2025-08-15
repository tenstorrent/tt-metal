// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <optional>

namespace ttsl {

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
    T& value_or(const T& default_value) const { return value_ ? *value_ : default_value; }

    friend bool operator==(const optional_reference& lhs, const optional_reference& rhs) {
        return lhs.value_ == rhs.value_;
    }
    friend bool operator!=(const optional_reference& lhs, const optional_reference& rhs) { return !(lhs == rhs); }

private:
    T* value_ = nullptr;
};

}  // namespace ttsl
