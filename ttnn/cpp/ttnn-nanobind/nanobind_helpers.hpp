// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include <nanobind/stl/unique_ptr.h>

namespace nanobind_helpers {

// https://nanobind.readthedocs.io/en/latest/api_extra.html#unique-pointer-deleter
// https://nanobind.readthedocs.io/en/latest/ownership.html#unique-pointers
// Calling a function taking a unique pointer from Python invalidates the
// passed Python object. Nanobind will refuse further use of it. Use this
// custom deleter version to allow for ownership transfer.
template <typename T>
using unique_ptr = std::unique_ptr<T, nanobind::deleter<T>>;

template <class T, class... Args>
auto make_unique(Args&&... args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename Deleter>
auto steal_rewrap_unique(std::unique_ptr<T, Deleter>& uptr) noexcept {
    return unique_ptr<T>(uptr.release());
}

template <typename T, typename Deleter>
auto steal_rewrap_unique(std::unique_ptr<T, Deleter>&& uptr) noexcept {
    return unique_ptr<T>(uptr.release());
}

// nanobind does not bind properly to std::optional<std::reference_wrapper<T>>, so use a pointer
// instead and rewrap the argument for passing into the c++ interface
template <typename T>
constexpr std::optional<std::reference_wrapper<T>> rewrap_optional(std::optional<T*> arg) noexcept {
    if (arg.has_value() && arg.value() != nullptr) {
        return std::make_optional(std::ref(*arg.value()));
    }
    return std::nullopt;
}

}  // namespace nanobind_helpers

namespace nbh = nanobind_helpers;
