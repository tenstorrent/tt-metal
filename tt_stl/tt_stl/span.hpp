// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

namespace ttsl {

template <typename T, std::size_t Extent = std::dynamic_extent>
using Span = std::span<T, Extent>;

template <class Container>
auto make_const_span(const Container& vec) {
    using T = std::remove_reference_t<decltype(*std::begin(std::declval<Container&>()))>;
    return Span<const T>(vec.data(), vec.size());
}

template <class Container>
auto make_span(Container& vec) {
    using T = std::remove_reference_t<decltype(*std::begin(std::declval<Container&>()))>;
    return Span<T>(vec.data(), vec.size());
}

template <class T>
auto as_bytes(Span<T> span) noexcept {
    return Span<const std::byte>(reinterpret_cast<const std::byte*>(span.data()), span.size_bytes());
}

template <class T>
auto as_writable_bytes(Span<T> span) noexcept {
    return Span<std::byte>(reinterpret_cast<std::byte*>(span.data()), span.size_bytes());
}

}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
