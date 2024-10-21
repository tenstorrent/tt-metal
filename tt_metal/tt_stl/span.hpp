// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// prefer standard library implementation
#if __has_include(<span>)

#include <span>
#define _TT_STL_SPAN_NS ::std

// fallback to boost library implementation
#elif __has_include(<boost/core/span.hpp>)

#include <boost/core/span.hpp>
#define _TT_STL_SPAN_NS ::boost

#else

#error "No implementation available for tt::stl::Span"

#endif

namespace tt::stl {

using _TT_STL_SPAN_NS::dynamic_extent;

namespace detail {

using _TT_STL_SPAN_NS::span;

#undef _TT_STL_SPAN_NS

template <class T, std::size_t Extent>
class SpanBase : public span<T, Extent> {
   public:
    using span<T, Extent>::span;
};

template <class T, std::size_t Extent>
class SpanBase<const T, Extent> : public span<const T, Extent> {
   public:
    using span<const T>::span;

    // expose constructor from initializer_list for const-qualified element_type
    explicit(Extent != dynamic_extent) constexpr SpanBase(std::initializer_list<T> il) noexcept : span<const T>(il) {}
};

}  // namespace detail

template <class T, std::size_t Extent = dynamic_extent>
class Span : detail::SpanBase<T, Extent> {
    using base = detail::SpanBase<T, Extent>;

   public:
    // Member types
    using typename base::const_pointer;
    using typename base::const_reference;
    using typename base::difference_type;
    using typename base::element_type;
    using typename base::iterator;
    using typename base::pointer;
    using typename base::reference;
    using typename base::reverse_iterator;
    using typename base::size_type;
    using typename base::value_type;

    // Member constants
    using base::extent;

    using base::base;
    using base::operator=;

    // Iterators
    using base::begin;
    using base::end;
    using base::rbegin;
    using base::rend;

    // Element access
    using base::back;
    using base::front;
    using base::operator[];
    using base::data;

    // Observers
    using base::empty;
    using base::size;
    using base::size_bytes;

    // Subviews
    using base::first;
    using base::last;
    using base::subspan;
};

template <class It, class EndOrSize>
Span(It, EndOrSize) -> Span<std::remove_reference_t<std::iter_reference_t<It>>>;

template <class T, std::size_t N>
Span(T (&)[N]) -> Span<T, N>;

template <class T, std::size_t N>
Span(std::array<T, N> &) -> Span<T, N>;

template <class T, std::size_t N>
Span(const std::array<T, N> &) -> Span<const T, N>;

template <class R>
Span(R &&) -> Span<std::remove_reference_t<std::ranges::range_reference_t<R>>>;

}  // namespace tt::stl

#if __has_include(<ranges>)

#include <ranges>

// https://en.cppreference.com/w/cpp/ranges/borrowed_range
// The concept std::ranges::borrowed_range defines the requirements of a range such that a function can take it by value
// and return iterators obtained from it without danger of dangling.
template <class T, std::size_t Extent>
constexpr bool std::ranges::enable_borrowed_range<tt::stl::Span<T, Extent>> = true;

// https://en.cppreference.com/w/cpp/ranges/view
// The std::ranges::view concept specifies the requirements of a range type that has suitable semantic properties for
// use in constructing range adaptor pipelines.
template <class T, std::size_t Extent>
constexpr bool std::ranges::enable_view<tt::stl::Span<T, Extent>> = true;

#endif
