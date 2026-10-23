// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

// Compile-time field count of an aggregate, via C++20 parenthesised aggregate initialisation: T is
// constructible from N stand-in arguments exactly when it has at least N fields. The stand-in
// converts to anything except T itself, so a copy constructor never counts as a field.
//
// Used to pin the shape of the experimental Metal 2.0 host API structs in the code that enumerates
// their fields (the Python binding and the generic_op structural hash), so that a field added in
// tt_metal fails the build there instead of silently being left out.
//
// Known limits: a C-array member is not counted (parenthesised init cannot fill it element-wise),
// and a struct with a user-declared constructor counts that constructor's arity instead of its
// fields. Neither shape exists in the pinned headers.
namespace ttnn::metal2_field_count {

template <typename T>
struct AnyFieldOf {
    template <typename U>
        requires(!std::is_same_v<std::remove_cvref_t<U>, T>)
    operator U() const;
};

template <typename T, std::size_t>
using any_field_t = AnyFieldOf<T>;

template <typename T, std::size_t N>
constexpr bool constructible_from_n_fields_v = []<std::size_t... I>(std::index_sequence<I...>) {
    return std::is_constructible_v<T, any_field_t<T, I>...>;
}(std::make_index_sequence<N>{});

}  // namespace ttnn::metal2_field_count

// A field deliberately left out of the enumeration (deprecated, or holding a non-owning reference)
// still counts: bump N and say why next to the use.
#define TTNN_M2_ASSERT_FIELD_COUNT(TYPE, N)                                                           \
    static_assert(                                                                                    \
        ::ttnn::metal2_field_count::constructible_from_n_fields_v<TYPE, N>,                           \
        #TYPE " has fewer than " #N                                                                   \
              " fields: a field was removed or renamed in tt_metal; update every place "              \
              "that enumerates its fields (see the TTNN_M2_ASSERT_FIELD_COUNT uses) and this count"); \
    static_assert(                                                                                    \
        !::ttnn::metal2_field_count::constructible_from_n_fields_v<TYPE, N + 1>,                      \
        #TYPE " has more than " #N                                                                    \
              " fields: a field was added in tt_metal; handle it in every place that "                \
              "enumerates its fields (see the TTNN_M2_ASSERT_FIELD_COUNT uses) and bump this count")
