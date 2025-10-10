// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/lazy/expression.hpp"

#include <boost/mp11.hpp>

namespace ttnn::operations::lazy {

inline Expression convert(const Tensor& tensor) { return lazy::defer(tensor).value(); }

constexpr Param convert(float value) noexcept { return value; }

constexpr Param convert(std::int32_t value) noexcept { return value; }

constexpr Param convert(std::uint32_t value) noexcept { return value; }

// additional conversion overloads go here

namespace mp = boost::mp11;

using mp_convert_map = mp::mp_list<
    // convert const Tensor& to ExpressionView
    mp::mp_list<ExpressionView, const Tensor&>,
    // convert float, std::int32_t, and std::uint32_t to Param
    mp::mp_list<Param, float, std::int32_t, std::uint32_t>
    // additional conversion map elements go here
    >;

// Set of all source types in mp_convert_map
// Used to prevent fallback from unintentionally being selected due to value category
using mp_conversions_set = mp::mp_apply<mp::mp_append, mp::mp_transform<mp::mp_rest, mp_convert_map>>;
using mp_remove_cvref_conversions_set = mp::mp_transform<std::remove_cvref_t, mp_conversions_set>;

static_assert(
    mp::mp_is_set<mp_remove_cvref_conversions_set>::value,
    "no_conversion requires mp_remove_cvref_conversions_set to be set type");

template <typename T>
concept no_conversion = not mp::mp_set_contains<mp_remove_cvref_conversions_set, std::remove_cvref_t<T>>::value;

template <no_conversion T>
constexpr T&& convert(T&& value) noexcept {
    return std::forward<T>(value);
}

template <typename Derived, typename From>
struct Overload;

template <typename Derived, template <typename...> typename List, typename... From>
struct Overload<Derived, List<From...>> {
    [[nodiscard]] auto operator()(From... from) const -> decltype(auto) {
        return static_cast<const Derived&>(*this)(lazy::convert(std::forward<From>(from))...);
    }
};

template <typename Derived, typename From>
struct Overloads;

template <typename Derived, template <typename...> typename List, typename... From>
struct Overloads<Derived, List<From...>> : Overload<Derived, From>... {
    using Overload<Derived, From>::operator()...;
};

// Quoted metafunction similar to std::conditional_t
// Reduces class template instantiations by using type alias templates within branches
template <bool>
struct mp_conditional_q;

template <>
struct mp_conditional_q<true> {
    template <typename T, typename F>
    using fn = T;
};

template <>
struct mp_conditional_q<false> {
    template <typename T, typename F>
    using fn = F;
};

// If P<T>::value is true, alias for T, otherwise alias for F
template <typename T, template <typename...> typename P, typename F>
using mp_or_default = mp::mp_invoke_q<mp_conditional_q<P<T>::value>, T, F>;

static_assert(mp::mp_is_map<mp_convert_map>::value, "mp_find_from requires mp_convert_map to be map type");

// Finds all source types convertible to To, or defaults to To itself if none found
template <typename To>
using mp_find_from = mp_or_default<mp::mp_map_find<mp_convert_map, To>, mp::mp_is_list, mp::mp_list<To>>;

// Builds all overload combinations for Derived that map target types (To...)
// to their corresponding convertible source types as defined in mp_convert_map
template <typename Derived, typename... To>
using OverloadsFor = Overloads<Derived, mp::mp_rest<mp::mp_product<mp::mp_list, mp_find_from<To>...>>>;

}  // namespace ttnn::operations::lazy
