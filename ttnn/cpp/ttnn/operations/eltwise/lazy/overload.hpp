// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/lazy/expression.hpp"

#include <boost/mp11.hpp>

namespace ttnn::operations::lazy {

template <std::same_as<ExpressionView>>
Expression convert_to(std::same_as<const Tensor&> auto tensor) {
    return lazy::defer(tensor).value();
}

// additional conversion overloads go here

template <typename To, typename From>
constexpr To convert_to(From value) {
    return static_cast<To>(std::forward<From>(value));
}

template <typename Derived, typename To, typename From>
struct Overload;

template <typename Derived, template <typename...> typename List, typename... To, typename... From>
struct Overload<Derived, List<To...>, List<From...>> {
    [[nodiscard]] auto operator()(From... from) const -> decltype(auto) {
        return static_cast<const Derived&>(*this)(lazy::convert_to<To, From>(std::forward<From>(from))...);
    }
};

template <typename Derived, typename To, typename From>
struct Overloads;

template <typename Derived, typename To, template <typename...> typename List, typename... From>
struct Overloads<Derived, To, List<From...>> : Overload<Derived, To, From>... {
    using Overload<Derived, To, From>::operator()...;
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

namespace mp = boost::mp11;

using mp_convert_map = mp::mp_list<
    // convert const Tensor& to ExpressionView
    mp::mp_list<ExpressionView, const Tensor&>,
    // convert float, std::int32_t, and std::uint32_t to Param
    mp::mp_list<Param, float, std::int32_t, std::uint32_t>
    // additional conversion map elements go here
    >;

// If P<T>::value is true, alias for T, otherwise alias for F
template <typename T, template <typename...> typename P, typename F>
using mp_or_default = mp::mp_invoke_q<mp_conditional_q<P<T>::value>, T, F>;

static_assert(mp::mp_is_map<mp_convert_map>::value, "mp_find_from requires mp_convert_map to be map type");

// Finds all source types convertible to To, or defaults to To itself if none found
template <typename To, typename Map>
using mp_find_from = mp_or_default<mp::mp_map_find<Map, To>, mp::mp_is_list, mp::mp_list<To>>;

// Builds all overload combinations for Derived that map target types (To...)
// to their corresponding convertible source types as defined in mp_convert_map
template <typename Derived, typename... To>
struct OverloadsFor : Overloads<
                          Derived,
                          mp::mp_list<To...>,
                          mp::mp_rest<mp::mp_product<mp::mp_list, mp_find_from<To, mp_convert_map>...>>> {};

}  // namespace ttnn::operations::lazy
