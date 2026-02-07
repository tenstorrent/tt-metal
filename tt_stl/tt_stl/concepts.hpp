// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <experimental/type_traits>
#include <tuple>
#include <type_traits>

namespace ttsl {

namespace detail {
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};
}  // namespace detail

template <typename Test, template <typename...> class Ref>
constexpr bool is_specialization_v = detail::is_specialization<Test, Ref>::value;

namespace concepts {

template <typename... T>
inline constexpr bool always_false_v = false;

namespace detail {

template <typename T>
using has_to_hash_t = decltype(std::declval<const T>().to_hash());

template <typename T>
constexpr bool supports_to_hash_v = std::experimental::is_detected_v<has_to_hash_t, T>;

template <typename T>
using has_to_string_t = decltype(std::declval<const T>().to_string());

template <typename T>
constexpr bool supports_to_string_v = std::experimental::is_detected_v<has_to_string_t, T>;

template <typename T>
using has_attribute_names_t = decltype(std::declval<T>().attribute_names);

template <typename T>
using has_attribute_values_t = decltype(std::declval<T>().attribute_values());

template <typename T>
constexpr bool supports_compile_time_attributes_v = std::experimental::is_detected_v<has_attribute_names_t, T> and
                                                    std::experimental::is_detected_v<has_attribute_values_t, T>;

template <typename T>
static constexpr std::size_t get_num_attributes() {
    static_assert(
        std::tuple_size_v<decltype(T::attribute_names)> ==
            std::tuple_size_v<decltype(std::declval<T>().attribute_values())>,
        "Number of attribute_names must match number of attribute_values");
    return std::tuple_size_v<decltype(T::attribute_names)>;
}

template <typename T>
constexpr bool supports_conversion_to_string_v = supports_to_string_v<T> or supports_compile_time_attributes_v<T>;

}  // namespace detail
}  // namespace concepts
}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
