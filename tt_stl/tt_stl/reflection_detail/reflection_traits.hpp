// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <experimental/type_traits>
#include <tuple>
#include <utility>

// Detection of the compile-time reflection-attribute protocol: a static
// `attribute_names` tuple paired with an `attribute_values()` method. Shared by the
// reflection facility (reflection.hpp) and the hashing facility (hash.hpp) so neither
// has to duplicate it; both consume it through this namespace.
namespace ttsl::reflection::detail {

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

}  // namespace ttsl::reflection::detail

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
