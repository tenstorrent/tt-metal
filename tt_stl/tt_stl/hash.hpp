// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt_stl/concepts.hpp>

namespace ttsl {
namespace hash {

using hash_t = std::uint64_t;
constexpr hash_t DEFAULT_SEED = 1234;

// stuff this in a header somewhere
inline int type_hash_counter = 0;
template <typename T>
inline const int type_hash = type_hash_counter++;

// Trait to check if a type can be hashed via hash_object without hitting static_assert.
// Used by Attribute to guard eager instantiation.
namespace detail {

template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

}  // namespace detail

template <typename T>
inline constexpr bool is_hashable_v =
    std::numeric_limits<T>::is_integer || detail::is_std_hashable_v<T> || concepts::detail::supports_to_hash_v<T> ||
    concepts::detail::supports_compile_time_attributes_v<T> || is_specialization_v<T, std::tuple> ||
    is_specialization_v<T, std::pair> || is_specialization_v<T, std::vector> || is_specialization_v<T, std::set> ||
    is_specialization_v<T, std::map> || is_specialization_v<T, std::unordered_map> ||
    is_specialization_v<T, std::optional> || is_specialization_v<T, std::variant> ||
    is_specialization_v<T, std::reference_wrapper>;

namespace detail {

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>&) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>&) noexcept;

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>&) noexcept;

template <typename T>
inline hash_t hash_object(const T&) noexcept;

template <typename... Types>
inline hash_t hash_objects(hash_t, const Types&...) noexcept;

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept {
    std::size_t hash = 0;
    [&array, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
        (
            [&array, &hash] {
                const auto& element = std::get<Ns>(array);
                hash = hash_objects(hash, element);
            }(),
            ...);
    }(std::make_index_sequence<N>{});
    return hash;
}

template <typename... Ts>
inline hash_t hash_object(const std::variant<Ts...>& variant) noexcept {
    auto active_variant = variant.index();
    return std::visit([&](const auto& value) { return hash_objects(active_variant, value); }, variant);
}

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>& reference) noexcept {
    return hash_object(reference.get());
}

template <typename T>
inline hash_t hash_object(const T& object) noexcept {
    if constexpr (std::numeric_limits<T>::is_integer) {
        return object;
    } else if constexpr (detail::is_std_hashable_v<T>) {
        return std::hash<T>{}(object);
    } else if constexpr (ttsl::concepts::detail::supports_to_hash_v<T>) {
        return object.to_hash();
    } else if constexpr (ttsl::concepts::detail::supports_compile_time_attributes_v<T>) {
        constexpr auto num_attributes = concepts::detail::get_num_attributes<T>();
        hash_t hash = 0;
        const auto attribute_values = object.attribute_values();
        [&hash, &attribute_values]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&hash, &attribute_values] {
                    const auto& attribute = std::get<Ns>(attribute_values);
                    hash = hash_objects(hash, attribute);
                }(),
                ...);
        }(std::make_index_sequence<num_attributes>{});
        return hash;
    } else if constexpr (is_specialization_v<T, std::tuple>) {
        constexpr auto num_elements = std::tuple_size_v<T>;
        hash_t hash = 0;
        [&object, &hash]<size_t... Ns>(std::index_sequence<Ns...>) {
            (
                [&object, &hash] {
                    const auto& element = std::get<Ns>(object);
                    hash = hash_objects(hash, element);
                }(),
                ...);
        }(std::make_index_sequence<num_elements>{});
        return hash;
    } else if constexpr (is_specialization_v<T, std::pair>) {
        hash_t hash = 0;
        return hash_objects(hash, object.first, object.second);
    } else if constexpr (is_specialization_v<T, std::vector>) {
        hash_t hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::set>) {
        hash_t hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::map>) {
        hash_t hash = 0;
        for (const auto& [key, value] : object) {
            hash = hash_objects(hash, key, value);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::unordered_map>) {
        // Sort the unordered map by key to make the hash order invariant
        std::vector<typename T::const_iterator> iterators;
        iterators.reserve(object.size());
        for (auto it = object.begin(); it != object.end(); ++it) {
            iterators.push_back(it);
        }
        std::sort(iterators.begin(), iterators.end(), [](const auto& a, const auto& b) { return a->first < b->first; });

        hash_t hash = 0;
        for (const auto& it : iterators) {
            hash = hash_objects(hash, it->first, it->second);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::optional>) {
        if (object.has_value()) {
            return hash_object(object.value());
        }
        return 0;
    } else {
        static_assert(ttsl::concepts::always_false_v<T>, "Type doesn't support hashing using ttsl::hash::hash_object");
    }
}

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    ([&seed](const auto& arg) { seed ^= hash_object(arg) + 0x9e3779b9 + (seed << 6) + (seed >> 2); }(args), ...);
    return seed;
}

}  // namespace detail

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    return detail::hash_objects(seed, args...);
}

template <typename... Types>
inline hash_t hash_objects_with_default_seed(const Types&... args) noexcept {
    return detail::hash_objects(DEFAULT_SEED, args...);
}

// Ripped out of boost for std::size_t so as to not pull in bulky boost dependencies
template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace hash
}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
