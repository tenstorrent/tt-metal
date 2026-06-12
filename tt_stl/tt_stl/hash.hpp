// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <experimental/type_traits>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <reflect>

#include <tt_stl/concepts.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/type_name.hpp>

// Recursive, structural hashing for arbitrary objects.
//
// ttsl::hash::hash_objects_with_default_seed(a, b, ...) returns a hash_t for any
// number of hashable objects; hash_objects(seed, ...) folds them into an existing
// seed. Hashing is recursive: the hash of a composite — a container, tuple,
// optional, or struct — is built by hashing each of its elements/members in turn,
// all the way down to scalars. You only make the leaf types hashable; composites
// compose for free.
//
// A type is hashable if it is (resolved in this order):
//   - an integer, or anything with a std::hash<T> specialization;
//   - a standard container/wrapper: tuple, pair, array, vector, span, set, map,
//     unordered_map (hashed order-independently), optional, variant, or
//     reference_wrapper — each hashed by recursing into its elements;
//   - a type that opts in via a customization point below; or
//   - an aggregate struct, whose members are hashed via reflection.
//
// Customizing hashing for your own type — pick one (resolved highest-priority first):
//
//   1. Specialize std::hash<YourType>.
//
//   2. Add a to_hash() method, when you want to compute the hash yourself:
//          hash_t to_hash() const { return ...; }
//
//   3. Declare the attributes to hash (idiomatic — the same hook also drives
//      printing/serialization). Give a static tuple of names and a method
//      returning a tuple of the values; the values are hashed recursively:
//          static constexpr auto attribute_names = std::forward_as_tuple("height", "width");
//          auto attribute_values() const { return std::forward_as_tuple(height_, width_); }
//
//   4. Do nothing: a plain aggregate struct has its members hashed automatically
//      via reflection.
namespace ttsl::hash {

using hash_t = std::uint64_t;

// Public hashing API (defined below).
template <typename... Types>
hash_t hash_objects(hash_t seed, const Types&... args) noexcept;

template <typename... Types>
hash_t hash_objects_with_default_seed(const Types&... args) noexcept;

template <typename T>
void hash_combine(std::size_t& seed, const T& value);

}  // namespace ttsl::hash

namespace ttsl {

namespace detail {
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};
}  // namespace detail

template <typename Test, template <typename...> class Ref>
constexpr bool is_specialization_v = detail::is_specialization<Test, Ref>::value;

template <typename T>
struct is_span : std::false_type {};

template <typename T, std::size_t Extent>
struct is_span<std::span<T, Extent>> : std::true_type {};

template <typename T>
constexpr bool is_span_v = is_span<T>::value;

namespace hash {

constexpr bool DEBUG_HASH_OBJECT_FUNCTION = false;

constexpr hash_t DEFAULT_SEED = 1234;

// stuff this in a header somewhere
inline int type_hash_counter = 0;
template <typename T>
inline const int type_hash = type_hash_counter++;

namespace detail {

// Forward declare the hash_object overloads and hash_objects so the definitions below can recurse freely.
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

// Detection of the member-based hashing customization points a type can opt into:
//  - a `to_hash()` method, or
//  - the `attribute_names` / `attribute_values()` pair.
template <typename T>
using has_to_hash_t = decltype(std::declval<const T>().to_hash());

template <typename T>
constexpr bool supports_to_hash_v = std::experimental::is_detected_v<has_to_hash_t, T>;

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

template <typename T, typename = std::void_t<>>
struct is_std_hashable : std::false_type {};

template <typename T>
struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>>()(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

template <typename T, std::size_t N>
inline hash_t hash_object(const std::array<T, N>& array) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::array<{}, {}>: {}\n", short_type_name<std::decay_t<T>>, N, array);
    }
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
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::variant: {}\n", variant);
    }
    auto active_variant = variant.index();
    return std::visit([&](const auto& value) { return hash_objects(active_variant, value); }, variant);
}

template <typename... Ts>
inline hash_t hash_object(const std::reference_wrapper<Ts...>& reference) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::reference_wrapper: {}\n", reference.get());
    }
    return hash_object(reference.get());
}

template <typename T>
inline hash_t hash_object(const T& object) noexcept {
    if constexpr (std::numeric_limits<T>::is_integer) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing integer of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        return object;
    } else if constexpr (detail::is_std_hashable_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing {} using std::hash: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        return std::hash<T>{}(object);
    } else if constexpr (detail::supports_to_hash_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using to_hash method: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        return object.to_hash();
    } else if constexpr (detail::supports_compile_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print(
                "Hashing struct {} using compile-time attributes: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        constexpr auto num_attributes = detail::get_num_attributes<T>();
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
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::tuple of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
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
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::pair of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        hash_t hash = 0;
        return hash_objects(hash, object.first, object.second);
    } else if constexpr (is_specialization_v<T, std::vector>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::vector of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        hash_t hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_span_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::span of type {}\n", short_type_name<std::decay_t<T>>);
        }
        hash_t hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::set>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::set of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        hash_t hash = 0;
        for (const auto& element : object) {
            hash = hash_objects(hash, element);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::map>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::map of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        hash_t hash = 0;
        for (const auto& [key, value] : object) {
            hash = hash_objects(hash, key, value);
        }
        return hash;
    } else if constexpr (is_specialization_v<T, std::unordered_map>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::unordered_map of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
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
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing std::optional of type {}: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        if (object.has_value()) {
            return hash_object(object.value());
        }
        return 0;

    } else if constexpr (ttsl::concepts::Reflectable<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing struct {} using reflect library: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        hash_t hash = 0;
        reflect::for_each([&hash, &object](auto I) { hash = hash_objects(hash, reflect::get<I>(object)); }, object);
        return hash;
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

template <typename T, size_t PREALLOCATED_SIZE>
struct std::hash<ttsl::SmallVector<T, PREALLOCATED_SIZE>> {
    size_t operator()(const ttsl::SmallVector<T, PREALLOCATED_SIZE>& vec) const noexcept {
        size_t hash = 0;
        for (const auto& element : vec) {
            hash = ttsl::hash::detail::hash_objects(hash, element);
        }
        return hash;
    }
};

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
