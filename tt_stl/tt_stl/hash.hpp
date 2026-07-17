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
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <reflect>

#include <tt_stl/concepts.hpp>
#include <tt_stl/reflection_detail/reflection_traits.hpp>
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

template <typename... Types>
std::string canonical_key(const Types&... args);

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

// Detection of a `to_hash()` customization method. (The compile-time attribute
// protocol detectors live in <tt_stl/reflection_detail/reflection_traits.hpp>.)
template <typename T>
using has_to_hash_t = decltype(std::declval<const T>().to_hash());

template <typename T>
constexpr bool supports_to_hash_v = std::experimental::is_detected_v<has_to_hash_t, T>;

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
    } else if constexpr (ttsl::reflection::detail::supports_compile_time_attributes_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print(
                "Hashing struct {} using compile-time attributes: {}\n", short_type_name<std::decay_t<T>>, object);
        }
        constexpr auto num_attributes = ttsl::reflection::detail::get_num_attributes<T>();
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

// Fold one value's hash into a running seed with the splitmix64 finalizer (David Stafford's
// "variant 13"), a strong 64-bit mixer. Shared by every combiner in this header (hash_objects and
// hash_combine) so their mixing behavior is identical.
//
// The previous combiner was the classic boost::hash_combine
// (`seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2)`). Its avalanche is poor for the small,
// structured integers that dominate our cache keys (tensor shapes, dtypes), so distinct sequences
// collided in 64 bits -- e.g. shapes [3, 17, 1, 1] and [1, 152, 1, 1] hashed identically, causing
// wrong program-cache hits (issue #45821).
//
// The multiply-xorshift finalizer is the state of the art for 64-bit hash mixing (boost >=1.81 and
// abseil use mixers of the same family, with different constants); we inline it with only <cstdint>
// arithmetic so tt_stl pulls in no extra dependency. 0x9e3779b97f4a7c15 is the 64-bit golden-ratio
// increment (the 64-bit analog of the old 0x9e3779b9), which keeps the fold order-dependent and
// gives a non-trivial result for an all-zero input.
inline hash_t mix_into(hash_t seed, hash_t value_hash) noexcept {
    hash_t x = seed + 0x9e3779b97f4a7c15ULL + value_hash;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

template <typename... Types>
inline hash_t hash_objects(hash_t seed, const Types&... args) noexcept {
    ([&seed](const auto& arg) { seed = mix_into(seed, hash_object(arg)); }(args), ...);
    return seed;
}

// ---------------------------------------------------------------------------------------------
// Canonical key encoding (collision-free companion to hash_objects).
//
// hash_objects folds a key down to 64 bits, which can collide (issue #45821). For a cache that
// must NEVER return a wrong entry, the 64-bit hash selects a bucket and an EXACT comparison of
// the key resolves collisions -- the textbook hash-map contract. append_canonical builds that
// exact key: a byte string whose traversal mirrors hash_object branch-for-branch, so it
// distinguishes precisely the inputs the hash combines (same coverage => no spurious misses, no
// missed collisions) and contains no volatile data (addresses, buffers) -- only the structural
// values, just like the hash.
//
// Encoding is exact for every type on the op-key path: integers, enums, floating point,
// std::string, the compile-time-attribute / reflect aggregates, and the standard containers
// (length-prefixed; unordered_map sorted by key to stay order-invariant, mirroring hash_object).
// The one lossy leaf is a type exposing ONLY to_hash(): its 8 hash bytes are appended, so for
// such a type equality degrades to hash equality -- no worse than today, and none occur on the
// tensor/shape path (TensorSpec is walked structurally down to the shape integers).
inline void append_bytes(std::string& out, const void* p, std::size_t n) { out.append(static_cast<const char*>(p), n); }

template <typename T>
inline void append_canonical(std::string& out, const T& object);

template <typename... Types>
inline void append_canonical_all(std::string& out, const Types&... args) {
    (append_canonical(out, args), ...);
}

template <typename T>
inline void append_canonical(std::string& out, const T& object) {
    out.push_back('\x1f');  // unit separator: disambiguates adjacent leaves/fields
    if constexpr (std::numeric_limits<T>::is_integer) {
        append_bytes(out, &object, sizeof(object));
    } else if constexpr (std::is_enum_v<T>) {
        const auto v = static_cast<std::underlying_type_t<T>>(object);
        append_bytes(out, &v, sizeof(v));
    } else if constexpr (std::is_floating_point_v<T>) {
        append_bytes(out, &object, sizeof(object));
    } else if constexpr (std::is_same_v<T, std::string>) {
        const std::uint64_t n = object.size();
        append_bytes(out, &n, sizeof(n));
        append_bytes(out, object.data(), object.size());
    } else if constexpr (ttsl::reflection::detail::supports_compile_time_attributes_v<T>) {
        std::apply([&out](const auto&... a) { (append_canonical(out, a), ...); }, object.attribute_values());
    } else if constexpr (is_specialization_v<T, std::tuple>) {
        std::apply([&out](const auto&... a) { (append_canonical(out, a), ...); }, object);
    } else if constexpr (is_specialization_v<T, std::pair>) {
        append_canonical(out, object.first);
        append_canonical(out, object.second);
    } else if constexpr (is_specialization_v<T, std::optional>) {
        const char has = object.has_value() ? 1 : 0;
        out.push_back(has);
        if (object.has_value()) {
            append_canonical(out, object.value());
        }
    } else if constexpr (is_specialization_v<T, std::variant>) {
        const std::uint64_t index = object.index();
        append_bytes(out, &index, sizeof(index));
        std::visit([&out](const auto& value) { append_canonical(out, value); }, object);
    } else if constexpr (is_specialization_v<T, std::reference_wrapper>) {
        append_canonical(out, object.get());
    } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
        // std::vector<bool> is a bit-packed specialization: iterating yields a proxy reference
        // (not bool&), so it can't go through the generic vector branch.
        // (hash_object never reaches its own vector branch for this type because
        // std::hash<std::vector<bool>> exists and the is_std_hashable_v branch wins first.)
        const std::uint64_t n = object.size();
        append_bytes(out, &n, sizeof(n));
        for (bool element : object) {
            append_canonical(out, element);
        }
    } else if constexpr (is_specialization_v<T, std::vector> || is_specialization_v<T, std::set> || is_span_v<T>) {
        const std::uint64_t n = object.size();
        append_bytes(out, &n, sizeof(n));
        for (const auto& element : object) {
            append_canonical(out, element);
        }
    } else if constexpr (is_specialization_v<T, std::map>) {
        const std::uint64_t n = object.size();
        append_bytes(out, &n, sizeof(n));
        for (const auto& [key, value] : object) {
            append_canonical(out, key);
            append_canonical(out, value);
        }
    } else if constexpr (is_specialization_v<T, std::unordered_map>) {
        // Sort by key so the encoding is order-invariant, mirroring hash_object.
        std::vector<typename T::const_iterator> iterators;
        iterators.reserve(object.size());
        for (auto it = object.begin(); it != object.end(); ++it) {
            iterators.push_back(it);
        }
        std::sort(iterators.begin(), iterators.end(), [](const auto& a, const auto& b) { return a->first < b->first; });
        const std::uint64_t n = object.size();
        append_bytes(out, &n, sizeof(n));
        for (const auto& it : iterators) {
            append_canonical(out, it->first);
            append_canonical(out, it->second);
        }
    } else if constexpr (ttsl::concepts::Reflectable<T>) {
        reflect::for_each([&out, &object](auto I) { append_canonical(out, reflect::get<I>(object)); }, object);
    } else if constexpr (detail::supports_to_hash_v<T>) {
        const hash_t h = object.to_hash();  // lossy leaf (see note above)
        append_bytes(out, &h, sizeof(h));
    } else if constexpr (detail::is_std_hashable_v<T>) {
        const std::size_t h = std::hash<T>{}(object);  // lossy fallback
        append_bytes(out, &h, sizeof(h));
    } else {
        static_assert(ttsl::concepts::always_false_v<T>, "Type doesn't support ttsl::hash::canonical_key");
    }
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

// Exact, collision-free encoding of `args` for use as a program-cache key alongside the 64-bit
// hash: two argument packs produce the same string iff the hash traversal cannot distinguish
// them. See detail::append_canonical.
template <typename... Types>
inline std::string canonical_key(const Types&... args) {
    std::string out;
    detail::append_canonical_all(out, args...);
    return out;
}

// std::hash-based combiner used by the hand-written program-descriptor hashers (generic_op,
// program_descriptors, fd_kernel, ...). Uses the same strong mixer as hash_objects so these
// program-cache-relevant hashes get the same collision resistance (issue #45821).
template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    std::hash<T> hasher;
    seed = static_cast<std::size_t>(
        detail::mix_into(static_cast<ttsl::hash::hash_t>(seed), static_cast<ttsl::hash::hash_t>(hasher(value))));
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
