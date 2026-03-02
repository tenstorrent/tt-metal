// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lightweight fmt formatters for standard containers, enums, and project types.
//
// Include this header instead of <tt_stl/reflection.hpp> when all you need is
// to format standard-library types (vector, array, optional, variant, map, …)
// or enums in log / fmt::format calls.  This header deliberately avoids the
// heavy <reflect> and <nlohmann/json.hpp> includes that reflection.hpp pulls in.
//
// Why not use fmt/std.h and fmt/ranges.h directly?
//
//   1. fmt/ranges.h detects any type with begin()/end() as "range-like".
//      Many project types (Shape, MeshShape, etc.) are both range-like AND use
//      the compile-time attribute protocol (attribute_names + attribute_values),
//      which has its own formatter in reflection.hpp.  Including fmt/ranges.h
//      creates ambiguous specializations.
//
//   2. fmt/std.h's std::optional<T> formatter requires is_formattable<T>.
//      The Attribute class type-erases over types like optional<const bool>,
//      and fmt does not consider `const bool` formattable.
//
//   3. fmt 11 disallows non-void pointer formatting via a static_assert in
//      make_arg, before any custom formatter is consulted.  Containers like
//      vector<IDevice*> require special handling at the container level.

#pragma once

#include <fmt/format.h>

#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <enchantum/scoped.hpp>
#include <tt_stl/small_vector.hpp>

// ============================================================================
// Implementation detail – format a single value, with pointer handling.
//
// fmt 11 rejects non-void pointers with a static_assert inside make_arg,
// which fires before the formatter is ever consulted.  We work around this
// by detecting pointer types at the call site and converting to void*.
// ============================================================================

namespace ttsl::fmt_detail {

template <typename T>
auto format_value(fmt::format_context::iterator out, const T& val) -> fmt::format_context::iterator {
    if constexpr (
        std::is_pointer_v<T> && !std::is_same_v<std::remove_const_t<std::remove_pointer_t<T>>, char> &&
        !std::is_same_v<std::remove_const_t<std::remove_pointer_t<T>>, void>) {
        if (val) {
            return fmt::format_to(out, "{}", fmt::ptr(val));
        }
        return fmt::format_to(out, "nullptr");
    } else {
        return fmt::format_to(out, "{}", val);
    }
}

// Helper to format a sequence of values with a given open/close bracket.
template <typename Iter>
auto format_sequence(
    Iter begin, Iter end, fmt::format_context::iterator out, std::string_view open, std::string_view close)
    -> fmt::format_context::iterator {
    out = fmt::format_to(out, "{}", open);
    bool first = true;
    for (auto it = begin; it != end; ++it) {
        if (!first) {
            out = fmt::format_to(out, ", ");
        }
        out = format_value(out, *it);
        first = false;
    }
    return fmt::format_to(out, "{}", close);
}

}  // namespace ttsl::fmt_detail

// ============================================================================
// fmt::formatter – enums (via enchantum, avoids <reflect>)
// ============================================================================

template <typename T>
    requires std::is_enum_v<T>
struct fmt::formatter<T> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& value, format_context& ctx) const -> format_context::iterator {
        return fmt::format_to(ctx.out(), "{}", enchantum::scoped::to_string(value));
    }
};

// ============================================================================
// fmt::formatter – std::filesystem::path
// ============================================================================

template <>
struct fmt::formatter<std::filesystem::path> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::filesystem::path& p, format_context& ctx) const -> format_context::iterator {
        return fmt::format_to(ctx.out(), "{}", p.string());
    }
};

// ============================================================================
// fmt::formatter – std::optional
// ============================================================================

template <typename T>
struct fmt::formatter<std::optional<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::optional<T>& opt, format_context& ctx) const -> format_context::iterator {
        if (opt.has_value()) {
            return ttsl::fmt_detail::format_value(ctx.out(), opt.value());
        }
        return fmt::format_to(ctx.out(), "std::nullopt");
    }
};

// ============================================================================
// fmt::formatter – std::variant
// ============================================================================

template <typename... Ts>
struct fmt::formatter<std::variant<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::variant<Ts...>& var, format_context& ctx) const -> format_context::iterator {
        return std::visit([&ctx](const auto& val) { return ttsl::fmt_detail::format_value(ctx.out(), val); }, var);
    }
};

// ============================================================================
// fmt::formatter – std::reference_wrapper
// ============================================================================

template <typename T>
struct fmt::formatter<std::reference_wrapper<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::reference_wrapper<T>& ref, format_context& ctx) const -> format_context::iterator {
        return ttsl::fmt_detail::format_value(ctx.out(), ref.get());
    }
};

// ============================================================================
// fmt::formatter – standard containers
// ============================================================================

// std::vector
template <typename T, typename Alloc>
struct fmt::formatter<std::vector<T, Alloc>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::vector<T, Alloc>& vec, format_context& ctx) const -> format_context::iterator {
        return ttsl::fmt_detail::format_sequence(vec.begin(), vec.end(), ctx.out(), "{", "}");
    }
};

// std::array
template <typename T, std::size_t N>
struct fmt::formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::array<T, N>& arr, format_context& ctx) const -> format_context::iterator {
        return ttsl::fmt_detail::format_sequence(arr.begin(), arr.end(), ctx.out(), "{", "}");
    }
};

// std::set
template <typename Key, typename Compare, typename Alloc>
struct fmt::formatter<std::set<Key, Compare, Alloc>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::set<Key, Compare, Alloc>& s, format_context& ctx) const -> format_context::iterator {
        return ttsl::fmt_detail::format_sequence(s.begin(), s.end(), ctx.out(), "{", "}");
    }
};

// std::map
template <typename Key, typename Value, typename Compare, typename Alloc>
struct fmt::formatter<std::map<Key, Value, Compare, Alloc>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::map<Key, Value, Compare, Alloc>& m, format_context& ctx) const -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "{{");
        bool first = true;
        for (const auto& [k, v] : m) {
            if (!first) {
                out = fmt::format_to(out, ", ");
            }
            out = ttsl::fmt_detail::format_value(out, k);
            out = fmt::format_to(out, ": ");
            out = ttsl::fmt_detail::format_value(out, v);
            first = false;
        }
        return fmt::format_to(out, "}}");
    }
};

// std::unordered_map
template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct fmt::formatter<std::unordered_map<Key, Value, Hash, Equal, Alloc>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::unordered_map<Key, Value, Hash, Equal, Alloc>& m, format_context& ctx) const
        -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "{{");
        bool first = true;
        for (const auto& [k, v] : m) {
            if (!first) {
                out = fmt::format_to(out, ", ");
            }
            out = ttsl::fmt_detail::format_value(out, k);
            out = fmt::format_to(out, ": ");
            out = ttsl::fmt_detail::format_value(out, v);
            first = false;
        }
        return fmt::format_to(out, "}}");
    }
};

// std::pair
template <typename T1, typename T2>
struct fmt::formatter<std::pair<T1, T2>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::pair<T1, T2>& p, format_context& ctx) const -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "(");
        out = ttsl::fmt_detail::format_value(out, p.first);
        out = fmt::format_to(out, ", ");
        out = ttsl::fmt_detail::format_value(out, p.second);
        return fmt::format_to(out, ")");
    }
};

// std::tuple
template <typename... Ts>
struct fmt::formatter<std::tuple<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::tuple<Ts...>& t, format_context& ctx) const -> format_context::iterator {
        auto out = fmt::format_to(ctx.out(), "(");
        std::apply(
            [&out, first = true](const auto&... args) mutable {
                ((out =
                      (first ? out : fmt::format_to(out, ", "),
                       ttsl::fmt_detail::format_value(out, args),
                       first = false,
                       out)),
                 ...);
            },
            t);
        return fmt::format_to(out, ")");
    }
};

// ============================================================================
// fmt::formatter – ttsl::SmallVector
// ============================================================================

template <typename T, std::size_t PREALLOCATED_SIZE>
struct fmt::formatter<ttsl::SmallVector<T, PREALLOCATED_SIZE>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const ttsl::SmallVector<T, PREALLOCATED_SIZE>& vec, format_context& ctx) const
        -> format_context::iterator {
        return ttsl::fmt_detail::format_sequence(vec.begin(), vec.end(), ctx.out(), "{", "}");
    }
};
