// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/base.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <enchantum/scoped.hpp>
#include <tt_stl/concepts.hpp>
#include <tt_stl/type_name.hpp>

// NOLINTBEGIN(bugprone-multi-level-implicit-pointer-conversion)

namespace ttsl::stl_fmt {

// Forward declaration of write_to helper.
// All operator<< overloads use this for recursive streaming to avoid ADL issues
// with types defined in other namespaces. Defined at the bottom of this namespace.
template <typename T>
std::ostream& write_to(std::ostream& os, const T& value);

// Container operator<< definitions

// Special case for bool to print true/false instead of 1/0
inline std::ostream& operator<<(std::ostream& os, bool value) {
    os << (value ? "true" : "false");
    return os;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& pair) {
    os << "{";
    write_to(os, pair.first);
    os << ", ";
    write_to(os, pair.second);
    os << "}";
    return os;
}

static std::ostream& operator<<(std::ostream& os, const std::filesystem::path& path) {
    os << path.c_str();
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& optional) {
    if (optional.has_value()) {
        write_to(os, optional.value());
    } else {
        os << "std::nullopt";
    }
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::variant<Ts...>& variant) {
    std::visit([&os](const auto& value) { write_to(os, value); }, variant);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::reference_wrapper<T> reference_wrapper) {
    write_to(os, reference_wrapper.get());
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& tuple) {
    [&os, &tuple]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        (
            [&os, &tuple]() {
                if constexpr (Ns != 0) {
                    os << ", ";
                }
                write_to(os, std::get<Ns>(tuple));
            }(),
            ...);
    }(std::make_index_sequence<sizeof...(Ts)>{});
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& array) {
    os << "{";
    for (auto index = 0; index < array.size(); index++) {
        const auto& element = array[index];
        write_to(os, element);
        if (index != array.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    os << "{";
    for (auto index = 0; index < vector.size(); index++) {
        const T& element = vector[index];
        write_to(os, element);
        if (index != vector.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
    os << "{";
    auto index = 0;
    for (const auto& element : set) {
        write_to(os, element);
        if (index != set.size() - 1) {
            os << ", ";
        }
        index++;
    }
    os << "}";
    return os;
}

template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::map<K, V>& map) {
    os << "{";
    for (auto it = map.begin(); it != map.end(); ++it) {
        write_to(os, it->first);
        os << ": ";
        write_to(os, it->second);
        if (it != map.end()) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<K, V>& map) {
    os << "{";
    for (auto it = map.begin(); it != map.end(); ++it) {
        write_to(os, it->first);
        os << ": ";
        write_to(os, it->second);
        if (it != map.end()) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template <typename T>
typename std::enable_if_t<std::is_enum_v<T>, std::ostream>& operator<<(std::ostream& os, const T& value) {
    os << enchantum::scoped::to_string(value);
    return os;
}

template <typename T>
typename std::enable_if_t<concepts::detail::supports_conversion_to_string_v<T>, std::ostream>& operator<<(
    std::ostream& os, const T& object) {
    if constexpr (concepts::detail::supports_to_string_v<T>) {
        os << object.to_string();
    } else if constexpr (concepts::detail::supports_compile_time_attributes_v<T>) {
        constexpr auto num_attributes = concepts::detail::get_num_attributes<T>();
        os << get_type_name<T>();
        os << "(";

        if constexpr (num_attributes > 0) {
            const auto attribute_values = object.attribute_values();
            [&os, &object, &attribute_values]<std::size_t... Ns>(std::index_sequence<Ns...>) {
                (
                    [&os, &object, &attribute_values] {
                        const auto& attribute = std::get<Ns>(attribute_values);
                        os << std::get<Ns>(object.attribute_names);
                        os << "=";
                        write_to(os, attribute);
                        os << ",";
                    }(),
                    ...);
            }(std::make_index_sequence<num_attributes - 1>{});

            const auto& attribute = std::get<num_attributes - 1>(attribute_values);
            os << std::get<num_attributes - 1>(object.attribute_names);
            os << "=";
            write_to(os, attribute);
        }

        os << ")";
    } else {
        static_assert(ttsl::concepts::always_false_v<T>, "Type cannot be converted to string");
    }
    return os;
}

// Definition of write_to helper. Defined after all operator<< overloads so that
// the using-declaration brings all of them into scope for dependent name lookup.
template <typename T>
std::ostream& write_to(std::ostream& os, const T& value) {
    if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
        // Special case for bool to print true/false instead of 1/0
        return operator<<(os, value);
    } else if constexpr (
        concepts::detail::supports_conversion_to_string_v<std::decay_t<T>> || std::is_enum_v<std::decay_t<T>>) {
        // Use our operator<< overloads for custom types and enums
        using ttsl::stl_fmt::operator<<;
        os << value;
    } else if constexpr (fmt::is_formattable<std::decay_t<T>>::value) {
        // Use fmt for types with their own fmt::formatter (CoreRangeSet, xy_pair, etc.)
        os << fmt::format("{}", value);
    } else {
        // Standard types (int, string, etc.) with built-in operator<<
        os << value;
    }
    return os;
}

}  // namespace ttsl::stl_fmt

// fmt formatters

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<ttsl::concepts::detail::supports_conversion_to_string_v<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& object, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << object;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<T, char, std::enable_if_t<std::is_enum_v<T>>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const T& value, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << value;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <>
struct fmt::formatter<std::filesystem::path> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::filesystem::path& path, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << path;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::optional<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::optional<T>& optional, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << optional;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename... Ts>
struct fmt::formatter<std::variant<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::variant<Ts...>& variant, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << variant;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::reference_wrapper<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::reference_wrapper<T> reference, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << reference;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename... Ts>
struct fmt::formatter<std::tuple<Ts...>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::tuple<Ts...>& tuple, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << tuple;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T, std::size_t N>
struct fmt::formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::array<T, N>& array, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << array;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::vector<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::vector<T>& vector, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << vector;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename T>
struct fmt::formatter<std::set<T>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::set<T>& set, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << set;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename K, typename V>
struct fmt::formatter<std::map<K, V>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::map<K, V>& map, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << map;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

template <typename K, typename V>
struct fmt::formatter<std::unordered_map<K, V>> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const std::unordered_map<K, V>& map, format_context& ctx) const -> format_context::iterator {
        using ttsl::stl_fmt::operator<<;
        std::stringstream ss;
        ss << map;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};

// NOLINTEND(bugprone-multi-level-implicit-pointer-conversion)
