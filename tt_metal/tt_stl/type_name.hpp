// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <source_location>
#include <string_view>

namespace tt::stl {

namespace detail {

// Helper function to retrieve the raw function name that includes the type name T.
template <typename T>
constexpr std::string_view type_to_string_raw() {
    const std::source_location location = std::source_location::current();
    return location.function_name();
}

// We need a reference type to use to extract positions of type names within the raw string.
// Here, we use `long double`.
constexpr std::string_view long_double_string = "long double";
constexpr std::string_view long_double_raw_string = type_to_string_raw<long double>();

// Find the position of "long double" within the raw string. Gives us a starting point for extracting type names.
constexpr std::size_t begin_type_name = long_double_raw_string.find(long_double_string);
static_assert(begin_type_name != std::string_view::npos);

// Calculate the end position of "long double" within the raw string.
constexpr std::size_t end_type_name = begin_type_name + long_double_string.size();
static_assert(begin_type_name != std::string_view::npos);

// Determine the size of the suffix after the type name in the raw string, to trim off the unwanted parts.
constexpr std::size_t suffix_type_name_size = long_double_raw_string.size() - end_type_name;

// Function to get a more human-readable type name by trimming the raw string.
template <typename T>
constexpr std::string_view long_name() {
    std::string_view raw_name = type_to_string_raw<T>();
    auto size = raw_name.size();
    raw_name.remove_prefix(begin_type_name);
    raw_name.remove_suffix(suffix_type_name_size);
    std::string_view struct_name("struct ");
    std::string_view class_name("class ");

    if (raw_name.substr(0, struct_name.size()) == struct_name) {
        raw_name.remove_prefix(struct_name.size());
    }
    if (raw_name.substr(0, class_name.size()) == class_name) {
        raw_name.remove_prefix(class_name.size());
    }

    while (!raw_name.empty() && raw_name.back() == ' ') {
        raw_name.remove_suffix(1);
    }
    return raw_name;
}

// Function to get the "short" type name, which is typically the part after any namespace or class scope qualifications.
template <typename T>
constexpr std::string_view short_name() {
    auto raw_str = long_name<T>();
    int last = -1;  // Position of the last ':' found at the top level (i.e., not inside template arguments).
    int count = 0;  // Counter to track nesting levels of template arguments.

    // Iterate through the string to find the last top-level ':'.
    for (std::size_t pos = 0; pos < raw_str.size(); ++pos) {
        auto& c = raw_str[pos];
        if (c == '<') {
            ++count;  // Increment count when entering a nested template argument.
        }
        if (c == '>') {
            --count;  // Decrement count when leaving a nested template argument.
        }
        if (c == ':' && count == 0) {
            last = pos;  // Update the last position of ':' if it's at the top level.
        }
    }
    if (last != -1) {
        raw_str.remove_prefix(last + 1);
    }
    return raw_str;
}

}  // namespace detail

template <typename T>
inline constexpr std::string_view short_type_name = detail::short_name<T>();

template <typename T>
inline constexpr std::string_view long_type_name = detail::long_name<T>();

}  // namespace tt::stl
