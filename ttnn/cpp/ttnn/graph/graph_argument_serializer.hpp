// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <any>
#include <functional>
#include <span>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace ttnn::graph {
std::string graph_demangle(const char* name);

struct GraphArgumentSerializer {
    using ConvertionFunction = std::function<std::string(const std::any&)>;

    static std::unordered_map<std::type_index, ConvertionFunction>& registry();

    template <typename T, std::size_t N>
    static void register_small_vector();

    // In case you don't care about all the variations of the type
    // such as const T, const T&, T, T&, etc
    template <typename T>
    static void register_special_type();

    template <typename T>
    static void register_type();

    template <typename OptionalT>
    static void register_optional_type();

    static std::vector<std::string> to_list(const std::span<std::any>& span);

    static void initialize();
};
}  // namespace ttnn::graph
