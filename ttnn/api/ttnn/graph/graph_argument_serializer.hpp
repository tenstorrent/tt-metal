// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <any>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <typeindex>
#include <unordered_map>

namespace ttnn::graph {
std::string graph_demangle(std::string_view name);

class GraphArgumentSerializer {
public:
    using ConvertionFunction = std::function<std::string(const std::any&)>;
    std::vector<std::string> to_list(const std::span<std::any>& span);

    static GraphArgumentSerializer& instance();

private:
    GraphArgumentSerializer();
    std::unordered_map<std::type_index, ConvertionFunction>& registry();

    template <typename T, std::size_t N>
    void register_small_vector();

    template <typename T, std::size_t N>
    void register_array();

    template <typename T>
    void register_vector();

    template <typename T>
    void register_type();

    template <typename OptionalT>
    void register_optional_type();

    template <typename T>
    void register_optional_reference_type();

    void initialize();

    std::unordered_map<std::type_index, GraphArgumentSerializer::ConvertionFunction> map;
};
}  // namespace ttnn::graph
