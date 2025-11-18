// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <string>
#include <vector>

namespace ttnn::graph {

class LevelizedGraph {
public:
    using VertexID = std::size_t;
    struct Vertex {
        VertexID id = 0;
        std::size_t stacking_level = 0;
        std::string name;
        std::vector<std::string> arguments;
        std::vector<VertexID> in_edges;
        std::vector<VertexID> out_edges;
        std::vector<VertexID> internals;
        std::vector<std::string> output_info;
        std::vector<std::string> output_shape;
    };
    using Graph = std::vector<Vertex>;

    LevelizedGraph(const nlohmann::json& trace, std::size_t max_level = 1);

    bool is_valid_vertex(std::size_t id) const;
    const Graph& vertices() const;
    const Vertex& get_vertex(VertexID id) const;
    std::size_t size() const;
    nlohmann::json to_json() const;
    auto get_vertices_at_level(std::size_t level) const {
        return graph | std::views::filter([&](const Vertex& vertex) { return vertex.stacking_level == level; });
    }

private:
    void init(const nlohmann::json& trace, std::size_t max_level);
    // Functions that are used to populate the graph from the trace.
    void populate_vertices(const nlohmann::json& trace, std::size_t max_level);
    void populate_fn_start_to_end_maps(const nlohmann::json& trace);
    void populate_tensor_to_producer_end_map(const nlohmann::json& trace);
    void populate_edges(const nlohmann::json& trace);
    void populate_internals(const nlohmann::json& trace, std::size_t max_level);
    void populate_output_info();
    void populate_output_shape(const nlohmann::json& trace);
    void populate_input_connections(const nlohmann::json& trace);

    // Maps to track the mapping between the node counter of the original graph and the id of the levelized graph
    std::unordered_map<int, size_t> id_map;
    std::unordered_map<size_t, int> reverse_id_map;

    // Maps to track data flow
    std::unordered_map<int, int> tensor_to_producer_end;  // tensor_counter -> function_end_counter that produced it
    std::unordered_map<int, int> function_end_to_start;   // function_end_counter -> function_start_counter
    std::unordered_map<int, int> function_start_to_end;   // function_start_counter -> function_end_counter

    Graph graph;
};

}  // namespace ttnn::graph
