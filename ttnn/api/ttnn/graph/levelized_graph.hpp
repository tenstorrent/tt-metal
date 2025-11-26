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

/*
LevelizedGraph allows extracting a hierarchical representation of traced operations at different stacking levels, which
helps analyze the operation call graph at varying levels of abstraction (particularly useful in generating IR from ttnn
input). A LevelizedGraph is a collection of vertices where each vertex is mostly identical to GraphProcessor::Vertex
with a couple of caveats:
    - LevelizedGraph::Vertex stores both out_edges and in_edges (as opposed to GraphProcessor::Vertex where we only
      store out_edges (or connections). This helps in constant lookup for incoming edges for a vertex (i.e. identifying
      input tensors for an op).
    - LevelizedGraph::Vertex stores the internals of a vertex. This is aligns perfectly with the hierarchical nature of
      LevelizedGraph. For instance, a ttnn::add node will have ttnn::prim::binary_ng as its internal.
    - LevelizedGraph::Vertex stores the output_shape of each node. This helps in constant lookup for the output shape of
      all nodes when building an IR from the trace. For instance, when ttnn::sum(input, 0, true) is applied on an input
      shaped [(64, 128)], it will store [(1, 128)] as the output_shape.
    - LevelizedGraph::Vertex stores the output_info of the nodes with a consumer. The output_info essentially tracks the
      layout information and sharding specs of vertices.
    - LevelizedGraph stores two types of vertices: 1) function_start nodes at or below max_level, and 2) tensor nodes
      at stacking_level 1 (input tensors). This allows explicit representation of input tensors in the graph. Other node
      types such as capture_start, capture_end, buffer, buffer_allocate, buffer_deallocate, circular_buffer_allocate,
      circular_buffer_deallocate_all, and intermediate tensor nodes are not stored in LevelizedGraph.
    - The edges of LevelizedGraph represent the data flow (as opposed to the edges of GraphProcessor which represent
      both data flow and parent-child relationship). This makes walks on LevelizedGraph much less complicated. Edges
      can be from tensor vertices to operation vertices (for input tensors) or from operation vertices to operation
      vertices (for intermediate results).
*/
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
        return graph | std::views::filter([level](const Vertex& vertex) { return vertex.stacking_level == level; });
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
