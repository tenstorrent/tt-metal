// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/levelized_graph.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include <nlohmann/json.hpp>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace {

nlohmann::json to_json(const ttnn::graph::LevelizedGraph::Vertex& data) {
    nlohmann::json j;
    j[ttnn::graph::kCounter] = data.id;
    j[ttnn::graph::kStackingLevel] = data.stacking_level;
    j[ttnn::graph::kName] = data.name;
    j[ttnn::graph::kArguments] = data.arguments;
    j[ttnn::graph::kInEdges] = data.in_edges;
    j[ttnn::graph::kOutEdges] = data.out_edges;
    j[ttnn::graph::kInternals] = data.internals;
    j[ttnn::graph::kOutputInfo] = data.output_info;
    j[ttnn::graph::kOutputShape] = data.output_shape;
    return j;
}

nlohmann::json to_json(const ttnn::graph::LevelizedGraph::Graph& graph) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& vertex : graph) {
        j.push_back(to_json(vertex));
    }
    return j;
}

}  // namespace

namespace ttnn::graph {

// -----------------------------------------------------------------------------------------------------------------
// Public APIs:
// -----------------------------------------------------------------------------------------------------------------
LevelizedGraph::LevelizedGraph(const nlohmann::json& trace, std::size_t max_level) { init(trace, max_level); }
nlohmann::json LevelizedGraph::to_json() const { return ::to_json(graph); }
bool LevelizedGraph::is_valid_vertex(std::size_t id) const { return id < graph.size(); }
std::size_t LevelizedGraph::size() const { return graph.size(); }
const LevelizedGraph::Graph& LevelizedGraph::vertices() const { return graph; }
const LevelizedGraph::Vertex& LevelizedGraph::get_vertex(std::size_t id) const {
    TT_ASSERT(is_valid_vertex(id));
    return graph[id];
}

// -----------------------------------------------------------------------------------------------------------------
// Helper functions:
// -----------------------------------------------------------------------------------------------------------------
std::vector<int> get_connections(const nlohmann::json& node) {
    if (!node.contains(kConnections)) {
        return {};
    }
    return node[kConnections].get<std::vector<int>>();
}

auto get_valid_connections(const nlohmann::json& node, const nlohmann::json& trace) {
    return get_connections(node) | std::views::filter([&](const auto& conn_counter) {
               return conn_counter >= 0 && static_cast<size_t>(conn_counter) < trace.size();
           });
}

// -----------------------------------------------------------------------------------------------------------------
// Private APIs:
// -----------------------------------------------------------------------------------------------------------------

void LevelizedGraph::init(const nlohmann::json& trace, std::size_t max_level) {
    populate_vertices(trace, max_level);
    populate_fn_start_to_end_maps(trace);
    populate_tensor_to_producer_end_map(trace);
    populate_edges(trace);
    populate_internals(trace, max_level);
    populate_output_info();
    populate_output_shape(trace);
    populate_input_connections(trace);
}

// Populates the vertices of the graph from the trace. We populate:
// 1. function start nodes that are at or below the max level
// 2. tensor nodes at stacking_level 1 (input tensors)
void LevelizedGraph::populate_vertices(const nlohmann::json& trace, std::size_t max_level) {
    // Verify that counters are sequential (counter == index)
    auto get_counter = [](const auto& node) { return node[kCounter].template get<int>(); };
    TT_ASSERT(std::ranges::equal(trace | std::views::transform(get_counter), std::views::iota(0u, trace.size())));

    // Process tensor nodes at stacking_level 1
    auto input_tensors = trace | std::views::filter([&](const auto& node) {
                             return node[kNodeType] == kNodeTensor && node[kStackingLevel].template get<size_t>() == 1;
                         });
    std::ranges::for_each(input_tensors, [&](const auto& node) {
        // Create a new vertex for the tensor
        Vertex vertex;
        vertex.id = graph.size();
        vertex.name = kNodeTensor;
        vertex.stacking_level = 1;

        // Extract tensor_id and shape from params
        if (node.contains(kParams)) {
            if (node[kParams].contains(kTensorId)) {
                // override tensor name to include tensor_id:
                vertex.name =
                    std::string(kNodeTensor) + "[" + node[kParams][kTensorId].template get<std::string>() + "]";
            }
            if (node[kParams].contains(kShape)) {
                vertex.output_shape.push_back(node[kParams][kShape].template get<std::string>());
            }
        }

        int counter = node[kCounter].template get<int>();
        id_map[counter] = vertex.id;
        reverse_id_map[vertex.id] = counter;

        graph.push_back(vertex);
    });

    // Process function_start nodes
    auto valid_starts =
        trace | std::views::filter([&](const auto& node) {
            return node[kNodeType] == kNodeFunctionStart && node[kStackingLevel].template get<size_t>() <= max_level;
        });
    std::ranges::for_each(valid_starts, [&](const auto& node) {
        // Create a new vertex
        Vertex vertex;
        vertex.id = graph.size();
        vertex.name = node[kParams][kName].template get<std::string>();
        vertex.stacking_level = node[kStackingLevel].template get<size_t>();

        // Extract arguments if they exist
        if (node.contains(kArguments) && node[kArguments].is_array()) {
            auto string_args = node[kArguments] | std::views::filter([](const auto& arg) { return arg.is_string(); }) |
                               std::views::transform([](const auto& arg) { return arg.template get<std::string>(); });
            vertex.arguments.assign(string_args.begin(), string_args.end());
        }

        int counter = node[kCounter].template get<int>();
        id_map[counter] = vertex.id;
        reverse_id_map[vertex.id] = counter;

        graph.push_back(vertex);
    });
}

// Map function_end to function_start (function_start has function_end in its connections)
void LevelizedGraph::populate_fn_start_to_end_maps(const nlohmann::json& trace) {
    auto starts = trace | std::views::filter([&](const auto& node) { return node[kNodeType] == kNodeFunctionStart; });
    std::ranges::for_each(starts, [&](const auto& node) {
        int start_counter = node[kCounter].template get<int>();
        auto ends = get_valid_connections(node, trace) | std::views::filter([&](const auto& neighbor_counter) {
                        return trace[neighbor_counter][kNodeType] == kNodeFunctionEnd &&
                               trace[neighbor_counter][kParams][kName] == node[kParams][kName];
                    });
        TT_ASSERT(std::ranges::distance(ends) >= 1);
        int end_counter = *ends.begin();
        function_start_to_end[start_counter] = end_counter;
        function_end_to_start[end_counter] = start_counter;
    });
}

// Map output tensors to function_end that produced them The correct producer is the function_end at the LOWEST stacking
// level (outermost operation) that has the tensor in its connections. This ensures we get the actual producer, not an
// intermediate nested operation.
void LevelizedGraph::populate_tensor_to_producer_end_map(const nlohmann::json& trace) {
    auto fn_ends = trace | std::views::filter([&](const auto& node) { return node[kNodeType] == kNodeFunctionEnd; });
    std::ranges::for_each(fn_ends, [&](const auto& node) {
        int end_counter = node[kCounter].template get<int>();
        size_t end_level = node[kStackingLevel].template get<size_t>();
        auto tensor_counters = get_valid_connections(node, trace) | std::views::filter([&](const auto& conn_counter) {
                                   return trace[conn_counter][kNodeType] == kNodeTensor;
                               });
        std::ranges::for_each(tensor_counters, [&](const auto& tensor_counter) {
            // Map tensor to this function_end if:
            // 1. Not already mapped, OR
            // 2. This function_end is at a lower stacking level (outermost operation)
            auto existing_it = tensor_to_producer_end.find(tensor_counter);
            if (existing_it == tensor_to_producer_end.end()) {
                tensor_to_producer_end[tensor_counter] = end_counter;
            } else {
                // Check if this function_end is at a lower level (more outer)
                int existing_end_counter = existing_it->second;
                if (existing_end_counter >= 0 && static_cast<size_t>(existing_end_counter) < trace.size()) {
                    const auto& existing_end_node = trace[existing_end_counter];
                    size_t existing_level = existing_end_node[kStackingLevel].get<size_t>();
                    if (end_level < existing_level) {
                        // This function_end is more outer, use it
                        tensor_to_producer_end[tensor_counter] = end_counter;
                    }
                }
            }
        });
    });
}

// Third pass: Build edges based on actual tensor flow
void LevelizedGraph::populate_edges(const nlohmann::json& trace) {
    auto vertex_ids = std::views::iota(0u, graph.size());
    std::ranges::for_each(vertex_ids, [&](size_t vertex_id) {
        int node_counter = reverse_id_map[vertex_id];
        const auto& node = trace[node_counter];

        // Handle edges differently based on node type
        if (node[kNodeType] == kNodeTensor) {
            // For tensor nodes, create edges to all operations that consume this tensor
            auto connections = get_connections(node);
            for (int conn_counter : connections) {
                // Check if the connection is a function_start in our graph
                auto consumer_id_it = id_map.find(conn_counter);
                if (consumer_id_it != id_map.end()) {
                    size_t consumer_vertex_id = consumer_id_it->second;
                    // Add edge from tensor to consumer
                    auto& tensor_out_edges = graph[vertex_id].out_edges;
                    if (std::ranges::find(tensor_out_edges, consumer_vertex_id) == tensor_out_edges.end()) {
                        tensor_out_edges.push_back(consumer_vertex_id);
                    }
                }
            }
        } else {
            // For function_start nodes, find input tensors and their producers
            int function_start_counter = node_counter;

            // Find input tensors: tensors that have connections to this function_start
            // This represents tensors that are used as inputs to this operation
            auto input_tensors =
                trace | std::views::filter([&](const auto& node) { return node[kNodeType] == kNodeTensor; }) |
                std::views::filter([&](const auto& node) {
                    auto connections = get_connections(node);
                    return std::ranges::find(connections, function_start_counter) != connections.end();
                }) |
                std::views::transform([&](const auto& node) { return node[kCounter].template get<int>(); });

            // For each input tensor, find the producer operation (or the tensor vertex itself)
            auto fn = [&](int tensor_counter) {
                // Check if this tensor is in our levelized graph (level 1 tensor)
                auto tensor_vertex_it = id_map.find(tensor_counter);
                if (tensor_vertex_it != id_map.end()) {
                    // This is a level 1 tensor vertex, create edge from it to this operation
                    size_t tensor_vertex_id = tensor_vertex_it->second;
                    auto& tensor_out_edges = graph[tensor_vertex_id].out_edges;
                    if (std::ranges::find(tensor_out_edges, vertex_id) == tensor_out_edges.end()) {
                        tensor_out_edges.push_back(vertex_id);
                    }
                } else {
                    // This tensor is not in the graph, find its producer operation
                    auto it = tensor_to_producer_end.find(tensor_counter);
                    if (it != tensor_to_producer_end.end()) {
                        int producer_end = it->second;
                        auto end_to_start_it = function_end_to_start.find(producer_end);
                        if (end_to_start_it != function_end_to_start.end()) {
                            int producer_start = end_to_start_it->second;
                            // Check if producer is in our levelized graph
                            auto producer_id_it = id_map.find(producer_start);
                            if (producer_id_it != id_map.end()) {
                                size_t producer_vertex_id = producer_id_it->second;
                                // Add edge from producer to consumer (data flow direction)
                                // Avoid self-loops and duplicate edges
                                auto& producer_connections = graph[producer_vertex_id].out_edges;
                                if (producer_vertex_id != vertex_id &&
                                    std::ranges::find(producer_connections, vertex_id) == producer_connections.end()) {
                                    producer_connections.push_back(vertex_id);
                                }
                            }
                        }
                    }
                }
            };
            std::ranges::for_each(input_tensors, fn);
        }
    });
}

// Fourth pass: Populate internals (children at stacking_level + 1)
void LevelizedGraph::populate_internals(const nlohmann::json& trace, std::size_t max_level) {
    auto vertex_ids = std::views::iota(0u, graph.size());
    std::ranges::for_each(vertex_ids, [&](size_t vertex_id) {
        int node_counter = reverse_id_map[vertex_id];
        const auto& node = trace[node_counter];

        // Skip tensor nodes - they don't have internals
        if (node[kNodeType] == kNodeTensor) {
            return;
        }

        int function_start_counter = node_counter;
        size_t parent_level = graph[vertex_id].stacking_level;

        // Find the function_end for this function_start
        auto start_to_end_it = function_start_to_end.find(function_start_counter);
        if (start_to_end_it == function_start_to_end.end()) {
            return;  // No function_end found, skip
        }
        int function_end_counter = start_to_end_it->second;

        // Find all function_start nodes that are children (between function_start and function_end)
        // and have stacking_level = parent_level + 1
        auto children = trace |
                        std::views::filter([&](const auto& node) { return node[kNodeType] == kNodeFunctionStart; }) |
                        std::views::filter([&](const auto& node) {
                            int child_counter = node[kCounter].template get<int>();
                            size_t child_level = node[kStackingLevel].template get<size_t>();
                            return child_level == parent_level + 1 && child_counter > function_start_counter &&
                                   child_counter < function_end_counter && child_level <= max_level;
                        }) |
                        std::views::transform([&](const auto& node) { return node[kCounter].template get<int>(); }) |
                        std::views::filter([&](int child_counter) { return id_map.contains(child_counter); }) |
                        std::views::transform([&](int child_counter) { return id_map[child_counter]; });

        graph[vertex_id].internals.assign(children.begin(), children.end());
    });
}

// Fifth pass: Populate output_info
// For each vertex, find same-level consumers and extract their first tensor argument
// This can later change if graph capture stores the output info of each vertex.
void LevelizedGraph::populate_output_info() {
    auto vertex_ids = std::views::iota(0u, graph.size());
    std::ranges::for_each(vertex_ids, [&](size_t vertex_id) {
        auto& vertex = graph[vertex_id];
        size_t vertex_level = vertex.stacking_level;

        // Find connections at the same stacking level
        auto same_level_consumers =
            vertex.out_edges | std::views::filter([&](size_t consumer_id) {
                return consumer_id < graph.size() && graph[consumer_id].stacking_level == vertex_level;
            });

        // Extract the first tensor argument from same-level consumers
        for (size_t consumer_id : same_level_consumers) {
            const auto& consumer = graph[consumer_id];
            if (!consumer.arguments.empty()) {
                const std::string& first_arg = consumer.arguments[0];
                // Check if it's a Tensor argument (starts with "Tensor(")
                if (first_arg.find("Tensor(") == 0) {
                    // Add to output_info if not already present (avoid duplicates)
                    // Note: this is not scalable, but the only viable solution for now since graph capture does not
                    // store the output info of each vertex.
                    if (std::ranges::find(vertex.output_info, first_arg) == vertex.output_info.end()) {
                        vertex.output_info.push_back(first_arg);
                    }
                    break;  // Found it, no need to check other consumers
                }
            }
        }
    });
}

// Sixth pass: Populate output_shape
// For each vertex, find its function_end and extract shape from output tensor nodes
void LevelizedGraph::populate_output_shape(const nlohmann::json& trace) {
    auto vertex_ids = std::views::iota(0u, graph.size());
    std::ranges::for_each(vertex_ids, [&](size_t vertex_id) {
        auto& vertex = graph[vertex_id];
        int node_counter = reverse_id_map[vertex_id];
        const auto& node = trace[node_counter];

        // Skip tensor nodes - they already have their shape from populate_vertices
        if (node[kNodeType] == kNodeTensor) {
            return;
        }

        int function_start_counter = node_counter;

        // Find the function_end for this function_start
        auto start_to_end_it = function_start_to_end.find(function_start_counter);
        if (start_to_end_it == function_start_to_end.end()) {
            return;  // No function_end found, skip
        }
        int function_end_counter = start_to_end_it->second;

        // Get the function_end node from trace
        if (function_end_counter < 0 || static_cast<size_t>(function_end_counter) >= trace.size()) {
            return;
        }
        const auto& function_end_node = trace[function_end_counter];

        // Get connections from function_end and extract shapes from tensor nodes
        auto shapes =
            get_valid_connections(function_end_node, trace) |
            std::views::filter([&](int conn_counter) { return trace[conn_counter][kNodeType] == kNodeTensor; }) |
            std::views::filter([&](int conn_counter) {
                const auto& connected_node = trace[conn_counter];
                return connected_node.contains(kParams) && connected_node[kParams].contains(kShape);
            }) |
            std::views::transform(
                [&](int conn_counter) { return trace[conn_counter][kParams][kShape].template get<std::string>(); }) |
            std::views::filter([&](const std::string& shape_str) {
                return std::ranges::find(vertex.output_shape, shape_str) == vertex.output_shape.end();
            });

        vertex.output_shape.insert(vertex.output_shape.end(), shapes.begin(), shapes.end());
    });
}

// Seventh pass: Populate input_connections
// For each vertex, find input tensor producers for each tensor argument
// Use the input_tensors field from graph capture if available
void LevelizedGraph::populate_input_connections(const nlohmann::json& trace) {
    auto vertex_ids = std::views::iota(0u, graph.size());
    std::ranges::for_each(vertex_ids, [&](size_t vertex_id) {
        auto& vertex = graph[vertex_id];
        int node_counter = reverse_id_map[vertex_id];
        const auto& node = trace[node_counter];

        // Skip tensor nodes - they don't have input connections in the same way
        if (node[kNodeType] == kNodeTensor) {
            // For tensor nodes, find if they have a producer
            auto tensor_producer_it = tensor_to_producer_end.find(node_counter);
            if (tensor_producer_it != tensor_to_producer_end.end()) {
                int producer_end = tensor_producer_it->second;
                auto end_to_start_it = function_end_to_start.find(producer_end);
                if (end_to_start_it != function_end_to_start.end()) {
                    int producer_start = end_to_start_it->second;
                    // Check if producer is in our levelized graph
                    auto producer_id_it = id_map.find(producer_start);
                    if (producer_id_it != id_map.end()) {
                        size_t producer_vertex_id = producer_id_it->second;
                        vertex.in_edges.push_back(producer_vertex_id);
                    }
                }
            }
            // If no producer, this is an input tensor (no in_edges)
            return;
        }

        // Check if input_tensors field is available, use it to populate in_edges
        if (node.contains(kInputTensors) && node[kInputTensors].is_array()) {
            // New format: directly use input_tensors field which preserves order
            auto input_tensors = node[kInputTensors];
            for (const auto& tensor_counter_json : input_tensors) {
                int tensor_counter = tensor_counter_json.template get<int>();

                // Check if this tensor is a vertex in our graph (level 1 tensor)
                auto tensor_vertex_it = id_map.find(tensor_counter);
                if (tensor_vertex_it != id_map.end()) {
                    // This is a level 1 tensor vertex, add it as input
                    size_t tensor_vertex_id = tensor_vertex_it->second;
                    vertex.in_edges.push_back(tensor_vertex_id);
                } else {
                    // Find which operation produced this tensor
                    auto tensor_producer_it = tensor_to_producer_end.find(tensor_counter);
                    if (tensor_producer_it != tensor_to_producer_end.end()) {
                        int producer_end = tensor_producer_it->second;
                        auto end_to_start_it = function_end_to_start.find(producer_end);
                        if (end_to_start_it != function_end_to_start.end()) {
                            int producer_start = end_to_start_it->second;
                            // Check if producer is in our levelized graph
                            auto producer_id_it = id_map.find(producer_start);
                            if (producer_id_it != id_map.end()) {
                                size_t producer_vertex_id = producer_id_it->second;
                                vertex.in_edges.push_back(producer_vertex_id);
                            }
                        }
                    }
                }
            }
        }
    });
}

}  // namespace ttnn::graph
