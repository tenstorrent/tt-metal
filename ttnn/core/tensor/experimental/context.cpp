#include "context.hpp"

#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace {
constexpr size_t INITIAL_NODES_CAPACITY = 1024;
}

Context::Context() {
    nodes_.reserve(INITIAL_NODES_CAPACITY);
    id_to_index_.reserve(INITIAL_NODES_CAPACITY);
}

Node* Context::get_node(NodeId id) {
    auto it = id_to_index_.find(id);
    return it != id_to_index_.end() ? &nodes_[it->second] : nullptr;
}

const Node* Context::get_node(NodeId id) const {
    auto it = id_to_index_.find(id);
    return it != id_to_index_.end() ? &nodes_[it->second] : nullptr;
}

// Get all nodes for inspection
const std::vector<Node>& Context::get_all_nodes() const { return nodes_; }

std::vector<Node>& Context::get_all_nodes() { return nodes_; }

// Build dependency graph from output tensors
std::unordered_set<NodeId> Context::get_dependencies(const std::vector<Tensor>& outputs) const {
    std::unordered_set<NodeId> deps;
    std::vector<NodeId> to_visit;

    // Start from output tensors
    for (const auto& tensor : outputs) {
        if (!tensor.is_constant() && tensor.producer_node() != 0) {
            to_visit.push_back(tensor.producer_node());
        }
    }

    // DFS to find all dependencies
    while (!to_visit.empty()) {
        NodeId current = to_visit.back();
        to_visit.pop_back();

        if (deps.count(current)) {
            continue;
        }
        deps.insert(current);

        if (const Node* node = get_node(current)) {
            for (const auto& input : node->inputs()) {
                if (!input.is_constant() && input.producer_node() != 0) {
                    to_visit.push_back(input.producer_node());
                }
            }
        }
    }

    return deps;
}

// Topological sort for execution
std::vector<NodeId> Context::topological_sort(const std::unordered_set<NodeId>& node_set) const {
    std::vector<NodeId> result;
    std::unordered_set<NodeId> visited;
    std::unordered_set<NodeId> temp_visited;

    std::function<void(NodeId)> visit = [&](NodeId id) {
        if (temp_visited.count(id)) {
            throw std::runtime_error("Cycle detected in graph");
        }
        if (visited.count(id) || !node_set.count(id)) {
            return;
        }

        temp_visited.insert(id);

        if (const Node* node = get_node(id)) {
            for (const auto& input : node->inputs()) {
                if (!input.is_constant() && input.producer_node() != 0) {
                    visit(input.producer_node());
                }
            }
        }

        temp_visited.erase(id);
        visited.insert(id);
        result.push_back(id);
    };

    for (NodeId id : node_set) {
        if (!visited.count(id)) {
            visit(id);
        }
    }

    return result;
}

size_t Context::size() const { return nodes_.size(); }

void Context::clear() {
    nodes_.clear();
    id_to_index_.clear();
    next_id_ = 1;
}

void Context::print_stats() const {
    std::unordered_map<OpTypeId, size_t> counts;
    for (const auto& node : nodes_) {
        counts[node.type_id()]++;
    }

    spdlog::info("Graph statistics:");
    spdlog::info("  Total nodes: {}", nodes_.size());
    spdlog::info("  Operation counts:");
    for (const auto& [type_id, count] : counts) {
        spdlog::info("    Type {}: {} nodes", type_id, count);
    }
}

Context& Context::instance() {
    static Context ctx;
    return ctx;
}
