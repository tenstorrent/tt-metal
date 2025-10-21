#include "ttnn/experimental/jit/context.hpp"
#include "ttnn/experimental/jit/node.hpp"
#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>
#pragma optimize("", off)
namespace {
constexpr size_t INITIAL_NODES_CAPACITY = 1024;
}

namespace ttnn::experimental::jit {
Context::Context() {
    nodes_.reserve(INITIAL_NODES_CAPACITY);
    id_to_index_.reserve(INITIAL_NODES_CAPACITY);
}

NodeId Context::create_node(
    const std::vector<Tensor>& inputs, const std::string&& operation_name, std::shared_ptr<IDeviceOperation>&& args) {
    NodeId id = next_id_++;
    size_t index = nodes_.size();

    args->validate(inputs);

    nodes_.emplace_back(
        id,
        inputs,
        std::forward<const std::string&&>(operation_name),
        std::forward<std::shared_ptr<IDeviceOperation>&&>(args));
    id_to_index_[id] = index;

    // Update connectivity for input nodes
    for (const auto& input : inputs) {
        if (input.producer_node() != 0) {
            if (Node* producer = get_node(input.producer_node())) {
                producer->add_output_node(id);
            }
        }
    }

    log_info(
        tt::LogOp, "LAZY MODE: Created node {} for operation '{}' with {} input(s)", id, operation_name, inputs.size());

    return id;
}

std::vector<const Node*> Context::find_nodes(const std::string& operation_name) const {
    std::vector<const Node*> result;
    for (const auto& node : nodes_) {
        if (node.operation_name() == operation_name) {
            result.push_back(&node);
        }
    }
    return result;
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

void Context::execute_node(NodeId id) {
    std::vector<NodeId> nodes = topological_sort({id});
    log_info(tt::LogOp, "LAZY MODE: Executing {} node(s) in topological order", nodes.size());
    for (auto node_id : nodes) {
        auto node = get_node(node_id);
        if (node != nullptr) {
            log_info(tt::LogOp, "LAZY MODE: Executing node {} - operation '{}'", node_id, node->operation_name());
            node->execute();
            log_info(tt::LogOp, "LAZY MODE: Completed node {} - operation '{}'", node_id, node->operation_name());
        }
    }
    log_info(tt::LogOp, "LAZY MODE: Finished executing all nodes");
}

// Build dependency graph from output tensors
std::unordered_set<NodeId> Context::get_dependencies(const std::vector<Tensor>& outputs) const {
    std::unordered_set<NodeId> deps;
    std::vector<NodeId> to_visit;

    // Start from output tensors
    for (const auto& tensor : outputs) {
        // if the tensor has a producer and the producer node is not materialized, add it to the to_visit list
        // this avoids reevaluating parts of the graph that have already been materialized
        if (tensor.producer_node() != 0 && !get_node(tensor.producer_node())->is_materialized()) {
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
                if (input.producer_node() != 0) {
                    to_visit.push_back(input.producer_node());
                }
            }
        }
    }

    return deps;
}

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
                if (input.producer_node() != 0) {
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

Context& Context::instance() {
    static Context ctx;
    return ctx;
}
}  // namespace ttnn::experimental::jit
