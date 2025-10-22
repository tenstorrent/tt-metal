#include "ttnn/experimental/jit/context.hpp"
#include "ttnn/experimental/jit/node.hpp"
#include <stdexcept>

#include <spdlog/spdlog.h>

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
    std::unordered_set<NodeId> nodes_to_execute = get_dependencies(id);
    // Now do topological sort on the full dependency set
    std::vector<NodeId> nodes = topological_sort(nodes_to_execute);
    log_info(tt::LogOp, "LAZY MODE: Executing {} node(s) in topological order", nodes.size());
    for (auto node_id : nodes) {
        auto node = get_node(node_id);
        if (node != nullptr) {
            log_info(tt::LogOp, "LAZY MODE: Executing node {} - operation '{}'", node_id, node->operation_name());

            // Materialize inputs: replace placeholder tensors with actual outputs from producer nodes
            auto& inputs = node->inputs_mut();
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto producer_node_id = inputs[i].producer_node();
                if (producer_node_id != 0) {
                    // This input is a placeholder from a producer node
                    const Node* producer = get_node(producer_node_id);
                    if (producer != nullptr && producer->is_materialized()) {
                        const auto& producer_outputs = producer->outputs();
                        // For now, assume single output or match by tensor_id later
                        // TODO: Handle multiple outputs properly
                        if (!producer_outputs.empty()) {
                            // Replace placeholder with first (and typically only) output
                            // TODO: This assumes single output - need to track output index for multi-output ops
                            inputs[i] = producer_outputs[0];
                            log_info(
                                tt::LogOp,
                                "LAZY MODE: Materialized input {} from producer node {}",
                                i,
                                producer_node_id);
                        }
                    }
                }
            }

            node->execute();
            log_info(tt::LogOp, "LAZY MODE: Completed node {} - operation '{}'", node_id, node->operation_name());
        }
    }
    log_info(tt::LogOp, "LAZY MODE: Finished executing all nodes");
}

// Build dependency graph from output tensors
std::unordered_set<NodeId> Context::get_dependencies(NodeId id) const {
    // Build a set of all nodes we need to execute
    std::unordered_set<NodeId> dependencies;
    dependencies.insert(id);

    // Add all dependencies by traversing inputs
    std::vector<NodeId> to_visit = {id};
    while (!to_visit.empty()) {
        NodeId current = to_visit.back();
        to_visit.pop_back();

        const Node* node = get_node(current);
        if (node) {
            for (const auto& input : node->inputs()) {
                NodeId input_producer = input.producer_node();
                if (input_producer != 0 && !dependencies.count(input_producer)) {
                    dependencies.insert(input_producer);
                    to_visit.push_back(input_producer);
                }
            }
        }
    }

    return dependencies;
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

Tensor Context::get_materialized_tensor(const Tensor& placeholder) const {
    NodeId producer_id = placeholder.producer_node();
    if (producer_id == 0) {
        // Not a lazy tensor, return as is
        return placeholder;
    }

    std::cout << "[Context::get_materialized_tensor] producer_id: " << producer_id << std::endl;
    const Node* producer = get_node(producer_id);
    if (producer == nullptr || !producer->is_materialized()) {
        TT_THROW("Producer node not found or not materialized");
    }

    const auto& outputs = producer->outputs();
    if (outputs.empty()) {
        TT_THROW("Producer node has no outputs");
    }

    // For now, assume single output or return first output
    // TODO: Handle multiple outputs by matching tensor IDs
    return outputs[0];
}

Context& Context::instance() {
    static Context ctx;
    return ctx;
}
}  // namespace ttnn::experimental::jit
