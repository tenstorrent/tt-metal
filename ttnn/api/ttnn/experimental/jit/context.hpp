// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/experimental/jit/node.hpp"
#include "ttnn/experimental/jit/to_organize.hpp"

namespace ttnn::experimental::jit {

class Context {
public:
    Context();

    Node* get_node(NodeId id);
    const Node* get_node(NodeId id) const;

    const std::vector<Node>& get_all_nodes() const;
    std::vector<Node>& get_all_nodes();

    std::unordered_set<NodeId> get_dependencies(NodeId id) const;

    std::vector<NodeId> topological_sort(const std::unordered_set<NodeId>& node_set) const;

    size_t size() const;
    void clear();

    static Context& instance();

    NodeId create_node(
        const std::vector<Tensor>& inputs,
        const std::string&& operation_name,
        std::shared_ptr<IDeviceOperation>&& Args);

    std::vector<const Node*> find_nodes(const std::string& operation_name) const;

    void execute_node(NodeId id);

    // Get the materialized output tensor for a placeholder tensor
    Tensor get_materialized_tensor(const Tensor& placeholder) const;

private:
    std::vector<Node> nodes_;
    std::unordered_map<NodeId, size_t> id_to_index_;
    NodeId next_id_ = 1;
};
}  // namespace ttnn::experimental::jit
