// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/jit/graph_utils.hpp"

namespace ttnn::experimental::jit {

void GraphUtils::dfs_visit(
    const LazyTensor& node, std::unordered_set<LazyTensorId>& visited, std::function<void(const LazyTensor&)> visitor) {
    LazyTensorId id = node.id();
    if (visited.find(id) != visited.end()) {
        return;
    }
    visited.insert(node.id());
    visitor(node);

    // Visit all input nodes
    for (const auto& input : node.inputs()) {
        dfs_visit(input, visited, visitor);
    }
}

LazyTensorId GraphUtils::get_available_lazy_tensor_id() {
    LazyTensorId available_lazy_tensor_id = 0;
    return available_lazy_tensor_id++;
}

std::vector<LazyTensor> GraphUtils::get_all_lazy_tensors(const LazyTensor& root) {
    std::unordered_set<LazyTensorId> visited;
    std::vector<LazyTensor> all_nodes;

    dfs_visit(root, visited, [&](const LazyTensor& node) { all_nodes.push_back(node); });

    return all_nodes;
}

std::vector<LazyTensor> GraphUtils::get_ancestors(const LazyTensor& root) {
    std::unordered_set<LazyTensorId> visited;
    std::vector<LazyTensor> ancestors;

    dfs_visit(root, visited, [&](const LazyTensor& node) {
        for (const auto& input : node.inputs()) {
            ancestors.push_back(input);
        }
    });

    return ancestors;
}

std::vector<LazyTensor> GraphUtils::get_descendants(const LazyTensor& root) {
    std::unordered_set<LazyTensorId> visited;
    std::vector<LazyTensor> descendants;

    dfs_visit(root, visited, [&](const LazyTensor& node) {
        for (const auto& output : node.outputs()) {
            descendants.push_back(output);
        }
    });

    return descendants;
}

std::vector<LazyTensor> GraphUtils::topological_sort(const LazyTensor& root) {
    std::unordered_set<LazyTensorId> visited;
    std::unordered_set<LazyTensorId> temp_visited;
    std::vector<LazyTensor> result;

    std::function<void(const LazyTensor&)> dfs = [&](const LazyTensor& node) {
        LazyTensorId id = node.id();

        TT_FATAL(temp_visited.find(id) == temp_visited.end(), "Cycle detected in computation graph");

        temp_visited.insert(id);

        // Visit all input nodes first
        for (const auto& input : node.inputs()) {
            dfs(input);
        }

        temp_visited.erase(id);
        visited.insert(id);
        result.push_back(node);
    };

    dfs(root);
    return result;
}
}  // namespace ttnn::experimental::jit
