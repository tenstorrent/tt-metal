// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/lazy/graph_utils.hpp"

namespace ttnn::experimental::lazy {

void GraphUtils::dfs_visit(
    const std::shared_ptr<LazyTensor>& node,
    std::unordered_set<LazyTensorId>& visited,
    const std::function<void(const std::shared_ptr<LazyTensor>&)>& visitor) {
    LazyTensorId id = node->id();
    if (visited.find(id) != visited.end()) {
        return;
    }
    visited.insert(node->id());
    visitor(node);

    // Visit all input nodes
    for (const auto& input : node->op_inputs()) {
        dfs_visit(input, visited, visitor);
    }
}

LazyTensorId GraphUtils::get_available_lazy_tensor_id() {
    static LazyTensorId next_id = 0;
    return next_id++;
}

std::vector<std::shared_ptr<LazyTensor>> GraphUtils::get_all_lazy_tensors(const std::shared_ptr<LazyTensor>& root) {
    std::unordered_set<LazyTensorId> visited;
    std::vector<std::shared_ptr<LazyTensor>> all_nodes;

    dfs_visit(root, visited, [&](const std::shared_ptr<LazyTensor>& node) { all_nodes.push_back(node); });

    return all_nodes;
}

std::vector<std::shared_ptr<LazyTensor>> GraphUtils::get_ancestors(const std::shared_ptr<LazyTensor>& root) {
    std::unordered_set<LazyTensorId> visited;
    std::vector<std::shared_ptr<LazyTensor>> ancestors;

    dfs_visit(root, visited, [&](const std::shared_ptr<LazyTensor>& node) {
        for (const auto& input : node->op_inputs()) {
            ancestors.push_back(input);
        }
    });

    return ancestors;
}

std::vector<std::shared_ptr<LazyTensor>> GraphUtils::get_descendants(const std::shared_ptr<LazyTensor>& root) {
    std::unordered_set<LazyTensorId> visited;
    std::vector<std::shared_ptr<LazyTensor>> descendants;

    dfs_visit(root, visited, [&](const std::shared_ptr<LazyTensor>& node) {
        for (const auto& output : node->siblings()) {
            descendants.push_back(output);
        }
    });

    return descendants;
}

std::vector<std::shared_ptr<LazyTensor>> GraphUtils::topological_sort(const std::shared_ptr<LazyTensor>& root) {
    std::unordered_set<LazyTensorId> visited;
    std::unordered_set<LazyTensorId> temp_visited;
    std::vector<std::shared_ptr<LazyTensor>> result;

    std::function<void(const std::shared_ptr<LazyTensor>&)> dfs = [&](const std::shared_ptr<LazyTensor>& node) {
        LazyTensorId id = node->id();

        // If already permanently visited, skip
        if (visited.find(id) != visited.end()) {
            return;
        }

        // Check for cycle (node is in the current recursion stack)
        TT_FATAL(temp_visited.find(id) == temp_visited.end(), "Cycle detected in computation graph");

        temp_visited.insert(id);

        // Visit all input nodes first (dependencies)
        for (const auto& input : node->op_inputs()) {
            dfs(input);
        }

        temp_visited.erase(id);
        visited.insert(id);
        result.push_back(node);
    };

    dfs(root);
    return result;
}
}  // namespace ttnn::experimental::lazy
