// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include <unordered_set>
#include <vector>
#include <functional>

namespace ttnn::experimental::jit {

class GraphUtils {
public:
    static void dfs_visit(
        const LazyTensor& node,
        std::unordered_set<LazyTensorId>& visited,
        std::function<void(const LazyTensor&)> visitor) {
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

    static LazyTensorId get_available_lazy_tensor_id() {
        static LazyTensorId available_lazy_tensor_id = 0;
        return available_lazy_tensor_id++;
    }

    static std::vector<LazyTensor> get_all_lazy_tensors(const LazyTensor& root) {
        std::unordered_set<LazyTensorId> visited;
        std::vector<LazyTensor> all_nodes;

        dfs_visit(root, visited, [&](const LazyTensor& node) { all_nodes.push_back(node); });

        return all_nodes;
    }

    static std::vector<LazyTensor> get_ancestors(const LazyTensor& root) {
        std::unordered_set<LazyTensorId> visited;
        std::vector<LazyTensor> ancestors;

        dfs_visit(root, visited, [&](const LazyTensor& node) {
            for (const auto& input : node.inputs()) {
                ancestors.push_back(input);
            }
        });

        return ancestors;
    }

    static std::vector<LazyTensor> get_descendants(const LazyTensor& root) {
        std::unordered_set<LazyTensorId> visited;
        std::vector<LazyTensor> descendants;

        dfs_visit(root, visited, [&](const LazyTensor& node) {
            for (const auto& output : node.outputs()) {
                descendants.push_back(output);
            }
        });

        return descendants;
    }

    static std::vector<LazyTensor> topological_sort(const LazyTensor& root) {
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
};
}  // namespace ttnn::experimental::jit
