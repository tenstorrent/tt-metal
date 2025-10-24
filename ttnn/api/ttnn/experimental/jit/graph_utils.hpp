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
        std::function<void(const LazyTensor&)> visitor);

    static LazyTensorId get_available_lazy_tensor_id();

    static std::vector<LazyTensor> get_all_lazy_tensors(const LazyTensor& root);

    static std::vector<LazyTensor> get_ancestors(const LazyTensor& root);

    static std::vector<LazyTensor> get_descendants(const LazyTensor& root);

    static std::vector<LazyTensor> topological_sort(const LazyTensor& root);
};
}  // namespace ttnn::experimental::jit
