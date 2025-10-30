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
        const std::shared_ptr<LazyTensor>& node,
        std::unordered_set<LazyTensorId>& visited,
        const std::function<void(const std::shared_ptr<LazyTensor>&)>& visitor);

    static LazyTensorId get_available_lazy_tensor_id();

    static std::vector<std::shared_ptr<LazyTensor>> get_all_lazy_tensors(const std::shared_ptr<LazyTensor>& root);

    static std::vector<std::shared_ptr<LazyTensor>> get_ancestors(const std::shared_ptr<LazyTensor>& root);

    static std::vector<std::shared_ptr<LazyTensor>> get_descendants(const std::shared_ptr<LazyTensor>& root);

    static std::vector<std::shared_ptr<LazyTensor>> topological_sort(const std::shared_ptr<LazyTensor>& root);
};
}  // namespace ttnn::experimental::jit
