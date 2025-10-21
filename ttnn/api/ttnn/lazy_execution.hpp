// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/jit/context.hpp"
#include <vector>

namespace ttnn {

// Execute the lazy computation graph for the given output tensors
inline void execute_lazy_graph(const std::vector<Tensor>& output_tensors) {
    auto& context = ttnn::experimental::jit::Context::instance();

    log_info(tt::LogOp, "LAZY MODE: Starting lazy graph execution for {} output tensor(s)", output_tensors.size());

    // Get all dependencies for the output tensors
    auto dependencies = context.get_dependencies(output_tensors);

    // Get topological order
    auto execution_order = context.topological_sort(dependencies);

    log_info(tt::LogOp, "LAZY MODE: Executing {} node(s) in topological order", execution_order.size());

    // Execute each node in order
    for (auto node_id : execution_order) {
        context.execute_node(node_id);
    }

    log_info(tt::LogOp, "LAZY MODE: Finished lazy graph execution");
}

// Execute the lazy computation graph for a single output tensor
inline void execute_lazy_graph(const Tensor& output_tensor) { execute_lazy_graph(std::vector<Tensor>{output_tensor}); }

// Clear the lazy computation graph
inline void clear_lazy_graph() {
    auto& context = ttnn::experimental::jit::Context::instance();
    context.clear();
}

}  // namespace ttnn
