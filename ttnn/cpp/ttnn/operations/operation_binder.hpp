// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ttnn/experimental/jit/IDeviceOperation.hpp>
#include <ttnn/tensor/tensor.hpp>

// This is not the most elegant file in the world, for now I am just encapsulating everything
// I will deal with this later.

namespace ttnn::operations {

enum class JitOperationType {
    LAZY_JIT,
    EAGER_JIT,
};

static JitOperationType jit_operation_type = JitOperationType::LAZY_JIT;

static std::vector<ttnn::Tensor> bind_operation(
    std::vector<ttnn::Tensor> inputs,
    const std::string&& operation_name,
    std::shared_ptr<ttnn::experimental::jit::IDeviceOperation> operation) {
    // Use Context to add a node with the args
    auto& context = ttnn::experimental::jit::Context::instance();

    // Create shared_ptr to hold the args
    auto args_ptr = std::move(operation);

    args_ptr->validate(inputs);

    // Add node to context
    auto node_id = context.create_node(
        inputs,
        std::move(operation_name),
        std::static_pointer_cast<ttnn::experimental::jit::IDeviceOperation>(args_ptr));

    if (jit_operation_type == JitOperationType::LAZY_JIT) {
        return context.get_node(node_id)->create_output_tensors();
    }

    return args_ptr->invoke(inputs);
}
}  // namespace ttnn::operations
