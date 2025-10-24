// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/experimental/jit/IDeviceOperation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/experimental/jit/lazy_tensor.hpp>

namespace ttnn::python_binding {

enum class JitOperationType {
    LAZY_JIT,
    EAGER_JIT,
};

static JitOperationType jit_operation_type = JitOperationType::LAZY_JIT;

static std::vector<ttnn::experimental::jit::LazyTensor> bind_operation(
    std::vector<ttnn::experimental::jit::LazyTensor> inputs,
    const std::string&& operation_name,
    std::shared_ptr<ttnn::experimental::jit::IDeviceOperation> operation) {
    auto args_ptr = std::move(operation);

    args_ptr->validate(inputs);

    // if (jit_operation_type == JitOperationType::LAZY_JIT) {
    ttnn::experimental::jit::LazyTensor lazy_tensor(
        args_ptr->compute_output_specs(inputs), inputs, operation_name, args_ptr);
    return {std::move(lazy_tensor)};
    // return args_ptr->invoke(inputs);
}
}  // namespace ttnn::python_binding
