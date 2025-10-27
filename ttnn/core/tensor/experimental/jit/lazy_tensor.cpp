// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include "ttnn/experimental/jit/graph_utils.hpp"
#include "ttnn/experimental/jit/lazy_operation.hpp"

namespace ttnn::experimental::jit {

LazyTensor::LazyTensor(std::vector<LazyTensor> inputs, ttnn::experimental::jit::LazyOperation* Args) :
    inputs_(std::move(inputs)), args_(std::move(Args)) {
    state_ = LazyTensorState::LAZY;
    id_ = GraphUtils::get_available_lazy_tensor_id();

    if (jit_operation_type == JitOperationType::LAZY_JIT) {
        output_tensors_ = args_->invoke(inputs_);
        state_ = LazyTensorState::MATERIALIZED;
    }
}

LazyTensor::~LazyTensor() {}

void LazyTensor::execute() {
    output_tensors_ = args_->invoke(inputs_);
    state_ = LazyTensorState::MATERIALIZED;
}
}  // namespace ttnn::experimental::jit
