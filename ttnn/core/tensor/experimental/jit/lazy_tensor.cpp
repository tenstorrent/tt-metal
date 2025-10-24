// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include "ttnn/experimental/jit/graph_utils.hpp"
#include "ttnn/experimental/jit/IDeviceOperation.hpp"

namespace ttnn::experimental::jit {

LazyTensor::LazyTensor(
    TensorSpec tensor_spec,
    const std::vector<LazyTensor>& inputs,
    const std::string&& operation_name,
    std::shared_ptr<ttnn::experimental::jit::IDeviceOperation>&& args) :
    tensor_spec_(std::move(tensor_spec)),
    inputs_(std::move(inputs)),
    operation_name_(std::move(operation_name)),
    args_(std::move(args)) {
    state_ = LazyTensorState::LAZY;
    id_ = GraphUtils::get_available_lazy_tensor_id();

    args_->validate(inputs_);
    tensor_spec_ = args_->compute_output_specs(inputs_).at(0);

    for (auto& input : inputs_) {
        input.add_output_node(id_);
    }
}

LazyTensor::~LazyTensor() {}

void LazyTensor::execute() {
    // auto output_tensors = args_->invoke(inputs_);
    //  for (const auto& output_tensor : output_tensors) {
    //      output_tensors_.push_back(output_tensor);
    //  }
    //  state_ = LazyTensorState::MATERIALIZED;
}
}  // namespace ttnn::experimental::jit
