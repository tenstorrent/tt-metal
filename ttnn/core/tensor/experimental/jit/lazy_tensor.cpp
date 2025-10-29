// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/tensor/tensor.hpp>
#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include "ttnn/experimental/jit/graph_utils.hpp"
#include "ttnn/experimental/jit/lazy_operation.hpp"
#include "ttnn/experimental/jit/lazy_mode.hpp"

namespace ttnn::experimental::jit {

// Lazy Tensor
LazyTensor::LazyTensor(const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, TensorSpec tensor_spec) :
    op_inputs_(op_inputs),
    op_(std::move(op)),
    tensor_spec_(std::move(tensor_spec)),
    id_(GraphUtils::get_available_lazy_tensor_id()) {}

LazyTensor LazyTensor::make_lazy_tensor(
    const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, TensorSpec tensor_spec) {
    return LazyTensor(op_inputs, std::move(op), std::move(tensor_spec));
}

LazyTensor LazyTensor::make_materialized_tensor(const tt::tt_metal::metal_tensor::Tensor& metal_tensor) {
    return LazyTensor(metal_tensor);
}

std::vector<LazyTensor> LazyTensor::make_lazy_tensors(
    const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, const std::vector<TensorSpec>& tensor_specs) {
    std::vector<LazyTensor> lazy_tensors;
    lazy_tensors.reserve(tensor_specs.size());
    for (const auto& tensor_spec : tensor_specs) {
        lazy_tensors.push_back(make_lazy_tensor(op_inputs, op, tensor_spec));
    }
    for (size_t i = 0; i < lazy_tensors.size(); i++) {
        auto siblings = lazy_tensors;
        siblings.erase(siblings.begin() + i);
        lazy_tensors[i].set_siblings(siblings);
        lazy_tensors[i].set_materialized_output_idx(i);
    }
    return lazy_tensors;
}

LazyTensor::LazyTensor(const tt::tt_metal::metal_tensor::Tensor& metal_tensor) :
    op_inputs_({}),
    op_(nullptr),
    tensor_spec_(metal_tensor.tensor_spec()),
    siblings_({}),
    materialized_outputs_({metal_tensor}),
    materialized_output_idx_(0),
    state_(LazyTensorState::MATERIALIZED),
    id_(GraphUtils::get_available_lazy_tensor_id()) {}

// Getters

const std::vector<LazyTensor>& LazyTensor::op_inputs() const { return op_inputs_; };
const std::vector<LazyTensor>& LazyTensor::siblings() const { return siblings_; }
const std::vector<tt::tt_metal::metal_tensor::Tensor>& LazyTensor::materialized_tensors() const {
    return materialized_outputs_;
}
const tt::tt_metal::metal_tensor::Tensor& LazyTensor::materialized_tensor() const {
    return materialized_outputs_[materialized_output_idx_];
}
tt::tt_metal::metal_tensor::Tensor& LazyTensor::materialized_tensor() {
    return materialized_outputs_[materialized_output_idx_];
}
const TensorSpec& LazyTensor::tensor_spec() const { return tensor_spec_.value(); }
LazyTensorState LazyTensor::state() const { return state_; }
LazyTensorId LazyTensor::id() const { return id_; }
bool LazyTensor::is_materialized() const { return state_ == LazyTensorState::MATERIALIZED; }
const LazyTensor::LazyOperationPtr& LazyTensor::op() const { return op_; }

void LazyTensor::materialize() {
    if (state_ == LazyTensorState::MATERIALIZED || state_ == LazyTensorState::SCHEDULED) {
        return;
    }

    state_ = LazyTensorState::SCHEDULED;

    // Verify that all inputs are materialized
    std::vector<MaterializedTensor> input_tensors;
    for (const auto& input : op_inputs_) {
        auto op_name = input.op() ? input.op()->name() : "Unknown";
        TT_FATAL(
            input.is_materialized(),
            "Input tensor {} produced by operation {} is not materialized",
            input.id(),
            op_name);
        input_tensors.push_back(input.materialized_tensor());
    }

    materialized_outputs_ = op_->invoke(input_tensors);
    state_ = LazyTensorState::MATERIALIZED;
    // Now update siblings' materialized tensors
    for (auto& sibling : siblings_) {
        // TODO: Make sure that this is not expensive copy
        sibling.materialized_outputs_ = materialized_outputs_;
        sibling.state_ = LazyTensorState::MATERIALIZED;
    }
}

//
}  // namespace ttnn::experimental::jit
