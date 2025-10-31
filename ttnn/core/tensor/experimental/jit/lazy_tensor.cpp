// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/tensor/tensor.hpp>
#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include "ttnn/experimental/jit/graph_utils.hpp"
#include "ttnn/experimental/jit/lazy_operation.hpp"

namespace ttnn::experimental::jit {

// Lazy Tensor
LazyTensor::LazyTensor(
    const std::vector<std::shared_ptr<LazyTensor>>& op_inputs, const LazyOperationPtr& op, TensorSpec tensor_spec) :
    op_inputs_(op_inputs),
    op_(op),
    tensor_spec_(std::move(tensor_spec)),
    id_(GraphUtils::get_available_lazy_tensor_id()) {
    // Inherit device and storage_type from first input tensor
    // TODO: Is this correct though?
    if (!op_inputs_.empty() && op_inputs_[0]) {
        device_ = op_inputs_[0]->device();
        storage_type_ = op_inputs_[0]->storage_type();
    } else {
        // Default values when no inputs
        device_ = nullptr;
        storage_type_ = tt::tt_metal::StorageType::DEVICE;
    }
}

std::shared_ptr<LazyTensor> LazyTensor::make_lazy_tensor(
    const std::vector<std::shared_ptr<LazyTensor>>& op_inputs, const LazyOperationPtr& op, TensorSpec tensor_spec) {
    return std::make_shared<LazyTensor>(op_inputs, op, std::move(tensor_spec));
}

std::shared_ptr<LazyTensor> LazyTensor::make_materialized_tensor(
    const tt::tt_metal::metal_tensor::Tensor& metal_tensor) {
    return std::make_shared<LazyTensor>(metal_tensor);
}

std::vector<std::shared_ptr<LazyTensor>> LazyTensor::make_lazy_tensors(
    const std::vector<std::shared_ptr<LazyTensor>>& op_inputs,
    const LazyOperationPtr& op,
    const std::vector<TensorSpec>& tensor_specs) {
    std::vector<std::shared_ptr<LazyTensor>> lazy_tensors;
    lazy_tensors.reserve(tensor_specs.size());
    for (const auto& tensor_spec : tensor_specs) {
        lazy_tensors.push_back(make_lazy_tensor(op_inputs, op, tensor_spec));
    }
    for (size_t i = 0; i < lazy_tensors.size(); i++) {
        std::vector<std::shared_ptr<LazyTensor>> siblings;
        siblings.reserve(lazy_tensors.size() - 1);
        for (size_t j = 0; j < lazy_tensors.size(); j++) {
            if (i != j) {
                siblings.push_back(lazy_tensors[j]);
            }
        }
        lazy_tensors[i]->set_siblings(siblings);
        lazy_tensors[i]->set_materialized_output_idx(i);
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
    id_(GraphUtils::get_available_lazy_tensor_id()),
    device_(metal_tensor.device()),
    storage_type_(metal_tensor.storage_type()) {}

// Getters

const std::vector<std::shared_ptr<LazyTensor>>& LazyTensor::op_inputs() const { return op_inputs_; };
const std::vector<std::shared_ptr<LazyTensor>>& LazyTensor::siblings() const { return siblings_; }
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
tt::tt_metal::distributed::MeshDevice* LazyTensor::device() const { return device_; }
tt::tt_metal::StorageType LazyTensor::storage_type() const { return storage_type_; }

void LazyTensor::materialize() {
    if (state_ == LazyTensorState::MATERIALIZED || state_ == LazyTensorState::SCHEDULED) {
        return;
    }

    state_ = LazyTensorState::SCHEDULED;

    // Verify that all inputs are materialized
    for (const auto& input : op_inputs_) {
        auto op_name = input->op() ? input->op()->name() : "Unknown";
        TT_FATAL(
            input->is_materialized(),
            "Input tensor {} produced by operation {} is not materialized",
            input->id(),
            op_name);
    }

    materialized_outputs_ = op_->invoke();
    state_ = LazyTensorState::MATERIALIZED;
    // Now update siblings' materialized tensors
    for (auto& sibling : siblings_) {
        // TODO: Make sure that this is not expensive copy
        sibling->materialized_outputs_ = materialized_outputs_;
        sibling->state_ = LazyTensorState::MATERIALIZED;
    }
}

void LazyTensor::set_siblings(const std::vector<std::shared_ptr<LazyTensor>>& siblings) { siblings_ = siblings; }
void LazyTensor::set_materialized_output_idx(size_t idx) { materialized_output_idx_ = idx; }
void LazyTensor::set_state(LazyTensorState state) { state_ = state; }

//
}  // namespace ttnn::experimental::jit
