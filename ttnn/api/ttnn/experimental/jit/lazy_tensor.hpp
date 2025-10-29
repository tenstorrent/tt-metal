// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttnn::experimental::jit {

struct LazyOperation;
using LazyTensorId = uint32_t;
enum class LazyTensorState {
    LAZY,         // Contains graph node information
    SCHEDULED,    // Operation scheduled for evaluation
    MATERIALIZED  // Contains actual data
};

class LazyTensor {
    using LazyOperationPtr = std::shared_ptr<ttnn::experimental::jit::LazyOperation>;
    using Tensor = tt::tt_metal::Tensor;
    using TensorSpec = tt::tt_metal::TensorSpec;

public:
    LazyTensor() = default;
    // This used for ops that return a single tensor
    static LazyTensor make_lazy_tensor(
        const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, TensorSpec tensor_spec);

    // This used for ops that return multiple tensors
    static std::vector<LazyTensor> make_lazy_tensors(
        const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, const std::vector<TensorSpec>& tensor_specs);

    static LazyTensor make_materialized_tensor(const Tensor& metal_tensor);

    // Note: no public setters, LazyTensor is immutable after construction

    // Getters
    const std::vector<LazyTensor>& op_inputs() const { return op_inputs_; };
    const std::vector<LazyTensor>& siblings() const { return siblings_; }
    const std::vector<Tensor>& materialized_tensors() const { return materialized_outputs_; }
    const Tensor& materialized_tensor() const { return materialized_outputs_[materialized_output_idx_]; }
    Tensor& materialized_tensor() { return materialized_outputs_[materialized_output_idx_]; }
    const TensorSpec& tensor_spec() const { return tensor_spec_.value(); }
    LazyTensorState state() const { return state_; }
    LazyTensorId id() const { return id_; }
    bool is_materialized() const { return state_ == LazyTensorState::MATERIALIZED; }
    const LazyOperationPtr& op() const { return op_; }

    void materialize();

private:
    LazyTensor(const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, TensorSpec tensor_spec);
    LazyTensor(const tt::tt_metal::Tensor& metal_tensor);

    void set_siblings(const std::vector<LazyTensor>& siblings) { siblings_ = siblings; }
    void set_materialized_output_idx(size_t idx) { materialized_output_idx_ = idx; }
    void set_state(LazyTensorState state) { state_ = state; }

    std::vector<LazyTensor> op_inputs_;
    LazyOperationPtr op_;
    std::optional<TensorSpec> tensor_spec_;  // <- Note really optional, but I need to quickly get default ctor working
    std::vector<LazyTensor> siblings_;
    std::vector<Tensor> materialized_outputs_;
    size_t materialized_output_idx_ = 0;  // In case op produces multiple tensors, we want to know which one is this
    LazyTensorState state_ = LazyTensorState::LAZY;
    LazyTensorId id_ = 0;
};

}  // namespace ttnn::experimental::jit
