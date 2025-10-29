// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/metal_tensor.hpp>
#include <ttnn/tensor/tensor_spec.hpp>

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
    using TensorSpec = tt::tt_metal::TensorSpec;
    using MaterializedTensor = tt::tt_metal::metal_tensor::Tensor;

public:
    LazyTensor() = default;
    // This used for ops that return a single tensor
    static LazyTensor make_lazy_tensor(
        const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, TensorSpec tensor_spec);

    // This used for ops that return multiple tensors
    static std::vector<LazyTensor> make_lazy_tensors(
        const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, const std::vector<TensorSpec>& tensor_specs);

    static LazyTensor make_materialized_tensor(const MaterializedTensor& metal_tensor);

    // Note: no public setters, LazyTensor is immutable after construction

    // Getters
    const std::vector<LazyTensor>& op_inputs() const;
    const std::vector<LazyTensor>& siblings() const;
    const std::vector<MaterializedTensor>& materialized_tensors() const;
    const MaterializedTensor& materialized_tensor() const;
    MaterializedTensor& materialized_tensor();
    const TensorSpec& tensor_spec() const;
    LazyTensorState state() const;
    LazyTensorId id() const;
    bool is_materialized() const;
    const LazyOperationPtr& op() const;

    void materialize();

private:
    LazyTensor(const std::vector<LazyTensor>& op_inputs, LazyOperationPtr op, TensorSpec tensor_spec);
    LazyTensor(const MaterializedTensor& metal_tensor);

    void set_siblings(const std::vector<LazyTensor>& siblings) { siblings_ = siblings; }
    void set_materialized_output_idx(size_t idx) { materialized_output_idx_ = idx; }
    void set_state(LazyTensorState state) { state_ = state; }

    std::vector<LazyTensor> op_inputs_;
    LazyOperationPtr op_;
    std::optional<TensorSpec> tensor_spec_;  // <- Note really optional, but I need to quickly get default ctor working
    std::vector<LazyTensor> siblings_;
    std::vector<MaterializedTensor> materialized_outputs_;
    size_t materialized_output_idx_ = 0;  // In case op produces multiple tensors, we want to know which one is this
    LazyTensorState state_ = LazyTensorState::LAZY;
    LazyTensorId id_ = 0;
};

}  // namespace ttnn::experimental::jit
