// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/tensor_topology.hpp"

namespace ttnn::experimental::jit {

struct LazyOperation;
using LazyTensorId = uint32_t;
enum class LazyTensorState {
    LAZY,         // Contains graph node information
    SCHEDULED,    // Operation scheduled for evaluation
    MATERIALIZED  // Contains actual data
};

enum class JitOperationType {
    LAZY_JIT,
    EAGER_JIT,
};

static JitOperationType jit_operation_type = JitOperationType::LAZY_JIT;

class LazyTensor {
public:
    LazyTensor() = delete;
    LazyTensor(std::vector<LazyTensor> inputs, ttnn::experimental::jit::LazyOperation* Args);

    ~LazyTensor();

    const std::vector<LazyTensor>& inputs() const { return inputs_; };
    const std::vector<LazyTensor>& outputs() const { return outputs_; };
    const std::vector<LazyTensorId> output_lazy_tensor_ids() const { return output_nodes_; };

    void add_output_node(const LazyTensorId node_id) { output_nodes_.push_back(node_id); }

    void execute();

    bool is_materialized() const { return output_tensors_.size() > 0; }

    LazyTensorState get_state() const { return state_; };

    const LazyTensorId id() const { return id_; };

    // This is not optional, but let's keep it like this for now
    std::optional<ttnn::TensorSpec> get_tensor_spec() const { return tensor_spec_; }

    void set_tensor_spec(const ttnn::TensorSpec& tensor_spec) { tensor_spec_ = tensor_spec; }

private:
    std::vector<tt::tt_metal::Tensor> output_tensors_;
    std::optional<ttnn::TensorSpec> tensor_spec_;
    std::shared_ptr<ttnn::experimental::jit::LazyOperation> args_;
    std::vector<LazyTensor> inputs_;
    std::vector<LazyTensor> outputs_;
    LazyTensorState state_;
    std::vector<LazyTensorId> output_nodes_;
    LazyTensorId id_;
};

}  // namespace ttnn::experimental::jit
