// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/tensor_topology.hpp"

namespace ttnn::experimental::jit {

struct IDeviceOperation;
using LazyTensorId = uint32_t;
enum class LazyTensorState {
    LAZY,         // Contains graph node information
    SCHEDULED,    // Operation scheduled for evaluation
    MATERIALIZED  // Contains actual data
};

class LazyTensor {
public:
    LazyTensor() = delete;
    LazyTensor(
        ttnn::TensorSpec tensor_spec,
        const std::vector<LazyTensor>& inputs,
        const std::string&& operation_name,
        std::shared_ptr<ttnn::experimental::jit::IDeviceOperation>&& Args);

    ~LazyTensor();

    std::string_view operation_name() const { return operation_name_; };

    const std::vector<LazyTensor>& inputs() const { return inputs_; };
    const std::vector<LazyTensor>& outputs() const { return outputs_; };
    const std::vector<LazyTensorId> output_lazy_tensor_ids() const { return output_nodes_; };

    void add_output_node(const LazyTensorId node_id) { output_nodes_.push_back(node_id); }

    void execute();

    bool is_materialized() const { return output_tensors_.size() > 0; }

    LazyTensorState get_state() const { return state_; };

    const LazyTensorId id() const { return id_; };

    ttnn::TensorSpec get_tensor_spec() const { return tensor_spec_; }

private:
    std::vector<tt::tt_metal::Tensor> output_tensors_;
    ttnn::TensorSpec tensor_spec_;
    std::shared_ptr<ttnn::experimental::jit::IDeviceOperation> args_;
    std::vector<LazyTensor> inputs_;
    std::vector<LazyTensor> outputs_;
    LazyTensorState state_;
    std::vector<LazyTensorId> output_nodes_;
    std::string operation_name_;
    LazyTensorId id_;
};

}  // namespace ttnn::experimental::jit
