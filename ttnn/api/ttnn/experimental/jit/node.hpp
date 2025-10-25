// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/jit/to_organize.hpp"
#include "ttnn/experimental/jit/IDeviceOperation.hpp"

namespace ttnn::experimental::jit {

class Node {
public:
    Node() = delete;

    Node(
        NodeId id,
        const std::vector<Tensor>& inputs,
        const std::string&& operation_name,
        std::shared_ptr<IDeviceOperation>&& Args);

    NodeId id() const;

    std::string_view operation_name() const;

    const std::vector<Tensor>& inputs() const;
    const std::vector<NodeId>& output_nodes() const;

    void add_output_node(NodeId node_id);

    void execute();

    void set_is_materialized(bool is_materialized) { is_materialized_ = is_materialized; }
    bool is_materialized() const { return is_materialized_; }

    const std::vector<Tensor>& outputs() const { return outputs_; }
    std::vector<Tensor> create_output_tensors() const { return Args->create_output_tensors(inputs_); }

private:
    NodeId id_;
    std::string operation_name_;
    std::vector<Tensor> inputs_;
    std::vector<NodeId> output_nodes_;
    std::shared_ptr<IDeviceOperation> Args;
    bool is_materialized_ = false;
    std::vector<Tensor> outputs_;
};

}  // namespace ttnn::experimental::jit
