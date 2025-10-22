// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/jit/node.hpp"
#include "ttnn/lazy_mode.hpp"

namespace ttnn::experimental::jit {
Node::Node(
    NodeId id,
    const std::vector<Tensor>& inputs,
    const std::string&& operation_name,
    std::shared_ptr<IDeviceOperation>&& Args) :
    id_(id), operation_name_(operation_name), Args(std::move(Args)) {
    inputs_.reserve(inputs.size());
    for (const auto& input : inputs) {
        inputs_.push_back(input);
    }
}

NodeId Node::id() const { return id_; }

std::string_view Node::operation_name() const { return operation_name_; }

const std::vector<ttnn::Tensor>& Node::inputs() const { return inputs_; }

std::vector<ttnn::Tensor>& Node::inputs_mut() { return inputs_; }

const std::vector<NodeId>& Node::output_nodes() const { return output_nodes_; }

void Node::add_output_node(NodeId node_id) { output_nodes_.push_back(node_id); }

void Node::execute() {
    TT_FATAL(Args != nullptr, "Args is not set");
    // Temporarily disable lazy mode during execution to prevent re-entry
    ttnn::lazy_mode::ScopedDisable disable_lazy;

    // Execute the operation with the (now materialized) inputs
    outputs_ = Args->invoke(inputs_);

    is_materialized_ = true;
}

}  // namespace ttnn::experimental::jit
