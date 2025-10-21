// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/jit/node.hpp"

namespace ttnn::experimental::jit {
Node::Node(
    NodeId id,
    const std::vector<Tensor>& inputs,
    const std::string&& operation_name,
    std::shared_ptr<IDeviceOperation>&& args) :
    id_(id), operation_name_(std::move(operation_name)) {
    Args = std::move(args);
    inputs_.reserve(inputs.size());
    for (const auto& input : inputs) {
        inputs_.push_back(input);
    }

    TT_FATAL(Args != nullptr, "Args is not set");
    auto output_tensors = Args->create_output_tensors(inputs_);
    for (auto& output_tensor : output_tensors) {
        output_tensor.set_producer_node(id_);
    }

    Args->set_output_tensors(output_tensors);
}

NodeId Node::id() const { return id_; }

std::string_view Node::operation_name() const { return operation_name_; }

const std::vector<ttnn::Tensor>& Node::inputs() const { return inputs_; }

const std::vector<NodeId>& Node::output_nodes() const { return output_nodes_; }

void Node::add_output_node(NodeId node_id) { output_nodes_.push_back(node_id); }

void Node::execute() {
    TT_FATAL(Args != nullptr, "Args is not set");
    Args->invoke(inputs_);
}

}  // namespace ttnn::experimental::jit
