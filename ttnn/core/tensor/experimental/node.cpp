// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/experimental/node.hpp"

namespace tt::tt_metal::experimental {
NodeId Node::id() const { return id_; }

std::string_view Node::operation_name() const { return operation_name_; }

const std::vector<Tensor>& Node::inputs() const { return inputs_; }

const std::vector<NodeId>& Node::output_nodes() const { return output_nodes_; }

void Node::add_output_node(NodeId node_id) { output_nodes_.push_back(node_id); }

}  // namespace tt::tt_metal::experimental
