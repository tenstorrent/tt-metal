// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <vector>

namespace tt::tt_metal::experimental {

using NodeId = uint32_t;

class Node {
public:
    template <typename ArgsT>
    Node(NodeId id, const std::vector<Tensor>& inputs, const std::string&& operation_name, ArgsT&& args) : id_(id) {
        TT_ASSERT(sizeof(ArgsT) <= sizeof(args_storage_), "Args too large for inline storage");

        // Copy inputs to our larger container
        for (const auto& input : inputs) {
            inputs_.push_back(input);
        }

        operation_name_ = std::move(operation_name);

        new (args_storage_) std::decay_t<ArgsT>(std::forward<ArgsT>(args));
    }

    NodeId id() const;

    std::string_view operation_name() const;

    const std::vector<Tensor>& inputs() const;
    const std::vector<NodeId>& output_nodes() const;

    void add_output_node(NodeId node_id);

private:
    std::string operation_name_;
    NodeId id_;
    std::vector<Tensor> inputs_;
    std::vector<NodeId> output_nodes_;
    static constexpr size_t ARGS_STORAGE_SIZE = 256;

    // Not on my watch, but I need to think how to replace it
    alignas(std::max_align_t) char args_storage_[ARGS_STORAGE_SIZE];  // NOLINT(cppcoreguidelines-avoid-c-arrays) - Type
                                                                      // erasure storage requires C-style array
};

}  // namespace tt::tt_metal::experimental
