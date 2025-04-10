// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stack.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "cpp/ttnn/operations/data_movement/concat/concat.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor StackOperation::invoke(const std::vector<ttnn::Tensor>& input_tensors, const int dim) {
    TT_FATAL(!input_tensors.empty(), "Stack expects at least one tensor");
    std::vector<ttnn::Tensor> expanded_tensors;
    for (const auto& tensor : input_tensors) {
        expanded_tensors.push_back(ttnn::unsqueeze(tensor, dim));
    }
    return ttnn::concat(expanded_tensors, dim);
}

}  // namespace ttnn::operations::data_movement
