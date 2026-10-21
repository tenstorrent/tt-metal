// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stack.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

namespace ttnn {

namespace {

// Stacking row-major inputs on a new trailing axis through unsqueeze-then-concat is pathological:
// the unsqueeze turns every input into one-element sticks on device, and concat on that axis
// takes its transpose fallback. The same result is a free view (unsqueeze at 0), a whole-stick
// concat, and one permute that moves the new axis to the end.
Tensor stack_row_major_on_trailing_axis(const std::vector<Tensor>& input_tensors, const int rank) {
    std::vector<Tensor> leading;
    leading.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        leading.push_back(ttnn::unsqueeze(tensor, 0));
    }
    const Tensor stacked = ttnn::concat(leading, 0);
    ttsl::SmallVector<int64_t> dims;
    dims.reserve(rank + 1);
    for (int axis = 1; axis <= rank; ++axis) {
        dims.push_back(axis);
    }
    dims.push_back(0);
    return ttnn::permute(stacked, dims);
}

}  // namespace

Tensor stack(const std::vector<Tensor>& input_tensors, const int dim) {
    TT_FATAL(!input_tensors.empty(), "Stack expects at least one tensor");
    const auto& first = input_tensors.front();
    const int rank = static_cast<int>(first.logical_shape().rank());
    const int normalized_dim = dim < 0 ? dim + rank + 1 : dim;
    TT_FATAL(
        0 <= normalized_dim && normalized_dim <= rank,
        "Stack dim {} is out of range for rank {} inputs (expected [{}, {}])",
        dim,
        rank,
        -(rank + 1),
        rank);
    if (first.layout() == Layout::ROW_MAJOR && rank >= 1 && normalized_dim == rank) {
        return stack_row_major_on_trailing_axis(input_tensors, rank);
    }
    std::vector<Tensor> expanded_tensors;
    expanded_tensors.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        expanded_tensors.push_back(ttnn::unsqueeze(tensor, dim));
    }
    return ttnn::concat(expanded_tensors, dim);
}

}  // namespace ttnn
