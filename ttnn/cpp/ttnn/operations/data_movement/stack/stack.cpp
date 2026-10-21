// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stack.hpp"

#include <algorithm>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

namespace ttnn {

namespace {

// Stacking on a new trailing axis through unsqueeze-then-concat is pathological: for row-major
// the unsqueeze turns every input into one-element sticks on device and concat on that axis takes
// its transpose fallback; for tile layout the unsqueeze pads the two new inner dims to a full tile,
// a 32x blowup materialized per input. Unsqueezing at 0 is a view in both layouts, concat at 0
// copies whole pages, and one permute moves the new axis to the end.
Tensor stack_via_leading_axis(const std::vector<Tensor>& input_tensors, const int rank) {
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
    // Sharded or mixed-layout inputs keep the unsqueeze-then-concat path, where concat's own checks
    // decide what is supported. Rank-1 tile inputs keep it too: a [W] tile tensor is physically a
    // padded [32, W] row, so concat at 0 would cross a padded axis and run untilize and re-tilize.
    const auto layout = first.layout();
    const bool uniform_interleaved =
        std::all_of(input_tensors.begin(), input_tensors.end(), [layout](const Tensor& tensor) {
            return tensor.layout() == layout && !tensor.memory_config().is_sharded();
        });
    const bool trailing_axis_on_pages =
        (layout == Layout::ROW_MAJOR && rank >= 1) || (layout == Layout::TILE && rank >= 2);
    if (uniform_interleaved && trailing_axis_on_pages && normalized_dim == rank) {
        return stack_via_leading_axis(input_tensors, rank);
    }
    std::vector<Tensor> expanded_tensors;
    expanded_tensors.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        expanded_tensors.push_back(ttnn::unsqueeze(tensor, dim));
    }
    return ttnn::concat(expanded_tensors, dim);
}

}  // namespace ttnn
