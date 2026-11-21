// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stack.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include <tt-metalium/tensor/tensor_types.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

namespace ttnn {

namespace {

// Stacking on a new trailing axis through unsqueeze-then-concat is pathological: for row-major the
// unsqueeze turns every input into one-element rows (sticks) on device and concat on that axis takes
// its transpose fallback; for tile layout the unsqueeze pads the two new inner dims to a full tile,
// up to a 32x blowup materialized per input. Unsqueezing at 0 is a view in both layouts, concat at 0
// copies whole pages, and one permute moves the new axis to the end.
Tensor stack_via_leading_axis(const std::vector<Tensor>& input_tensors, const int rank) {
    std::vector<Tensor> leading;
    leading.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        leading.push_back(ttnn::unsqueeze(tensor, 0));
    }
    const Tensor stacked = ttnn::concat(leading, 0);
    ttsl::SmallVector<int64_t> dims(rank);
    std::iota(dims.begin(), dims.end(), 1);
    dims.push_back(0);
    return ttnn::permute(stacked, dims);
}

}  // namespace

Tensor stack(const std::vector<Tensor>& input_tensors, const int dim) {
    TT_FATAL(!input_tensors.empty(), "Stack expects at least one tensor");
    const auto& first = input_tensors.front();
    const int rank = static_cast<int>(first.logical_shape().rank());
    // A negative dim indexes the rank+1 output, since the new axis counts.
    const int normalized_dim = dim < 0 ? dim + rank + 1 : dim;
    TT_FATAL(
        0 <= normalized_dim && normalized_dim <= rank,
        "Stack dim {} is out of range for rank {} inputs (expected [{}, {}])",
        dim,
        rank,
        -(rank + 1),
        rank);
    // Concat divides by the padded last dim of its output, so an empty input would end in SIGFPE.
    TT_FATAL(first.logical_volume() > 0, "Stack does not support zero-volume inputs (shape {})", first.logical_shape());
    // A new trailing axis goes through stack_via_leading_axis. Kept on unsqueeze-then-concat on
    // purpose: block floats, because for rank >= 4 inputs the stacked tensor is rank 5 and
    // ttnn::permute hands rank > 4 to prim::permute without its bfloat8_b typecast (lower ranks
    // would be fine; excluded at every rank for uniformity); rank-1 tile inputs, since a [W] tile
    // tensor is physically a padded [32, W] row, so concat at 0 would untilize and re-tilize, which
    // measures slower than the existing path. Sharded, mixed-layout and rank-0 inputs also keep it:
    // concat validates the first two, and for rank 0 both routes are the same ops.
    const auto layout = first.layout();
    const bool uniform_interleaved =
        std::all_of(input_tensors.begin(), input_tensors.end(), [layout](const Tensor& tensor) {
            return tensor.layout() == layout && !tensor.is_sharded();
        });
    const bool trailing_axis_on_pages =
        (layout == Layout::ROW_MAJOR && rank >= 1) || (layout == Layout::TILE && rank >= 2);
    // One input on a trailing axis is only a reshape, and the leading-axis route would spend a
    // permute on it (permute's no-op check refuses to move a last axis even of size 1). For tile
    // the one tiled reshape writes the same padded output as that permute and measures cheaper, so
    // every single tile input keeps unsqueeze-then-concat. For row-major the reshape is a view when
    // the last dim is 1 and otherwise the one-element-stick rewrite, so only the view case keeps it.
    const auto& shape = first.logical_shape();
    const bool single_input_prefers_reshape =
        input_tensors.size() == 1 && rank >= 1 && (layout == Layout::TILE || shape[-1] == 1);
    if (uniform_interleaved && !single_input_prefers_reshape && !tt::tt_metal::is_block_float(first.dtype()) &&
        trailing_axis_on_pages && normalized_dim == rank) {
        return stack_via_leading_axis(input_tensors, rank);
    }
    std::vector<Tensor> expanded_tensors;
    expanded_tensors.reserve(input_tensors.size());
    for (const auto& tensor : input_tensors) {
        expanded_tensors.push_back(ttnn::unsqueeze(tensor, normalized_dim));
    }
    return ttnn::concat(expanded_tensors, normalized_dim);
}

}  // namespace ttnn
