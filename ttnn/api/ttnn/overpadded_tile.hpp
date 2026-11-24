// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation::detail {

// A TILE tensor is normally stored in the logical shape rounded up to whole 32x32 tiles. Some
// interleaved tensors carry a whole extra tile of padding (for example logical height 32 padded to
// 64). An op that sizes its output from the logical shape but counts its work from the padded shape
// then returns wrong values or writes past its output, so each such op rejects the input instead.
// Sharded tensors are exempt: shard alignment pads rows legitimately.
inline void validate_no_overpadded_tile_input(const Tensor& input_tensor, std::string_view operation_name) {
    if (input_tensor.layout() != Layout::TILE || input_tensor.is_sharded()) {
        return;
    }
    const auto logical_shape = input_tensor.logical_shape();
    const auto padded_shape = input_tensor.padded_shape();
    for (int offset = 1; offset <= 2; ++offset) {
        const int dim = static_cast<int>(logical_shape.rank()) - offset;
        const auto tile_aligned = tt::round_up(logical_shape[dim], tt::constants::TILE_HEIGHT);
        TT_FATAL(
            padded_shape[dim] <= tile_aligned,
            "Input tensor to {} has logical dimension {} of {} padded to {}, past the {} needed for whole tiles. "
            "Ops that size their output from the logical shape must not be given extra tile padding.",
            operation_name,
            dim,
            logical_shape[dim],
            padded_shape[dim],
            tile_aligned);
    }
}

}  // namespace ttnn::device_operation::detail
