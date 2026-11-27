// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttnn::experimental {

// With runtime_shapes, a tiled sharded input's slice of the same extent uses one program at every start and output
// size, which take their effect through runtime args, and its H and W ends may lie past the output (the W end past
// its whole tiles), where the slice is clipped. It costs host time on every call, to recompute the per-core args.

ttnn::Tensor slice_write(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& output_tensor,
    const ttsl::SmallVector<uint32_t>& begins,
    const ttsl::SmallVector<uint32_t>& ends,
    const ttsl::SmallVector<uint32_t>& step,
    bool runtime_shapes = false);

}  // namespace ttnn::experimental
