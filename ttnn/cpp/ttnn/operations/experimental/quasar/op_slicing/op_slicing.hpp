// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

// Reuse the arch-neutral base types (Op2DSliceConfig / OpSliceAttr / determine_slice_config).
// This quasar fork only overrides run_sliced_op so its per-slice padded_slice / slice_write
// unconditionally route through the quasar Metal-2 ops (the shared ops build a legacy
// DataMovementKernel that Quasar rejects).
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"

namespace ttnn::operations::experimental::quasar::op_slicing {

// Re-export the arch-neutral base types so callers inside the experimental::quasar namespace
// (e.g. conv2d.cpp) can refer to them via the unqualified `op_slicing::` prefix, which now
// resolves to THIS namespace rather than the shared ttnn::operations::op_slicing one.
using ttnn::operations::op_slicing::Op2DSliceConfig;
using ttnn::operations::op_slicing::OpSliceAttr;

void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<ttnn::operations::op_slicing::OpSliceAttr::RefTensor>& output_tensors,
    ttnn::operations::op_slicing::OpSliceAttr* op_slice_attr,
    std::optional<ttnn::operations::op_slicing::Op2DSliceConfig> dram_slice_config_);

}  // namespace ttnn::operations::experimental::quasar::op_slicing
