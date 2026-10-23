// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::rebank_rm_merge.
//
// rebank_rm_merge is the inverse of rebank_rm: it converts a
// (B_total * chunks_per_merge, N1) ROW_MAJOR tensor with
// page_size = N1 * elem_bytes into a (B_total, N1 * chunks_per_merge)
// tensor with page_size = N1 * chunks_per_merge * elem_bytes.
//
// This avoids the L1 overflow caused by ttnn::reshape when the destination
// page exceeds L1 capacity.  The CB is tiny: 2 * N1 * elem_bytes.
//
// Constraint: chunks_per_merge must be a power-of-2; input must be 2D.

#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RebankRmMergeParams {
    uint32_t chunks_per_merge;  // number of source rows merged into one output row
};

struct RebankRmMergeTensorArgs {
    Tensor input;
};

}  // namespace ttnn::experimental::prim
