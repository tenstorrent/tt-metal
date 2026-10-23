// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// Writes row i of input [1, 1, n, W] into row rows[i] of output [1, 1, R, W], in place; both ROW_MAJOR DRAM, same
// dtype. rows is a device tensor of n uint32/int32 row indices (in its last dimension), so a captured trace writes
// wherever the replay's indices point, and the cost is the n rows written rather than R. An index at or past R is
// skipped. Two indices naming the same output row leave either input row there. Returns output.
ttnn::Tensor deepseek_v4_prefill_write_rows(
    const ttnn::Tensor& input, const ttnn::Tensor& output, const ttnn::Tensor& rows);

}  // namespace ttnn::transformer
