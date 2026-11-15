// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// One decode step's small per-step tensors, computed by one program from the step's INT32 ROW_MAJOR packet
// [1, 1, 1, W] (DRAM interleaved): integer index tensors and RoPE rows. Outputs are DRAM interleaved: first one
// ROW_MAJOR int_dtypes[i] tensor of int_shapes[i] per integer output, then a (cos, sin, -sin) triple of
// [1, 1, 1, Rd] BFLOAT16 TILE rows per RoPE output.
//
// Each element of an integer output is a term of 7 int32 words {w0, pre_add, div, mul, post_add, clamp_max,
// add_mul}, w0 = (src_field + 1) | mode << 8 | (add_field + 1) << 16 (a field of -1 reads 0):
//   v = pkt[src_field] + pre_add; v = floor(v / div) (mode 1) or v mod div (mode 2, floored);
//   v = min(v * mul + post_add, clamp_max) + pkt[add_field] * add_mul.
//
// A RoPE table [1, 1, H + 1025, 2 * Rd] INT32 (Q28 fixed point) holds rows [cos | sin] of h * 1024 * w for
// h < H, then scale * [cos | sin] of l * w for l = -1 .. 1023; position p = pkt[field] + offset (>= -1) takes
// the angle-sum of its 1024-block row and its offset row.
std::vector<Tensor> decode_step_context(
    const Tensor& packet,
    const std::vector<Tensor>& rope_tables,
    const std::vector<std::vector<uint32_t>>& int_shapes,
    const std::vector<DataType>& int_dtypes,
    const std::vector<std::vector<int32_t>>& int_terms,
    const std::vector<uint32_t>& rope_table_ids,
    const std::vector<uint32_t>& rope_fields,
    const std::vector<int32_t>& rope_offsets);

}  // namespace ttnn::experimental::deepseek
