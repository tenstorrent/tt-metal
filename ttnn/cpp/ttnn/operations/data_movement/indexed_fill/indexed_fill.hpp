// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

// Replaces slices of input_tensor_a along `dim` with the corresponding slices from
// input_tensor_b, where the indices of slices to replace are given by batch_id.
//
// Mirrors torch.Tensor.index_copy_:
//   out = input_tensor_a.clone()
//   for i, idx in enumerate(batch_id.flatten()):
//       out.select(dim, idx).copy_(input_tensor_b.select(dim, i))
//
// Supported configurations:
//   Layouts    : ROW_MAJOR, TILE (both inputs must match)
//   Rank       : exactly 4 ([B, H, W, D] layout required)
//   Memory     : INTERLEAVED or HEIGHT/WIDTH/BLOCK_SHARDED, L1 or DRAM
//   dim        : any axis (dim < rank-1 for ROW_MAJOR, dim < rank-2 for TILE handled natively;
//                sub-page dims use a permute-based fallback)
//   batch_id   : must be ROW_MAJOR, UINT32 or INT32
//
// Known limitations:
//   - Sharded input_a in the generic fallback path (non-native geometry) produces wrong
//     results and is not supported. Use INTERLEAVED input_a if the geometry doesn't satisfy
//     the native HEIGHT_SHARDED fast path.
//   - Dtypes beyond bfloat16 are untested (should work via CB format inference).
//   - The generic path supports any S_dim via work-splitting (multiple slices per core).
Tensor indexed_fill(
    const Tensor& batch_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    int64_t dim = 0);

}  // namespace ttnn
