// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// matmul_decode: decode-optimized matrix multiply C = A @ B for width-sharded operands.
//
// Both inputs are L1 width-sharded. A ([M, K]) is width(K)-sharded and B is width-sharded;
// the result ([M, N]) is L1 width-sharded across the core grid. Backed by the
// MatmulDecodeDeviceOperation (ttnn::prim::matmul_decode) with two program factories:
//   - default: B is width(N)-sharded; the full A is gathered onto every core.
//   - partial_width_sharded: B is sharded along both K and N, with a cross-core
//     K-reduction onto the output cores (see PartialWidthSharded for the B layout).
//   - batched (rank-3 A [batch, M, K]): B is folded along both batch and N across
//     b_blocks * n_blocks cores; each core computes an independent [Bc, M, Nc] block
//     (block-diagonal, no reduction). The fold geometry (Bc, b_blocks, N, n_blocks) is
//     inferred from the operand shapes. See BatchedWidthSharded for the B layout.
Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::experimental
