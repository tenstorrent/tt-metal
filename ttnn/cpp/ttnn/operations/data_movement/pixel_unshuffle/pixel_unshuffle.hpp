// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

// pixel_unshuffle: NCHW input [N, C, H, W] -> [N, C*r^2, H/r, W/r]
//
// Equivalent to torch.nn.functional.pixel_unshuffle(input, downscale_factor).
// H and W must be divisible by downscale_factor.
//
// Input layout:  TILE or ROW_MAJOR — both accepted; TILE is untilized internally.
// Input memory:  DRAM or L1, interleaved or sharded.
//                Sharded input is accepted natively: TensorAccessor resolves page_id
//                across cores via NOC. No intermediate DRAM copy is performed.
// Output layout: ROW_MAJOR by default; pass output_layout=TILE for TILE output.
// Output memory: Controlled by memory_config (DRAM, L1, or sharded L1).
Tensor pixel_unshuffle(
    const Tensor& input_tensor,
    uint32_t downscale_factor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Layout>& output_layout = std::nullopt);

}  // namespace ttnn
