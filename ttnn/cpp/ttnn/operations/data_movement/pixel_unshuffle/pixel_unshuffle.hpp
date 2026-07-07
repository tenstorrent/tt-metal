// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

// How the R×R spatial block and the input channel are packed into the output
// channel axis. Both produce shape [N, C*r^2, H/r, W/r]; they differ only in
// which value lands at each output channel index (matters only when C > 1).
// Declaration order is significant: the value is passed to the kernels as a
// compile-time arg (CHANNEL_MAJOR -> 0, SPATIAL_MAJOR -> 1).
enum class PixelUnshuffleChannelOrder {
    // Input channel varies slowest: each channel's r^2 sub-pixels stay contiguous.
    //   c_out = c_in*r^2 + rh*r + rw
    // Matches torch.nn.functional.pixel_unshuffle. This is the default.
    CHANNEL_MAJOR,

    // Spatial sub-position varies slowest: input channels are interleaved across sub-pixels.
    //   c_out = rh*(r*C) + rw*C + c_in
    // Matches ONNX SpaceToDepth channel ordering.
    SPATIAL_MAJOR,
};

// pixel_unshuffle: NCHW input [N, C, H, W] -> [N, C*r^2, H/r, W/r]
//
// Equivalent to torch.nn.functional.pixel_unshuffle(input, downscale_factor)
// when channel_order == CHANNEL_MAJOR (the default).
// H and W must be divisible by downscale_factor.
//
// Input layout:  TILE or ROW_MAJOR — both accepted; TILE is untilized internally.
// Input memory:  DRAM or L1, interleaved or sharded.
//                Sharded input is accepted natively: TensorAccessor resolves page_id
//                across cores via NOC. No intermediate DRAM copy is performed.
// Output layout: ROW_MAJOR by default; pass output_layout=TILE for TILE output.
// Output memory: Controlled by memory_config (DRAM, L1, or sharded L1).
// channel_order: CHANNEL_MAJOR (PyTorch, default) or SPATIAL_MAJOR (ONNX).
Tensor pixel_unshuffle(
    const Tensor& input_tensor,
    uint32_t downscale_factor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Layout>& output_layout = std::nullopt,
    PixelUnshuffleChannelOrder channel_order = PixelUnshuffleChannelOrder::CHANNEL_MAJOR);

}  // namespace ttnn
