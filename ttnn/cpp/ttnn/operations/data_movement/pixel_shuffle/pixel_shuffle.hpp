// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle.hpp"

namespace ttnn {

// pixel_shuffle: the inverse of pixel_unshuffle. NCHW input [N, C*r^2, H, W] -> [N, C, H*r, W*r]
//
// Equivalent to torch.nn.functional.pixel_shuffle(input, upscale_factor) when
// channel_order == CHANNEL_MAJOR (the default). The number of input channels must be divisible by r^2.
//
// channel_order says which input channel holds each output (channel, sub-pixel), the same packing
// pixel_unshuffle uses, so pixel_shuffle(pixel_unshuffle(x, r, order), r, order) == x for both:
//   CHANNEL_MAJOR: output[n, c, h*r+rh, w*r+rw] = input[n, c*r^2 + rh*r + rw, h, w]   (torch)
//   SPATIAL_MAJOR: output[n, c, h*r+rh, w*r+rw] = input[n, rh*(r*C) + rw*C + c, h, w]  (ONNX DepthToSpace, DCR)
//
// Input layout:  TILE or ROW_MAJOR (TILE is untilized internally).
// Input memory:  DRAM or L1, interleaved or sharded (the kernel addresses pages through a TensorAccessor).
// Output layout: ROW_MAJOR by default; output_layout=TILE tilizes the result.
// Output memory: memory_config (default: the input's memory config when it is interleaved, else DRAM).
//
// channels_last: the activation is NHWC on both sides, the layout a height-sharded conv2d produces and
//   consumes: input [N, H, W, C*r^2] ROW_MAJOR HEIGHT_SHARDED in L1 (e.g. the row-major output of a 1x1
//   matmul), output [N, H*r, W*r, padded_channels] ROW_MAJOR HEIGHT_SHARDED in L1 (memory_config, shard
//   [pixels_per_core, padded_channels]) with the C output channels of every pixel contiguous and zero-padded
//   up to padded_channels (default: C rounded up to the L1 alignment). It is the inverse of
//   pixel_unshuffle(..., channels_last=True) and turns the 1x1 matmul C_in -> C*r^2 into a stride-r
//   transposed conv with an r x r kernel (SPATIAL_MAJOR: W_1x1[(dy*r+dx)*C + co, ci] = W_t[ci, co, dy, dx]).
//   Every core reads its own input shard and writes the output sticks with NoC writes; for SPATIAL_MAJOR
//   without padding the r*C elements of one kernel row are one write.
Tensor pixel_shuffle(
    const Tensor& input_tensor,
    uint32_t upscale_factor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Layout>& output_layout = std::nullopt,
    PixelUnshuffleChannelOrder channel_order = PixelUnshuffleChannelOrder::CHANNEL_MAJOR,
    bool channels_last = false,
    const std::optional<uint32_t>& padded_channels = std::nullopt);

}  // namespace ttnn
