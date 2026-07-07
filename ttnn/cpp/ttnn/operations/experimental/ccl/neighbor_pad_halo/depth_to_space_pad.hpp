// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Fused depth-to-space + pad for the persistent-padded activation pipeline. Takes a conv output
// [B,T,H,W, p1*p2*p3*C] (channel order p1,p2,p3,C) and writes the depth-to-space result DIRECTLY into
// a newly-allocated padded buffer [B, T*p1-drop_first, H*p2+2pH, W*p3+2pW, C] interior — no dense
// depth-to-space intermediate and no separate pad copy. The border is left uninitialized for a later
// neighbor_pad_halo + halo_scatter(border_only) to fill (same contract as a conv padded-output). This
// eliminates the upsample-boundary interior copy in the LTX VAE decode.
//
// Args:
//   conv_out : conv output [B,T,H,W, p1*p2*p3*C] row-major
//   p1,p2,p3 : depth-to-space strides (temporal, height, width)
//   np_padding_h/w : halo padding per side for the output (must match the next conv's halo padding)
//   drop_first : drop the first output frame (causal temporal-upsample artifact); requires p1==2
//
// Returns: a padded buffer with the depth-to-space interior written (border uninitialized).
ttnn::Tensor depth_to_space_pad(
    const ttnn::Tensor& conv_out,
    uint32_t p1,
    uint32_t p2,
    uint32_t p3,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    bool drop_first = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
