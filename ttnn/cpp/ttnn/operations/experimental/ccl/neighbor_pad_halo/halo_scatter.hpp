// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Local (no-fabric) repack for the persistent-padded activation pipeline. Allocates a padded buffer
// [B,T,H+2pH,W+2pW,C] and fills it in one pass: INTERIOR from `interior_src` (the unpadded
// activation) and BORDER from the compact halo buffer [H-top | H-bot | W-left | W-right] produced by
// neighbor_pad_halo. Folds the old ttnn.pad (interior copy) + border scatter into one op; the next
// conv reads the result as a plain coalesced conv (pad=0).
//
// Args:
//   compact_buffer : the compact halo buffer returned by neighbor_pad_halo (border source)
//   interior_src   : the unpadded activation [B,T,H,W,C] (interior source)
//   np_padding_h/w : halo rows/cols per side (must match the neighbor_pad_halo call)
//
// Returns: a newly-allocated padded buffer with interior + border filled.
ttnn::Tensor halo_scatter(
    const ttnn::Tensor& compact_buffer,
    const ttnn::Tensor& interior_src,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool border_only = false);

}  // namespace ttnn::experimental
