// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// Local (no-fabric) border scatter for the persistent-padded activation pipeline.
//
// Copies the compact halo buffer [H-top | H-bot | W-left | W-right] produced by neighbor_pad_halo
// into the BORDER of a persistent padded buffer `padded_buffer` ([.., H+2pH, W+2pW, C]) IN PLACE.
// The padded interior is left untouched (it holds the previous conv's output, written via conv3d's
// padded-output mode), so the next conv can read `padded_buffer` as a plain coalesced conv (pad=0) —
// avoiding both the interior copy and the conv3d per-stick halo read.
//
// Args:
//   compact_buffer : the compact halo buffer returned by neighbor_pad_halo (source)
//   padded_buffer  : the persistent padded buffer (written in place; also the return value)
//   np_padding_h/w : halo rows/cols per side (must match the neighbor_pad_halo call)
//
// Returns: padded_buffer, with its border filled.
ttnn::Tensor halo_scatter(
    const ttnn::Tensor& compact_buffer,
    const ttnn::Tensor& padded_buffer,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
