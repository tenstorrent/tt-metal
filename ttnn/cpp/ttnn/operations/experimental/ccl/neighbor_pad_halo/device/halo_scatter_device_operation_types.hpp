// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Local (no-fabric) repack into a padded buffer [outer, H+2pH, W+2pW, C]: fill the INTERIOR from the
// unpadded activation `x` and the BORDER from the compact halo buffer [H-top | H-bot | W-left |
// W-right] produced by neighbor_pad_halo, in a single pass that writes every padded page once. This
// folds the old ttnn.pad (interior copy) + border scatter into one op; the next conv then reads the
// padded buffer as a plain coalesced conv (pad=0). The mapping is the inverse of
// compact_halo_reference() for the border, a plain strided placement for the interior — pure
// DRAM->DRAM stick copies, no arithmetic. The op ALLOCATES its padded output (uninitialized; every
// page is written), so no separate pad/zero pass is needed.
struct NpHaloScatterParams {
    uint32_t np_padding_h;  // H halo rows per side (pH)
    uint32_t np_padding_w;  // W halo cols per side (pW)
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::make_tuple("np_padding_h", "np_padding_w");
    auto attribute_values() const { return std::forward_as_tuple(np_padding_h, np_padding_w); }
};

struct NpHaloScatterInputs {
    Tensor compact_buffer;  // [total_sticks, C] compact halo buffer (border source, from neighbor_pad_halo)
    Tensor interior_src;    // [outer, H, W, C] unpadded activation (interior source)
};

}  // namespace ttnn::experimental::prim
