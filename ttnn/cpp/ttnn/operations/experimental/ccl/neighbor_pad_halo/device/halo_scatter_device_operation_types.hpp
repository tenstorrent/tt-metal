// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Local (no-fabric) border scatter: copy the compact halo buffer
// [H-top | H-bot | W-left | W-right] produced by neighbor_pad_halo into the BORDER of a persistent
// padded buffer [outer, H+2pH, W+2pW, C] IN PLACE — the interior (already written by the previous
// conv's padded-output mode) is left untouched. The next conv then reads the padded buffer as a
// plain coalesced conv (pad=0), avoiding both the interior copy AND the conv3d per-stick halo read.
//
// The scatter is the exact inverse of compact_halo_reference() in the NP-halo test: each compact
// stick maps to a fixed padded page, so it is a pure local DRAM->DRAM stick copy with no arithmetic.
struct NpHaloScatterParams {
    uint32_t np_padding_h;  // H halo rows per side (pH)
    uint32_t np_padding_w;  // W halo cols per side (pW)
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::make_tuple("np_padding_h", "np_padding_w");
    auto attribute_values() const { return std::forward_as_tuple(np_padding_h, np_padding_w); }
};

struct NpHaloScatterInputs {
    Tensor compact_buffer;  // [total_sticks, C] compact halo buffer (source, from neighbor_pad_halo)
    Tensor padded_buffer;   // [outer, H+2pH, W+2pW, C] persistent padded buffer (in place; also output)
};

}  // namespace ttnn::experimental::prim
