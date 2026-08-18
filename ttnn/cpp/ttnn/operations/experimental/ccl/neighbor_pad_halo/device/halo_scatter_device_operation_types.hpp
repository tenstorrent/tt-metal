// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Local (no-fabric) repack into a padded buffer [outer, H+2pH, W+2pW, C]
struct NpHaloScatterParams {
    uint32_t np_padding_h;  // H halo rows per side (pH)
    uint32_t np_padding_w;  // W halo cols per side (pW)
    tt::tt_metal::MemoryConfig output_mem_config;
    // border_only: copy-free mode
    bool border_only = false;

    static constexpr auto attribute_names = std::make_tuple("np_padding_h", "np_padding_w", "border_only");
    auto attribute_values() const { return std::forward_as_tuple(np_padding_h, np_padding_w, border_only); }
};

struct NpHaloScatterInputs {
    Tensor compact_buffer;  // [total_sticks, C] compact halo buffer (border source, from neighbor_pad_halo)
    Tensor interior_src;    // repack: unpadded [.,H,W,C] interior source. border_only: the padded buffer (in place).
};

}  // namespace ttnn::experimental::prim
