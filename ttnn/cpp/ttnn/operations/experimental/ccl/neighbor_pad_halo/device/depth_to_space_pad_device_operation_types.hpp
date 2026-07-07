// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Fused depth-to-space + pad. Reads a conv output [B,T,H,W, p1*p2*p3*C] (channel order p1,p2,p3,C) and
// writes DIRECTLY into a newly-allocated padded buffer [B, T*p1-drop, H*p2+2pH, W*p3+2pW, C] interior —
// no dense depth-to-space intermediate + no separate pad copy. The border is left uninitialized (filled
// later by neighbor_pad_halo + halo_scatter border_only), exactly like a conv padded-output. This
// eliminates the upsample-boundary interior copy in the LTX VAE decode copy-free pipeline.
struct DepthToSpacePadParams {
    uint32_t p1;            // temporal stride
    uint32_t p2;            // height stride
    uint32_t p3;            // width stride
    uint32_t np_padding_h;  // pH per side
    uint32_t np_padding_w;  // pW per side
    uint32_t drop_first;    // 1 => drop first output frame (causal temporal upsample artifact)
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names =
        std::make_tuple("p1", "p2", "p3", "np_padding_h", "np_padding_w", "drop_first");
    auto attribute_values() const { return std::forward_as_tuple(p1, p2, p3, np_padding_h, np_padding_w, drop_first); }
};

struct DepthToSpacePadInputs {
    Tensor conv_out;  // [B,T,H,W, p1*p2*p3*C] row-major, channel order (p1,p2,p3,C)
};

}  // namespace ttnn::experimental::prim
