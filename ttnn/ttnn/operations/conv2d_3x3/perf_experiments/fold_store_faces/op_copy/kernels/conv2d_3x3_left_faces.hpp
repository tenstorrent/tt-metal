// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Left-face output store (OUT_LEFT_FACES): an output tile whose real lanes all sit in columns [0, 16) (the last N tile
// where C_out % 32 is 1..16, so every tile of the C_out fold) holds output only in faces 0 and 2. The store writes
// those two faces and leaves faces 1 and 3 (columns 16..31, tile padding) as the output buffer held them; lanes
// [C_out % 32, 16) of faces 0 and 2 are written (zero, as packed). The logical output is unchanged: ttnn leaves tile
// padding unspecified, and the ops that need a neutral value there fill it (ttnn.fill_implicit_tile_padding).
//
// Face byte ranges: bf16 tile (2048 B) faces 0 / 2 at [0, 512) / [1024, 1536); bfp8_b tile (1088 B: 64 B of
// exponents, 16 per face, then 256 B of mantissas per face) the exponents and face 0's mantissas at [0, 320), face 2's
// mantissas at [576, 832). A bfp8_b face row shares one exponent, so the unwritten faces never share an exponent with
// an output lane.
//
// Measured (Blackhole p150b, 110 cores, bit-identical logical output, us): 1026x2048 128->3 (fold) 1824 -> 1698,
// 514x1024 128->3 (fold) 527 -> 498, TILE-input 1024x1024 128->3 1297 -> 1238. One run through face 2 ([0, 1536))
// writes 75% of the tile and measured 1721 on the first: the two-run store's extra NoC write costs less than the face
// it skips.
//
// Raw NoC (ergonomics): noc_async_write_page moves a whole page; a sub-page run is the page's NoC address plus an
// offset (TensorAccessor::get_noc_addr) and a sized noc_async_write.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace conv2d_3x3 {

template <uint32_t tile_bytes, bool bfp>
struct LeftFaces {
    static_assert(bfp ? tile_bytes == 64 + 4 * 256 : tile_bytes % 4 == 0);
    static constexpr uint32_t face_bytes = bfp ? 256 : tile_bytes / 4;
    // Run A [0, a_len): the exponents (bfp8_b) and face 0; run B [b_off, b_off + face_bytes): face 2.
    static constexpr uint32_t a_len = (bfp ? 64 : 0) + face_bytes;
    static constexpr uint32_t b_off = (bfp ? 64 : 0) + 2 * face_bytes;

    // Faces 0 and 2 of the tile at L1 src into the page whose NoC address is dst.
    static FORCE_INLINE void write(uint32_t src, uint64_t dst) {
        noc_async_write(src, dst, a_len);
        noc_async_write(src + b_off, dst + b_off, face_bytes);
    }
};

}  // namespace conv2d_3x3
