// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tile_constants.h"

namespace kda {

// Packed affine summaries are [S, N, K, K+V]: transform t of slot s is transform s * N + t, stored as Kt
// consecutive rows of (Kt + Vt) tiles holding the complement form [E | B] (E = A - I), E at column 0 and B at column
// Kt.
template <uint32_t Kt, uint32_t Vt>
constexpr uint32_t summary_row_page(uint32_t transform, uint32_t row) {
    return (transform * Kt + row) * (Kt + Vt);
}

// Split one transform's [E | B] rows into its E [Kt, Kt] and B [Kt, Vt] buffers.
template <uint32_t Kt, uint32_t Vt, typename Accessor, typename DFB>
FORCE_INLINE void read_summary_transform(Noc& noc, const Accessor& summaries, DFB& a, DFB& b, uint32_t transform) {
    const uint32_t tile_bytes = a.get_entry_size();
    for (uint32_t row = 0; row < Kt; ++row) {
        const uint32_t row_page = summary_row_page<Kt, Vt>(transform, row);
        for (uint32_t column = 0; column < Kt; ++column) {
            noc.async_read(
                summaries,
                a,
                tile_bytes,
                {.page_id = row_page + column},
                {.offset_bytes = (row * Kt + column) * tile_bytes});
        }
        for (uint32_t column = 0; column < Vt; ++column) {
            noc.async_read(
                summaries,
                b,
                tile_bytes,
                {.page_id = row_page + Kt + column},
                {.offset_bytes = (row * Vt + column) * tile_bytes});
        }
    }
}

// One FP32 32x32 identity tile, with which compute converts between A and E = A - I on diagonal tiles in FP32 DST.
template <typename DFB>
FORCE_INLINE void seed_identity_tile(Noc& noc, DFB& identity) {
    constexpr uint32_t one_fp32 = __builtin_bit_cast(uint32_t, 1.0F);
    identity.reserve_back(1);
    noc.async_write_zeros(identity, identity.get_entry_size());
    noc.write_zeros_l1_barrier();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(identity.get_write_ptr());
    // A tile's diagonal lies in its top-left and bottom-right faces.
    for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; ++row) {
        words[row * tt::constants::FACE_WIDTH + row] = one_fp32;
        words[3 * tt::constants::FACE_HW + row * tt::constants::FACE_WIDTH + row] = one_fp32;
    }
    identity.push_back(1);
}

}  // namespace kda
