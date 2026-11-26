// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul regime R1' (alignment = m_non_aligned): re-tile one quarter of a head's rows.
//
// Destination chip t's quarter is rows [t*Mq, (t+1)*Mq) of the head's S rows. When Mq % 32 != 0 the quarter starts at
// row shift = (t*Mq) % 32 of source tile row T0 = (t*Mq) / 32, so destination tile (mt, k) takes source rows
// shift + mt*32 + r (r < 32) from source tile rows T0 + mt and T0 + mt + 1. Rows past the quarter's end (the last tile
// row when Mq % 32 != 0) are padding and are zeroed. Shared by the port reader (remote quarters) and the compute-core
// reader (the local quarter); landing, arrival counters and compute see ordinary tiles (Mq_t = ceil(Mq/32) per image).
//
// Usage: stage_rows() DRAM-reads the nsrc x Kt source tiles of one image into a staging area (one barrier), then
// retile_tile() issues core-local NoC copies (contiguous face-row runs, <= 6 per destination tile + the zero fill) into
// the destination page. The copies complete at the caller's next noc_async_read_barrier(); stage_rows() starts with
// one, so the staging area is never overwritten under an in-flight copy.
//
// Face geometry is for a non-block 32x32 tile (4 faces of 16x16, row-major faces): row_bytes = tile_bytes / 64.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace a2amm_retile {

struct Quarter {
    uint32_t t0;     // first source tile row of the quarter
    uint32_t shift;  // first row of the quarter inside source tile row t0
    uint32_t nsrc;   // source tile rows the quarter spans (<= Mq_t + 1)
};

template <uint32_t Mq>
inline Quarter quarter_of(uint32_t t) {
    const uint32_t row0 = t * Mq;
    const uint32_t t0 = row0 / 32;
    return Quarter{t0, row0 % 32, (row0 + Mq - 1) / 32 - t0 + 1};
}

// DRAM -> staging: the quarter's nsrc x Kt source tiles of one image (consecutive pages from first_page).
template <uint32_t tile_bytes, uint32_t Kt, typename Accessor>
inline void stage_rows(const Accessor& in, uint32_t first_page, uint32_t nsrc, uint32_t stage) {
    noc_async_read_barrier();  // copies still reading the previous image's staging area
    const uint32_t n = nsrc * Kt;
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(in.get_noc_addr(first_page + i), stage + i * tile_bytes, tile_bytes);
    }
    noc_async_read_barrier();
}

// staging -> destination tile (mt, k): shifted face-row runs, padding rows zeroed. Core-local NoC reads.
template <uint32_t tile_bytes, uint32_t Kt, uint32_t Mq>
inline void retile_tile(uint32_t stage, uint32_t dst, uint32_t shift, uint32_t mt, uint32_t k) {
    constexpr uint32_t face_bytes = tile_bytes / 4;
    constexpr uint32_t row_bytes = face_bytes / 16;
    static_assert(face_bytes <= MEM_ZEROS_SIZE, "a zero run is at most one face");
    const uint32_t left = Mq - mt * 32;
    const uint32_t valid = left < 32 ? left : 32;
    uint32_t r = 0;
    while (r < valid) {
        const uint32_t sr = shift + mt * 32 + r;
        const uint32_t rr = sr & 31;
        const uint32_t src_tile = stage + ((sr >> 5) * Kt + k) * tile_bytes;
        uint32_t n = 16 - (r & 15);
        const uint32_t ns = 16 - (rr & 15);
        n = ns < n ? ns : n;
        n = (valid - r) < n ? (valid - r) : n;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t s = src_tile + ((rr >> 4) * 2 + half) * face_bytes + (rr & 15) * row_bytes;
            const uint32_t d = dst + ((r >> 4) * 2 + half) * face_bytes + (r & 15) * row_bytes;
            noc_async_read(get_noc_addr(s), d, n * row_bytes);
        }
        r += n;
    }
    while (r < 32) {  // padding rows: zero fill from the core's zero region (a run never exceeds one face)
        const uint32_t n = 16 - (r & 15);
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t d = dst + ((r >> 4) * 2 + half) * face_bytes + (r & 15) * row_bytes;
            noc_async_read(get_noc_addr(MEM_ZEROS_BASE), d, n * row_bytes);
        }
        r += n;
    }
}

}  // namespace a2amm_retile
