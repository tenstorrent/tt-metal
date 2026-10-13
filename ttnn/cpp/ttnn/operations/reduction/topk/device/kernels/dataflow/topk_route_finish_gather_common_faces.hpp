// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single face unit version of topk_route_finish_gather_common.hpp, which documents the row split and trid waves.

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace topk_route_finish {

constexpr uint32_t tile_width = 32;
constexpr uint32_t half_rows = 16;                    // rows per work unit (one face-pair)
constexpr uint32_t rows_per_risc = 8;                 // reader rows [0,8), writer rows [8,16)
constexpr uint32_t stick_seg_bytes = tile_width * 4;  // 128 B: 32 u32 indices
constexpr uint32_t bounce_slot_bytes = 64;            // Blackhole DRAM-read alignment
constexpr uint32_t gather_wave = 32;                  // reads in flight per trid wave
constexpr uint32_t wave_trid0 = 1;                    // waves use trids {1, 2}; 0 stays untagged

// With four units per tile a unit is one face, so col0 is 16 for the odd units.
struct UnitPos {
    uint32_t row_tile;
    uint32_t kt;
    uint32_t half;
    uint32_t col0;
    uint32_t ncols;
};

inline UnitPos decode_unit(uint32_t u, uint32_t k_tiles, uint32_t units_per_tile) {
    const uint32_t rem = u % (k_tiles * units_per_tile);
    const uint32_t sub = rem % units_per_tile;
    const bool face_units = units_per_tile == 4;
    return {
        u / (k_tiles * units_per_tile),
        rem / units_per_tile,
        face_units ? sub >> 1 : sub,
        face_units ? (sub & 1) * 16 : 0,
        face_units ? 16 : tile_width};
}

// Zero rows [lr0, lr1) of faces [face0, face1) of one staged half; elem_bytes 4 (u32 indices) doubles every stride.
template <uint32_t elem_bytes>
inline void zero_half_rows(uint32_t base, uint32_t lr0, uint32_t lr1, uint32_t face0, uint32_t face1) {
    constexpr uint32_t row_bytes = 16 * elem_bytes;
    constexpr uint32_t face_bytes = 16 * row_bytes;
    for (uint32_t face = face0; face < face1; ++face) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + face * face_bytes + lr0 * row_bytes);
        for (uint32_t w = 0; w < (lr1 - lr0) * row_bytes / 4; ++w) {
            p[w] = 0;
        }
    }
}

// Needs the stick reads barriered (stick_l1 row j is unit row lr_begin + j); returns with both wave trids drained.
template <bool index_is_u32, typename SrcAccessor>
inline void gather_unit_rows(
    const Noc& noc,
    const SrcAccessor& src,
    const CoreLocalMem<uint32_t>& bounce_dst,
    uint32_t bounce_base,
    volatile tt_l1_ptr uint32_t* stick_l1,
    uint32_t val_base,
    uint32_t idx_out_base,
    uint32_t row_tile,
    uint32_t width_tiles,
    uint32_t half,
    uint32_t lr_begin,
    uint32_t nrows,
    uint32_t col_begin,
    uint32_t col_end) {
    // In-flight bookkeeping, one set per wave parity (RISC-private; never a NoC target).
    uint16_t pend_off16[2][gather_wave];  // output staging offset (bf16/u16 flavor)
    uint8_t pend_sub[2][gather_wave];     // element offset within the 64 B bounce slot
    uint32_t pend_idxv[2][gather_wave];   // the gathered source index itself

    auto extract = [&](uint32_t p, uint32_t count) {
        const uint32_t slots = bounce_base + p * gather_wave * bounce_slot_bytes;
        for (uint32_t s = 0; s < count; ++s) {
            const uint16_t v =
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slots + s * bounce_slot_bytes + pend_sub[p][s]);
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(val_base + pend_off16[p][s]) = v;
            if constexpr (index_is_u32) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_out_base + (pend_off16[p][s] << 1)) =
                    pend_idxv[p][s];
            } else {
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_out_base + pend_off16[p][s]) =
                    static_cast<uint16_t>(pend_idxv[p][s]);
            }
        }
    };

    uint32_t parity = 0;
    uint32_t cnt = 0;
    bool other_in_flight = false;
    noc_async_read_set_trid(wave_trid0, noc.get_noc_id());
    for (uint32_t j = 0; j < nrows; ++j) {
        const uint32_t lr = lr_begin + j;
        for (uint32_t c = col_begin; c < col_end; ++c) {
            const uint32_t index_value = stick_l1[j * tile_width + c];
            // Source element (half * 16 + lr, index_value & 31); face math in reader_topk_route_finish_gather.cpp.
            const uint32_t src_page = row_tile * width_tiles + (index_value >> 5);
            const uint32_t byte =
                (half << 10) | (((index_value >> 4) & 1) << 9) | (lr << 5) | ((index_value & 15) << 1);
            noc.async_read(
                src,
                bounce_dst,
                bounce_slot_bytes,
                {.page_id = src_page, .offset_bytes = byte & ~(bounce_slot_bytes - 1)},
                {.offset_bytes = (parity * gather_wave + cnt) * bounce_slot_bytes});
            pend_sub[parity][cnt] = byte & (bounce_slot_bytes - 1);
            pend_off16[parity][cnt] = (c >> 4) << 9 | lr << 5 | (c & 15) << 1;
            pend_idxv[parity][cnt] = index_value;

            if (++cnt == gather_wave) {
                // This wave is in flight; retire the OTHER wave before reusing its slots.
                if (other_in_flight) {
                    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wave_trid0 + (parity ^ 1)});
                    extract(parity ^ 1, gather_wave);
                }
                other_in_flight = true;
                parity ^= 1;
                cnt = 0;
                noc_async_read_set_trid(wave_trid0 + parity, noc.get_noc_id());
            }
        }
    }
    // Drain. The full other-parity wave (if any) was issued before the current partial one.
    if (other_in_flight) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wave_trid0 + (parity ^ 1)});
        extract(parity ^ 1, gather_wave);
    }
    if (cnt > 0) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = wave_trid0 + parity});
        extract(parity, cnt);
    }
    // The trid tag is sticky across kernel exit (reset at boot, not per launch), so restore 0.
    noc_async_read_set_trid(0, noc.get_noc_id());
}

}  // namespace topk_route_finish
