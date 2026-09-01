// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared per-unit gather machinery for topk_route_finish, used by BOTH data-movement RISCs:
// the reader (BRISC) gathers unit rows [0, 8) and the writer (NCRISC) gathers rows [8, 16).
//
// Row-split safety (why two RISCs may fill the same staging page concurrently): a staged
// half is the output tile's face-pair range — two contiguous faces of 16 rows each, one
// row = 16 elements. The staging offset of element (lr, c) is
//   off16 = (c>>4)<<9 | lr<<5 | (c&15)<<1        (bf16 values / u16 indices; u32 doubles it)
// so row lr occupies byte range [lr*32, lr*32+32) WITHIN each 512 B face: lr in [0,8)
// touches only bytes [0,256) of each face, lr in [8,16) only [256,512). The two RISCs
// therefore write disjoint 32 B face rows (disjoint words), both for the gather stores and
// for the split zero-fill — no synchronization is needed on the staging bytes themselves.
//
// Wave pipeline (trid double-buffering): gather reads are issued in 32-deep waves whose
// packets are tagged with alternating NoC transaction ids (wave_trid0 + parity). The tag is
// applied through the sticky NOC_PACKET_TAG register — noc_async_read_set_trid() once per
// wave; every read issued from the read cmd buffer inherits it until it is re-set (the same
// stickiness Noc::set_async_read_state documents). While wave B's reads are in flight, wave
// A is retired with a per-trid barrier (noc_async_read_barrier_with_trid via
// Noc::async_read_barrier<NocOptions::TXN_ID>) followed by extraction — so extraction and
// the next wave's issue overlap the previous wave's flight instead of every wave paying a
// full-drain stall. The 64-slot bounce buffer holds exactly the two in-flight waves.
// The tag register is sticky, so between waves the OTHER reads issued from the read cmd
// buffer (index-stick reads) also carry the current wave trid; that is safe because every
// read -- tagged or not -- bumps noc_reads_num_issued, making the plain read barrier a
// global superset of any per-trid barrier. gather_unit_rows resets the tag to 0 before
// returning so traffic after it is genuinely untagged. Waves use trids 1 and 2, at most
// 32 reads outstanding per trid, far under the 255-per-trid hardware counter.

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

// Zero rows [lr0, lr1) of one staged half. elem_bytes = 2 (bf16 values / u16 indices) or
// 4 (u32 indices); face/row strides scale with it (see off16 above: u32 doubles every term).
template <uint32_t elem_bytes>
inline void zero_half_rows(uint32_t base, uint32_t lr0, uint32_t lr1) {
    constexpr uint32_t row_bytes = 16 * elem_bytes;  // 16 elements per face row
    constexpr uint32_t face_bytes = 16 * row_bytes;  // 16 rows per face
    for (uint32_t face = 0; face < 2; ++face) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + face * face_bytes + lr0 * row_bytes);
        for (uint32_t w = 0; w < (lr1 - lr0) * row_bytes / 4; ++w) {
            p[w] = 0;
        }
    }
}

// Gather unit rows [lr_begin, lr_begin + nrows) into the staged halves.
//
// stick_l1 holds the calling RISC's staged index-stick segments, locally indexed: local row
// j corresponds to global unit row lr = lr_begin + j. The caller must have barriered the
// stick reads before calling. On return every gathered element has been extracted into the
// staging halves and NO tagged reads remain outstanding (both wave trids are drained).
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
    uint32_t valid_cols) {
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
        for (uint32_t c = 0; c < valid_cols; ++c) {
            const uint32_t index_value = stick_l1[j * tile_width + c];
            // Source: wr = half*16 + lr, wc = index_value & 31 (see the reader's face math).
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
                // This wave is full and in flight; before reusing the OTHER wave's slots,
                // retire it.
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
    // The tag register is sticky across kernel exit (firmware resets it at boot, not per
    // launch): restore 0 so later reads -- this kernel's and the next kernel's on this
    // RISC -- are untagged.
    noc_async_read_set_trid(0, noc.get_noc_id());
}

}  // namespace topk_route_finish
