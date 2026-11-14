// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Row patch for bfp8 KV caches. A bfp8 tile is 64 exponent bytes (one per 16-element face row: face f,
// row i at byte f * 16 + i) followed by four 16x16 faces of one byte per element, so tile row r is two
// exponent bytes and two 16-byte mantissa rows, independent of the other 31 rows. Compute packs the input
// heads tile (head h in row h) into this format; this copies head h's row bytes into row r of each of the
// head's Wt cache tiles: one exponent-chunk read-modify-write and two 16-byte mantissa writes per tile,
// instead of reading, untilizing, retilizing and writing back Wt whole tiles per head.
//
// Heads [h_begin, h_end) are patched in groups of up to MAX_GROUP: every exponent chunk of a group is read
// under one barrier, then every write of the group is issued under one barrier. scratch must be 64-byte
// aligned and hold SLOT_BYTES per tile of a group: the exponent chunk, then the page's NoC address.
namespace row_patch_bfp8 {

constexpr uint32_t EXP_BYTES = 64;
constexpr uint32_t FACE_BYTES = 256;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t READ_ALIGN = NOC_DRAM_READ_ALIGNMENT_BYTES;  // 32 on WH, 64 on BH
constexpr uint32_t EXP_READ_BYTES = READ_ALIGN > 2 * FACE_ROWS ? READ_ALIGN : 2 * FACE_ROWS;
constexpr uint32_t MAX_GROUP = 16;
constexpr uint32_t SLOT_BYTES = 128;

template <uint32_t Wt, uint32_t head_offset_t, typename Accessor>
FORCE_INLINE void patch_heads(
    Noc& noc,
    const Accessor& cache,
    uint32_t packed_base,
    uint32_t packed_tile_bytes,
    uint32_t scratch,
    uint32_t cache_id,
    uint32_t r,
    uint32_t h_begin,
    uint32_t h_end) {
    const uint32_t dst_face = r < FACE_ROWS ? 0 : 2;
    const uint32_t dst_row = r % FACE_ROWS;
    const uint32_t exp_half = dst_face * FACE_ROWS;  // 0 or 32: exponents of the two faces holding row r
    const uint32_t read_off = exp_half & ~(READ_ALIGN - 1);
    const uint8_t noc_id = noc.get_noc_id();
    for (uint32_t h0 = h_begin; h0 < h_end; h0 += MAX_GROUP) {
        const uint32_t hn = (h_end - h0) < MAX_GROUP ? (h_end - h0) : MAX_GROUP;
        for (uint32_t h = 0; h < hn; ++h) {
            for (uint32_t t = 0; t < Wt; ++t) {
                const uint32_t slot = scratch + (h * Wt + t) * SLOT_BYTES;
                const uint64_t page = cache.get_noc_addr(cache_id + (h0 + h) * head_offset_t + t, 0, noc_id);
                *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot + EXP_BYTES) = page;
                noc.async_read(
                    PrecomposedUnicastEndpoint{},
                    CoreLocalMem<uint32_t>(slot + read_off),
                    EXP_READ_BYTES,
                    {.noc_addr = page + read_off},
                    {});
            }
        }
        noc.async_read_barrier();
        for (uint32_t h = 0; h < hn; ++h) {
            const uint32_t head = h0 + h;
            const uint32_t src_face = head < FACE_ROWS ? 0 : 2;
            const uint32_t src_row = head % FACE_ROWS;
            for (uint32_t t = 0; t < Wt; ++t) {
                const uint32_t src_tile = packed_base + t * packed_tile_bytes;
                const uint32_t slot = scratch + (h * Wt + t) * SLOT_BYTES;
                const uint64_t page = *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot + EXP_BYTES);
                volatile tt_l1_ptr uint8_t* src_exp =
                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_tile + src_face * FACE_ROWS);
                volatile tt_l1_ptr uint8_t* dst_exp = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(slot + exp_half);
                dst_exp[dst_row] = src_exp[src_row];
                dst_exp[FACE_ROWS + dst_row] = src_exp[FACE_ROWS + src_row];
                noc.async_write(
                    CoreLocalMem<uint32_t>(slot + exp_half),
                    PrecomposedUnicastEndpoint{},
                    2 * FACE_ROWS,
                    {},
                    {.noc_addr = page + exp_half});
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(
                            src_tile + EXP_BYTES + (src_face + f) * FACE_BYTES + src_row * FACE_ROWS),
                        PrecomposedUnicastEndpoint{},
                        FACE_ROWS,
                        {},
                        {.noc_addr = page + EXP_BYTES + (dst_face + f) * FACE_BYTES + dst_row * FACE_ROWS});
                }
            }
        }
        noc.async_write_barrier();
    }
}

FORCE_INLINE uint32_t align_scratch(uint32_t addr) { return (addr + EXP_BYTES - 1) & ~(EXP_BYTES - 1); }

}  // namespace row_patch_bfp8
