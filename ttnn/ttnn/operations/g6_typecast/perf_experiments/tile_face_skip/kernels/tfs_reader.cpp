// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tile_face_skip candidate reader (TILE, interleaved). Same block schedule as the op's TILE reader
// (block_tiles per reserve/push, one read barrier per block, read_inflight = 1), but each tile moves only
// its live face runs (tfs_faces.hpp) with noc_async_read(get_noc_addr(page, offset)) instead of one
// noc_async_read_page. No helper bypassed: the op's reader is raw TensorAccessor code as well.
// bf8b output: the partial-column zeroing of the straddling face (col >= w_tail % 16) stays; the whole-face
// zeroing of faces 1/3 is dropped when those faces are dead (a dead face only feeds dead output values, and
// bfp8 shares exponents per 16-value face row, so a dead face never shares an exponent with a live value).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tfs_faces.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mask_pad_columns = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t w_tail = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t h_tail = get_compile_time_arg_val(7);
    constexpr uint32_t face_skip = get_compile_time_arg_val(8);
    constexpr uint32_t hdr = get_compile_time_arg_val(9);
    constexpr uint32_t face = get_compile_time_arg_val(10);
    constexpr auto in_args = TensorAccessorArgs<11>();

    constexpr bool skip_r = face_skip && w_tail >= 1 && w_tail <= 16;
    constexpr bool skip_b = face_skip && h_tail >= 1 && h_tail <= 16;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);
    const auto acc = TensorAccessor(in_args, in_addr, page_bytes);

    // bf16 input only (mask path): element = 2 B, face = 16x16.
    constexpr uint32_t col0 = w_tail % 16;
    constexpr uint32_t face_a = (w_tail != 0 && w_tail < 16) ? 0 : 1;

    const uint32_t end_tile = start_tile + core_tiles;
    tfs::TilePos tp = tfs::tile_pos<Wt, Ht>(start_tile);
    for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_tiles) {
        const uint32_t extent = (end_tile - block_start < block_tiles) ? (end_tile - block_start) : block_tiles;
        cb_reserve_back(cb_in, extent);
        const uint32_t base = get_write_ptr(cb_in);
        uint32_t l1 = base;
        tfs::TilePos tpb = tp;
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t page = block_start + t;
            tfs::for_live_segments<page_bytes, hdr, face, skip_r, skip_b, Wt, Ht>(
                tp,
                [&](uint32_t off, uint32_t bytes) { noc_async_read(acc.get_noc_addr(page, off), l1 + off, bytes); });
            tfs::advance<Wt, Ht>(tp);
            l1 += page_bytes;
        }
        noc_async_read_barrier();
        if constexpr (mask_pad_columns) {
            for (uint32_t t = 0; t < extent; ++t) {
                if (tpb.col == Wt - 1) {
                    const uint32_t tile = base + t * page_bytes;
                    if constexpr (!skip_r && w_tail != 0 && w_tail < 16) {
                        // faces 1/3 read in full: zero them (the op's behaviour without face skip)
                        for (uint32_t f : {1u, 3u}) {
                            volatile tt_l1_ptr uint32_t* p =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + f * face);
                            for (uint32_t i = 0; i < face / 4; ++i) {
                                p[i] = 0;
                            }
                        }
                    }
                    if constexpr (col0 != 0) {
                        for (uint32_t f : {face_a, face_a + 2}) {
                            volatile tt_l1_ptr uint16_t* p =
                                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + f * face);
                            for (uint32_t r = 0; r < 16; ++r) {
                                for (uint32_t c = col0; c < 16; ++c) {
                                    p[r * 16 + c] = 0;
                                }
                            }
                        }
                    }
                }
                tfs::advance<Wt, Ht>(tpb);
            }
        }
        cb_push_back(cb_in, extent);
    }
}
