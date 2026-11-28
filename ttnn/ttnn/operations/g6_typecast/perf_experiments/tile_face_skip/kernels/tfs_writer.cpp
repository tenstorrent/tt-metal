// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tile_face_skip candidate writer (TILE, interleaved). Same schedule as the op's TILE writer (wait a block,
// write it, writes_flushed, pop; one final barrier) but each tile writes only its live face runs
// (tfs_faces.hpp) with noc_async_write(get_noc_addr(page, offset)). Output padding faces are left untouched
// (padding-in-ttnn §1). `write_group` = compute blocks drained per wait/flush/pop (1 = the op's schedule;
// 2 = one flush stall per two blocks, needs an output CB of >= 2 blocks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tfs_faces.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t w_tail = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t h_tail = get_compile_time_arg_val(6);
    constexpr uint32_t face_skip = get_compile_time_arg_val(7);
    constexpr uint32_t hdr = get_compile_time_arg_val(8);
    constexpr uint32_t face = get_compile_time_arg_val(9);
    constexpr uint32_t write_group = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();

    constexpr bool skip_r = face_skip && w_tail >= 1 && w_tail <= 16;
    constexpr bool skip_b = face_skip && h_tail >= 1 && h_tail <= 16;
    constexpr uint32_t group_tiles = write_group * block_tiles;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);
    const auto acc = TensorAccessor(out_args, out_addr, page_bytes);

    const uint32_t end_tile = start_tile + core_tiles;
    tfs::TilePos tp = tfs::tile_pos<Wt, Ht>(start_tile);
    for (uint32_t gs = start_tile; gs < end_tile; gs += group_tiles) {
        const uint32_t extent = (end_tile - gs < group_tiles) ? (end_tile - gs) : group_tiles;
        cb_wait_front(cb_out, extent);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t page = gs + t;
            tfs::for_live_segments<page_bytes, hdr, face, skip_r, skip_b, Wt, Ht>(
                tp,
                [&](uint32_t off, uint32_t bytes) { noc_async_write(l1 + off, acc.get_noc_addr(page, off), bytes); });
            tfs::advance<Wt, Ht>(tp);
            l1 += page_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, extent);
    }
    noc_async_write_barrier();
}
