// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Native TILE reader for pixel_unshuffle on NCHW TILE input.
//
// Work unit = one OUTPUT tile (n, c_out, HoT, WoT). Reads the (r+1)x(r+1) box of
// input tiles that the output tile gathers from (spatial stride r + channel
// remap) into CB0; the writer pops it and assembles the output tile. Work is
// distributed per output-tile so all cores stay busy. TILE in -> TILE out, no
// untilize/tilize round-trip.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t r = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t Ho = get_compile_time_arg_val(2);
    constexpr uint32_t Wo = get_compile_time_arg_val(3);
    constexpr uint32_t C_out = get_compile_time_arg_val(4);
    constexpr uint32_t HpTiles = get_compile_time_arg_val(5);
    constexpr uint32_t WpTiles = get_compile_time_arg_val(6);
    constexpr uint32_t HoTiles = get_compile_time_arg_val(7);
    constexpr uint32_t WoTiles = get_compile_time_arg_val(8);
    constexpr uint32_t channel_order = get_compile_time_arg_val(9);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cb_in = get_compile_time_arg_val(11);
    constexpr uint32_t box = get_compile_time_arg_val(12);  // r + 1
    constexpr auto src_args = TensorAccessorArgs<13>();

    constexpr uint32_t CHANNEL_MAJOR = 0;
    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t r2 = r * r;
    const uint32_t box2 = box * box;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    const auto s_in = TensorAccessor(src_args, src_addr);
    Noc noc;
    experimental::CB cb0(cb_in);

    const uint32_t in_plane_tiles = HpTiles * WpTiles;

    for (uint32_t k = 0; k < num_tiles; k++) {
        uint32_t idx = start_idx + k;
        uint32_t wot = idx % WoTiles;
        uint32_t t1 = idx / WoTiles;
        uint32_t hot = t1 % HoTiles;
        uint32_t t2 = t1 / HoTiles;  // n*C_out + c_out
        uint32_t c_out = t2 % C_out;
        uint32_t n = t2 / C_out;

        uint32_t c_in, rh, rw;
        if constexpr (channel_order == SPATIAL_MAJOR) {
            c_in = c_out % C;
            rh = (c_out / C) / r;
            rw = (c_out / C) % r;
        } else {
            c_in = c_out / r2;
            rh = (c_out % r2) / r;
            rw = c_out % r;
        }

        const uint32_t tr0 = (hot * 32 * r + rh) / 32;
        const uint32_t tc0 = (wot * 32 * r + rw) / 32;
        const uint32_t in_plane = n * C + c_in;

        cb0.reserve_back(box2);
        for (uint32_t br = 0; br < box; br++) {
            uint32_t tr = tr0 + br;
            if (tr >= HpTiles) {
                tr = HpTiles - 1;
            }
            for (uint32_t bc = 0; bc < box; bc++) {
                uint32_t tc = tc0 + bc;
                if (tc >= WpTiles) {
                    tc = WpTiles - 1;
                }
                uint32_t page = in_plane * in_plane_tiles + tr * WpTiles + tc;
                uint32_t slot = br * box + bc;
                noc.async_read(s_in, cb0, tile_bytes, {.page_id = page}, {.offset_bytes = slot * tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb0.push_back(box2);
    }
}
