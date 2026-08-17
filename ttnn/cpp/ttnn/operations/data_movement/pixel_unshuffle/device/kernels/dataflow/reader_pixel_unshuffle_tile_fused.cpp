// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused pixel_unshuffle TILE reader (Approach B, width-chunked).
// Work item = one (output tile-row, width-chunk): for chunk `wc` of output tiles
// [wc*WoC .. wc*WoC+WoC), load the input sub-band it gathers from — band_rows
// (=r+1) input tile-rows × WiC (=WoC*r+1) input tile-cols starting at
// (tr0, cs*r) — into c_0. This bounds L1 to the chunk width. The compute kernel
// untilizes it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t r = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t C_out = get_compile_time_arg_val(2);
    constexpr uint32_t HpTiles = get_compile_time_arg_val(3);
    constexpr uint32_t WpTiles = get_compile_time_arg_val(4);
    constexpr uint32_t HoTiles = get_compile_time_arg_val(5);
    constexpr uint32_t WoC = get_compile_time_arg_val(6);          // output chunk width in tiles
    constexpr uint32_t WiC = get_compile_time_arg_val(7);          // input sub-band width in tiles (WoC*r+1)
    constexpr uint32_t num_wchunks = get_compile_time_arg_val(8);  // ceil(WoTiles / WoC)
    constexpr uint32_t channel_order = get_compile_time_arg_val(9);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cb_in = get_compile_time_arg_val(11);
    constexpr uint32_t band_rows = get_compile_time_arg_val(12);
    constexpr auto src_args = TensorAccessorArgs<13>();

    constexpr uint32_t CHANNEL_MAJOR = 0;
    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t r2 = r * r;
    const uint32_t subband_tiles = band_rows * WiC;
    const uint32_t in_plane_tiles = HpTiles * WpTiles;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_item = get_arg_val<uint32_t>(1);
    uint32_t num_items = get_arg_val<uint32_t>(2);

    const auto s_in = TensorAccessor(src_args, src_addr);
    Noc noc;
    experimental::CB cb0(cb_in);

    for (uint32_t k = 0; k < num_items; k++) {
        uint32_t idx = start_item + k;
        uint32_t wc = idx % num_wchunks;
        uint32_t t = idx / num_wchunks;
        uint32_t hot = t % HoTiles;
        uint32_t t2 = t / HoTiles;  // n*C_out + c_out
        uint32_t c_out = t2 % C_out;
        uint32_t n = t2 / C_out;

        uint32_t c_in, rh;
        if constexpr (channel_order == SPATIAL_MAJOR) {
            c_in = c_out % C;
            rh = (c_out / C) / r;
        } else {
            c_in = c_out / r2;
            rh = (c_out % r2) / r;
        }

        const uint32_t tr0 = (hot * 32 * r + rh) / 32;
        const uint32_t tc0 = (wc * WoC) * r;  // input tile-col start of the sub-band
        const uint32_t in_plane = n * C + c_in;

        cb0.reserve_back(subband_tiles);
        for (uint32_t br = 0; br < band_rows; br++) {
            uint32_t tr = tr0 + br;
            if (tr >= HpTiles) {
                tr = HpTiles - 1;
            }
            uint32_t row_page0 = in_plane * in_plane_tiles + tr * WpTiles;
            uint32_t slot0 = br * WiC;
            for (uint32_t bc = 0; bc < WiC; bc++) {
                uint32_t tc = tc0 + bc;
                if (tc >= WpTiles) {
                    tc = WpTiles - 1;
                }
                noc.async_read(
                    s_in, cb0, tile_bytes, {.page_id = row_page0 + tc}, {.offset_bytes = (slot0 + bc) * tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb0.push_back(subband_tiles);
    }
}
