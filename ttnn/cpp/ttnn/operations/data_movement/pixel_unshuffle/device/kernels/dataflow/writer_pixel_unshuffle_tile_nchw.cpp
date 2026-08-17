// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Native TILE writer for pixel_unshuffle on NCHW TILE output.
//
// Pops the (r+1)^2-tile input box the reader placed in CB0, builds one 32x32
// output TILE datum-by-datum (gather with spatial stride r + channel remap),
// and writes it. Padding datums (ho>=Ho or wo>=Wo) are 0.
//
// Tile datum layout: 32x32 tile = 4 faces of 16x16 (row-major within a face);
// face index = (row/16)*2 + (col/16).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

inline uint32_t tile_off(uint32_t row, uint32_t col) {
    uint32_t face = ((row >> 4) << 1) + (col >> 4);
    return (face << 8) + ((row & 15) << 4) + (col & 15);
}

template <typename T>
inline void gather_tile(
    uint32_t in_base,
    uint32_t out_base,
    uint32_t hot,
    uint32_t wot,
    uint32_t r,
    uint32_t rh,
    uint32_t rw,
    uint32_t tr0,
    uint32_t tc0,
    uint32_t box,
    uint32_t Ho,
    uint32_t Wo) {
    volatile tt_l1_ptr T* in = reinterpret_cast<volatile tt_l1_ptr T*>(in_base);
    volatile tt_l1_ptr T* out = reinterpret_cast<volatile tt_l1_ptr T*>(out_base);
    for (uint32_t oi = 0; oi < 32; oi++) {
        uint32_t ho = hot * 32 + oi;
        bool row_ok = ho < Ho;
        uint32_t ir = ho * r + rh;
        uint32_t btr = (ir >> 5) - tr0;
        uint32_t band = btr * box;
        uint32_t ir_loc = ir & 31;
        for (uint32_t oj = 0; oj < 32; oj++) {
            T v = 0;
            uint32_t wo = wot * 32 + oj;
            if (row_ok && wo < Wo) {
                uint32_t ic = wo * r + rw;
                uint32_t slot = band + ((ic >> 5) - tc0);
                uint32_t src = slot * 1024 + tile_off(ir_loc, ic & 31);
                v = in[src];
            }
            out[tile_off(oi, oj)] = v;
        }
    }
}

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
    constexpr uint32_t box = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(13);
    constexpr uint32_t datum_bytes = get_compile_time_arg_val(14);
    constexpr auto dst_args = TensorAccessorArgs<15>();

    constexpr uint32_t CHANNEL_MAJOR = 0;
    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t r2 = r * r;
    const uint32_t box2 = box * box;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    const auto s_out = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    experimental::CB cb0(cb_in);
    experimental::CB cbo(cb_out);

    const uint32_t out_plane_tiles = HoTiles * WoTiles;

    for (uint32_t k = 0; k < num_tiles; k++) {
        uint32_t idx = start_idx + k;
        uint32_t wot = idx % WoTiles;
        uint32_t t1 = idx / WoTiles;
        uint32_t hot = t1 % HoTiles;
        uint32_t t2 = t1 / HoTiles;
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

        cb0.wait_front(box2);
        uint32_t in_base = cb0.get_read_ptr();
        uint32_t out_base = cbo.get_write_ptr();

        if constexpr (datum_bytes == 2) {
            gather_tile<uint16_t>(in_base, out_base, hot, wot, r, rh, rw, tr0, tc0, box, Ho, Wo);
        } else {
            gather_tile<uint32_t>(in_base, out_base, hot, wot, r, rh, rw, tr0, tc0, box, Ho, Wo);
        }

        uint32_t out_page = (n * C_out + c_out) * out_plane_tiles + hot * WoTiles + wot;
        noc.async_write(
            use<experimental::CB::AddrSelector::WRITE_PTR>(cbo), s_out, tile_bytes, {}, {.page_id = out_page});
        noc.async_write_barrier();

        cb0.pop_front(box2);
    }
}
