// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused pixel_unshuffle TILE gather+writer (Approach B, BRISC, width-chunked).
//
// Work item = one (output tile-row, width-chunk) (n, c_out, HoT, wc):
//  1. GATHER: waits the untilized RM sub-band in c_rm (contiguous (band_rows*32)
//     × (WiC*32) row-major) and builds one output chunk of RM (32 × WoC*32) in
//     c_gathered by the stride-r stick pick. For output-chunk col cc, the global
//     output col is cs*32+cc (cs=wc*WoC), and its source col inside the sub-band
//     is cc*r+rw. Padding (ho>=Ho / global wo>=Wo) is 0.
//  2. WRITE: waits the tilized output chunk in c_out and writes its valid tiles
//     (cs+wt < WoTiles) to DRAM.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <typename T>
inline void gather_chunk(
    uint32_t rm_base,
    uint32_t g_base,
    uint32_t hot,
    uint32_t cs,  // output tile-col start of chunk (in tiles) * 32 done by caller? no: tiles
    uint32_t r,
    uint32_t rh,
    uint32_t rw,
    uint32_t tr0,
    uint32_t WiCfull,  // WiC*32 (sub-band RM row width)
    uint32_t WoCfull,  // WoC*32 (output chunk RM row width)
    uint32_t Ho,
    uint32_t Wo) {
    volatile tt_l1_ptr T* rm = reinterpret_cast<volatile tt_l1_ptr T*>(rm_base);
    volatile tt_l1_ptr T* g = reinterpret_cast<volatile tt_l1_ptr T*>(g_base);
    const uint32_t band_row0 = tr0 * 32;
    const uint32_t wo0 = cs * 32;  // global output col of this chunk's first column
    for (uint32_t oi = 0; oi < 32; oi++) {
        uint32_t ho = hot * 32 + oi;
        uint32_t out_off = oi * WoCfull;
        if (ho < Ho) {
            uint32_t ir = ho * r + rh;
            uint32_t src_off = (ir - band_row0) * WiCfull;  // row `ir` within untilized sub-band
            uint32_t subcol = rw;                           // sub-band col for cc=0 (= cc*r+rw)
            uint32_t cc = 0;
            for (; cc < WoCfull; cc++) {
                uint32_t gwo = wo0 + cc;
                if (gwo < Wo) {
                    g[out_off + cc] = rm[src_off + subcol];
                } else {
                    g[out_off + cc] = 0;
                }
                subcol += r;
            }
        } else {
            for (uint32_t cc = 0; cc < WoCfull; cc++) {
                g[out_off + cc] = 0;
            }
        }
    }
}

void kernel_main() {
    constexpr uint32_t r = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t C_out = get_compile_time_arg_val(2);
    constexpr uint32_t Ho = get_compile_time_arg_val(3);
    constexpr uint32_t Wo = get_compile_time_arg_val(4);
    constexpr uint32_t HoTiles = get_compile_time_arg_val(5);
    constexpr uint32_t WoTiles = get_compile_time_arg_val(6);
    constexpr uint32_t WoC = get_compile_time_arg_val(7);
    constexpr uint32_t WiC = get_compile_time_arg_val(8);
    constexpr uint32_t num_wchunks = get_compile_time_arg_val(9);
    constexpr uint32_t channel_order = get_compile_time_arg_val(10);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(12);
    constexpr uint32_t cb_gathered = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out = get_compile_time_arg_val(14);
    constexpr uint32_t band_rows = get_compile_time_arg_val(15);
    constexpr uint32_t datum_bytes = get_compile_time_arg_val(16);
    constexpr auto dst_args = TensorAccessorArgs<17>();

    constexpr uint32_t CHANNEL_MAJOR = 0;
    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t r2 = r * r;
    const uint32_t subband_tiles = band_rows * WiC;
    const uint32_t WiCfull = WiC * 32;
    const uint32_t WoCfull = WoC * 32;
    const uint32_t out_plane_tiles = HoTiles * WoTiles;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_item = get_arg_val<uint32_t>(1);
    uint32_t num_items = get_arg_val<uint32_t>(2);

    const auto s_out = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    experimental::CB cb_rm_buf(cb_rm);
    experimental::CB cb_g(cb_gathered);
    experimental::CB cb_o(cb_out);

    for (uint32_t k = 0; k < num_items; k++) {
        uint32_t idx = start_item + k;
        uint32_t wc = idx % num_wchunks;
        uint32_t t = idx / num_wchunks;
        uint32_t hot = t % HoTiles;
        uint32_t t2 = t / HoTiles;
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
        const uint32_t cs = wc * WoC;  // output tile-col start of this chunk

        // 1. gather untilized sub-band (c_rm) -> output chunk RM (c_gathered)
        cb_rm_buf.wait_front(subband_tiles);
        cb_g.reserve_back(WoC);
        uint32_t rm_base = cb_rm_buf.get_read_ptr();
        uint32_t g_base = cb_g.get_write_ptr();
        if constexpr (datum_bytes == 2) {
            gather_chunk<uint16_t>(rm_base, g_base, hot, cs, r, rh, rw, tr0, WiCfull, WoCfull, Ho, Wo);
        } else {
            gather_chunk<uint32_t>(rm_base, g_base, hot, cs, r, rh, rw, tr0, WiCfull, WoCfull, Ho, Wo);
        }
        cb_g.push_back(WoC);
        cb_rm_buf.pop_front(subband_tiles);

        // 2. write tilized output chunk (c_out) -> DRAM (only tiles cs+wt < WoTiles)
        cb_o.wait_front(WoC);
        uint32_t out_tile0 = (n * C_out + c_out) * out_plane_tiles + hot * WoTiles + cs;
        for (uint32_t wt = 0; wt < WoC; wt++) {
            if (cs + wt < WoTiles) {
                noc.async_write(
                    cb_o, s_out, tile_bytes, {.offset_bytes = wt * tile_bytes}, {.page_id = out_tile0 + wt});
            }
        }
        noc.async_write_barrier();
        cb_o.pop_front(WoC);
    }
}
