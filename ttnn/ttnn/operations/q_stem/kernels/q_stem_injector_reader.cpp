// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem x injector, DRAM half (row 0, RISCV_0 / NoC1): the read side of `x_round_multicast_block`.
// For every stage-1 block in send order and every K-block of it, read the round (m row tiles x s1_kb K tiles) of the
// latent into the next cb_x_round slot. Runs up to x_round_depth rounds ahead of the multicast half (RISCV_1 / NoC0,
// q_stem_injector.cpp), so DRAM latency hides behind the receivers' handshakes.
//
// Rows are quarter-aligned: row tile r of quarter d holds latent rows g0 = d * Sq + 32 r .. g0 + 31 of the image.
//   tile_aligned (Sq % 32 == 0): g0 % 32 == 0 -> one latent tile per (row tile, K tile), read straight into the slot.
//   m_non_aligned (regime R9):  the rows straddle latent tiles Ta = g0 / 32 and Tb = Ta + 1 at shift delta = g0 % 32
//                               -> x_realign_block: read Ta and Tb into cb_x_stage, then assemble the slot tile from
//                               face-row runs of the two (out row i <- row i + delta of [Ta ; Tb]), as local NoC
//                               copies (L1 -> L1, 16 B aligned; the DRAM read alignment forbids shifting in flight).
//                               Exact copies, so stage 1 computes bit-identical q rows. A Tb past the image's last
//                               tile (only padding rows of the last quarter need it) is replaced by Ta: the padding
//                               rows of the output stay finite and are never part of the logical result.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/q_stem/kernels/q_stem_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_round = get_compile_time_arg_val(0);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t Q_t = get_compile_time_arg_val(2);
    constexpr uint32_t s1_kb = get_compile_time_arg_val(3);
    constexpr uint32_t nkb = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t qrt = get_compile_time_arg_val(6);
    constexpr uint32_t m = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr uint32_t Sq = get_compile_time_arg_val(10);    // rows per quarter
    constexpr uint32_t x_St = get_compile_time_arg_val(11);  // latent row tiles per image
    constexpr uint32_t cb_x_stage = get_compile_time_arg_val(12);
    constexpr uint32_t inj_idx = get_compile_time_arg_val(13);  // this injector owns global rounds r % NI == inj_idx
    constexpr uint32_t num_injectors = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr uint32_t round_tiles = m * s1_kb;
    // 32 x 32 tile of four 16 x 16 faces (f0 f1 / f2 f3), each face row-major: face row = 16 elements
    constexpr uint32_t face_bytes = x_tile_bytes / 4;
    constexpr uint32_t face_row_bytes = face_bytes / 16;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    uint32_t rorder[G];
    for (uint32_t i = 0; i + 1 < G; ++i) {
        rorder[i] = get_arg_val<uint32_t>(2 + i);
    }
    const auto x = TensorAccessor(x_args, x_addr, x_tile_bytes);

    // Byte offset of (row, column half) inside a tile.
    auto row_off = [](uint32_t row, uint32_t half) {
        return ((row >> 4) * 2 + half) * face_bytes + (row & 15) * face_row_bytes;
    };
    // cb_x_stage is this RISC's private scratch (2 * s1_kb pages, only allocated for m_non_aligned): reserved once,
    // never pushed. Layout per realigned row tile: [Ta tiles k0 ..][Tb tiles k0 ..].
    uint32_t stage = 0;
    if constexpr (Sq % 32 != 0) {
        cb_reserve_back(cb_x_stage, 2 * s1_kb);
        stage = get_write_ptr(cb_x_stage);
    }
    // x_realign_block: s1_kb output tiles at `dst` (K tiles k0 .. k0 + s1_kb - 1) from latent tile rows ta / tb.
    auto realign = [&](uint32_t dst, uint32_t ta, uint32_t tb, uint32_t delta, uint32_t k0) {
        for (uint32_t k = 0; k < s1_kb; ++k) {
#ifndef ABLATE_X_READ
            noc_async_read(x.get_noc_addr(ta * Q_t + k0 + k), stage + k * x_tile_bytes, x_tile_bytes);
            noc_async_read(x.get_noc_addr(tb * Q_t + k0 + k), stage + (s1_kb + k) * x_tile_bytes, x_tile_bytes);
#endif
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < s1_kb; ++k) {
            const uint32_t out_t = dst + k * x_tile_bytes;
            uint32_t i = 0;
            while (i < 32) {  // runs of output rows that stay inside one output face and one source face
                const uint32_t j = i + delta;
                const uint32_t src_t = stage + ((j < 32 ? 0 : s1_kb) + k) * x_tile_bytes;
                const uint32_t jj = j & 31;
                const uint32_t ra = 16 - (i & 15), rb = 16 - (jj & 15);
                const uint32_t run = ra < rb ? ra : rb;
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_read(
                        get_noc_addr(src_t + row_off(jj, half)), out_t + row_off(i, half), run * face_row_bytes);
                }
                i += run;
            }
        }
        noc_async_read_barrier();  // the stage is reused by the next realigned row tile
    };

    q_stem::for_each_block(B, qrt, m, G, p, rorder, [&](uint32_t j, uint32_t b, uint32_t d, uint32_t r0) {
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            if ((j * nkb + kb) % num_injectors != inj_idx) {
                continue;  // another injector's round
            }
            cb_reserve_back(cb_x_round, round_tiles);
            uint32_t off = get_write_ptr(cb_x_round);
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t g0 = d * Sq + (r0 + i) * 32;  // first latent row of the row tile (in image b)
                const uint32_t ta = b * x_St + g0 / 32;
                const uint32_t delta = g0 % 32;
                if (delta == 0) {
                    const uint32_t page0 = ta * Q_t + kb * s1_kb;
                    for (uint32_t k = 0; k < s1_kb; ++k) {
#ifndef ABLATE_X_READ
                        noc_async_read(x.get_noc_addr(page0 + k), off + k * x_tile_bytes, x_tile_bytes);
#endif
                    }
                } else {
                    const uint32_t tb = g0 / 32 + 1 < x_St ? ta + 1 : ta;
                    realign(off, ta, tb, delta, kb * s1_kb);
                }
                off += s1_kb * x_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_x_round, round_tiles);
        }
    });
}
