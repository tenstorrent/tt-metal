// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

#if COL_GROUP_B
#include <algorithm>

#include "ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/col_group_walk.hpp"
#endif

#if B_ROW_WORDS
// Row 0 of a 32x32 tile: the first rows of faces 0 and 1, held back to back.
constexpr uint32_t kRowBytes = B_ROW_WORDS * 4;
constexpr uint32_t kFaceBytes = 16 * kRowBytes;

FORCE_INLINE void copy_row0(uint32_t dst, uint32_t dst_face, uint32_t src, uint32_t src_face) {
    auto* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    auto* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + dst_face);
    auto* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    auto* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + src_face);
    for (uint32_t i = 0; i < B_ROW_WORDS; ++i) {
        d0[i] = s0[i];
        d1[i] = s1[i];
    }
}
#endif

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(4);
    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);  // collapsed dims > 5
    const uint32_t src_addr_b = get_arg_val<uint32_t>(15);
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(19);
    const uint32_t src_num_tiles_b = get_arg_val<uint32_t>(20);

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;
    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    constexpr auto src_b_args =
        TensorAccessorArgs<src_args.next_compile_time_args_offset(), src_args.next_common_runtime_args_offset()>();

    Noc noc;
    CircularBuffer cb_src(cb_id_src);
    CircularBuffer cb_src_b(cb_id_src_b);

#if SRC_SHARDED
    cb_src.reserve_back(src_num_tiles);
    cb_src.push_back(src_num_tiles);
#else
    const uint32_t src_tile_bytes = cb_src.get_tile_size();
    const auto src = TensorAccessor(src_args, src_addr);
#endif
#if SRC_SHARDED_B
    cb_src_b.reserve_back(src_num_tiles_b);
    cb_src_b.push_back(src_num_tiles_b);
#else
    const uint32_t src_tile_bytes_b = cb_src_b.get_tile_size();
    const auto src_b = TensorAccessor(src_b_args, src_addr_b);
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B
    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(src_b_args.next_compile_time_args_offset()) == 1;
    const uint32_t HtWt = Ht * Wt;

    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    const uint32_t offset_nd = start_tile_id % tiles_per_nd;
    const uint32_t offset_d = offset_nd % tiles_per_d;
    const uint32_t offset_n = offset_d % tiles_per_n;
    const uint32_t offset_c = offset_n % HtWt;
    uint32_t start_nd = start_tile_id / tiles_per_nd;
    uint32_t start_d = offset_nd / tiles_per_d;
    uint32_t start_n = offset_d / tiles_per_n;
    uint32_t start_c = offset_n / HtWt;
    uint32_t start_th = offset_c / Wt;
    uint32_t start_tw = offset_c % Wt;
    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
#if !SRC_BCAST
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b;
#if !SRC_BCAST_B
    tile_offset_b += start_th * Wt;
#endif
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

#if COL_GROUP_B
    // Column groups (0 = page order): each B tile is read once per column and held in c_7.
    const uint32_t group = get_arg_val<uint32_t>(23);
    if (group != 0) {
        CircularBuffer cb_held_b(tt::CBIndex::c_7);
        cb_held_b.reserve_back(COL_GROUP_B);
        const uint32_t held = cb_held_b.get_write_ptr();
        uint32_t held_page[COL_GROUP_B];
        std::fill_n(held_page, COL_GROUP_B, 0xFFFFFFFF);
        uint32_t channel = start_tile_id / HtWt;
        ColGroupWalk walk(start_tile_id % HtWt, Ht, Wt, group);
        uint32_t a_off = 0, b_off = 0;
        const auto set_channel = [&](uint32_t ch) {
            const uint32_t cc = ch % C, nn = ch / C % N, dd = ch / (C * N) % D, ndd = ch / (C * N * D);
            a_off = ndd * nD_stride + dd * d_stride + nn * n_stride + cc * c_stride;
            b_off = ndd * nD_stride_b + dd * d_stride_b + nn * n_stride_b + cc * c_stride_b;
        };
        set_channel(channel);
        for (uint32_t i = 0; i < dst_num_tiles; ++i) {
            const uint32_t tw = walk.tw();
            const uint32_t b_page = b_off + tw;
            const bool fetch = held_page[walk.k] != b_page;
            held_page[walk.k] = b_page;
            cb_src.reserve_back(onetile);
            cb_src_b.reserve_back(onetile);
            noc.async_read(src, cb_src, src_tile_bytes, {.page_id = a_off + walk.th * Wt + tw}, {.offset_bytes = 0});
#if B_ROW_WORDS
            // Restore row 0 from c_7 while A's read is in flight.
            const uint32_t b_tile = cb_src_b.get_write_ptr();
            const uint32_t slot = held + walk.k * 2 * kRowBytes;
            if (fetch) {
                noc.async_read(src_b, cb_src_b, src_tile_bytes_b, {.page_id = b_page}, {.offset_bytes = 0});
            } else {
                copy_row0(b_tile, kFaceBytes, slot, kRowBytes);
            }
            noc.async_read_barrier();
            if (fetch) {
                copy_row0(slot, kRowBytes, b_tile, kFaceBytes);
            }
#if !BCAST_LLK
            FILL_TILE_WITH_FIRST_ROW_B(b_tile);
#endif
#else
            // Block float: hold the whole tile.
            const uint32_t slot = held + walk.k * src_tile_bytes_b;
            if (fetch) {
                noc.async_read(
                    src_b,
                    cb_held_b,
                    src_tile_bytes_b,
                    {.page_id = b_page},
                    {.offset_bytes = walk.k * src_tile_bytes_b});
                noc.async_read_barrier();
#if !BCAST_LLK
                FILL_TILE_WITH_FIRST_ROW_B(slot);
#endif
            }
            noc.async_read(CoreLocalMem<uint32_t>(slot), cb_src_b, src_tile_bytes_b, {}, {});
            noc.async_read_barrier();
#endif
            cb_src.push_back(onetile);
            cb_src_b.push_back(onetile);
            if (walk.next()) {
                set_channel(++channel);
            }
        }
        return;
    }
#endif

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED
                            cb_src.reserve_back(onetile);
                            noc.async_read(
                                src, cb_src, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_B
                            // read a tile from src_b
                            cb_src_b.reserve_back(onetile);
                            noc.async_read(
                                src_b,
                                cb_src_b,
                                src_tile_bytes_b,
                                {.page_id = tile_offset_b + tw},
                                {.offset_bytes = 0});
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B
                            noc.async_read_barrier();
#endif
#if SRC_BCAST && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW(cb_src.get_write_ptr());
#endif
#if SRC_BCAST_B && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_B(cb_src_b.get_write_ptr());
#endif
#if !SRC_SHARDED
                            cb_src.push_back(onetile);
#endif
#if !SRC_SHARDED_B
                            cb_src_b.push_back(onetile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
#if !SRC_BCAST
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_B
                        tile_offset_b += Wt;
#endif
                    }
#if SRC_BCAST
                    // same as following logically
                    // tile_offset += HtWt;
                    // tile_offset += next_c_shift;
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#if SRC_BCAST_B
                    tile_offset_b += c_stride_b;
#else
                    tile_offset_b += next_c_shift_b;
#endif
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
    }
#endif
}
