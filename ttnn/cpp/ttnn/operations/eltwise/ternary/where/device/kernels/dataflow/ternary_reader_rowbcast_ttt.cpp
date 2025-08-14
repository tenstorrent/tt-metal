// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // Extended to 3-tensor ternary pattern (27 args total)
    const uint32_t src_addr = get_arg_val<uint32_t>(0);  // predicate address
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
    const uint32_t cND = get_arg_val<uint32_t>(14);         // collapsed dims > 5
    const uint32_t src_addr_b = get_arg_val<uint32_t>(15);  // true tensor address
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(19);
    const uint32_t src_num_tiles_b = get_arg_val<uint32_t>(20);
    const uint32_t src_addr_c = get_arg_val<uint32_t>(21);  // false tensor address
    const uint32_t nD_stride_c = get_arg_val<uint32_t>(22);
    const uint32_t d_stride_c = get_arg_val<uint32_t>(23);
    const uint32_t n_stride_c = get_arg_val<uint32_t>(24);
    const uint32_t c_stride_c = get_arg_val<uint32_t>(25);
    const uint32_t src_num_tiles_c = get_arg_val<uint32_t>(26);

    constexpr auto cb_id_src = tt::CBIndex::c_0;    // predicate CB
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;  // true tensor CB
    constexpr auto cb_id_src_c = tt::CBIndex::c_2;  // false tensor CB

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto src_c_args = TensorAccessorArgs<src_b_args.next_compile_time_args_offset()>();
#if SRC_SHARDED
    cb_reserve_back(cb_id_src, src_num_tiles);
    cb_push_back(cb_id_src, src_num_tiles);
#else
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const auto src = TensorAccessor(src_args, src_addr, src_tile_bytes);
#endif
#if SRC_SHARDED_B
    cb_reserve_back(cb_id_src_b, src_num_tiles_b);
    cb_push_back(cb_id_src_b, src_num_tiles_b);
#else
    const uint32_t src_tile_bytes_b = get_tile_size(cb_id_src_b);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, src_tile_bytes_b);
#endif
#if SRC_SHARDED_C
    cb_reserve_back(cb_id_src_c, src_num_tiles_c);
    cb_push_back(cb_id_src_c, src_num_tiles_c);
#else
    const uint32_t src_tile_bytes_c = get_tile_size(cb_id_src_c);
    const auto src_c = TensorAccessor(src_c_args, src_addr_c, src_tile_bytes_c);
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B || !SRC_SHARDED_C
    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(src_c_args.next_compile_time_args_offset()) == 1;
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

    uint32_t tile_offset_c =
        start_nd * nD_stride_c + start_d * d_stride_c + start_n * n_stride_c + start_c * c_stride_c;
#if !SRC_BCAST_C
    tile_offset_c += start_th * Wt;
#endif
    uint32_t next_c_shift_c = c_stride_c - HtWt;
    uint32_t next_n_shift_c = n_stride_c - c_stride_c * C;
    uint32_t next_d_shift_c = d_stride_c - n_stride_c * N;
    uint32_t next_nd_shift_c = nD_stride_c - d_stride_c * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED
                            cb_reserve_back(cb_id_src, onetile);
                            uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
                            noc_async_read_tile(tile_offset + tw, src, l1_write_addr_src);
#endif
#if !SRC_SHARDED_B
                            // read a tile from src_b (true tensor)
                            cb_reserve_back(cb_id_src_b, onetile);
                            uint32_t l1_write_addr_b = get_write_ptr(cb_id_src_b);
                            noc_async_read_tile(tile_offset_b + tw, src_b, l1_write_addr_b);
#endif
#if !SRC_SHARDED_C
                            // read a tile from src_c (false tensor)
                            cb_reserve_back(cb_id_src_c, onetile);
                            uint32_t l1_write_addr_c = get_write_ptr(cb_id_src_c);
                            noc_async_read_tile(tile_offset_c + tw, src_c, l1_write_addr_c);
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B || !SRC_SHARDED_C
                            noc_async_read_barrier();
#endif
#if SRC_BCAST && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW(cb_id_src);
#endif
#if SRC_BCAST_B && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_B(cb_id_src_b);
#endif
#if SRC_BCAST_C && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_C(cb_id_src_c);
#endif
#if !SRC_SHARDED
                            cb_push_back(cb_id_src, onetile);
#endif
#if !SRC_SHARDED_B
                            cb_push_back(cb_id_src_b, onetile);
#endif
#if !SRC_SHARDED_C
                            cb_push_back(cb_id_src_c, onetile);
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
#if !SRC_BCAST_C
                        tile_offset_c += Wt;
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
#if SRC_BCAST_C
                    tile_offset_c += c_stride_c;
#else
                    tile_offset_c += next_c_shift_c;
#endif
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
                tile_offset_c += next_n_shift_c;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
            tile_offset_c += next_d_shift_c;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
        tile_offset_c += next_nd_shift_c;
    }
#endif
}
