// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

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
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
#if !SRC_SHARDED
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const auto src = TensorAccessor(src_args, src_addr, src_tile_bytes);
#endif
#if !SRC_SHARDED_B
    const uint32_t src_tile_bytes_b = get_tile_size(cb_id_src_b);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, src_tile_bytes_b);
#endif
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

    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b;

    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
#if SRC_BCAST_COL
                        cb_reserve_back(cb_id_src, onetile);
#if !SRC_SHARDED
                        uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
                        noc_async_read_tile(tile_offset + th, src, l1_write_addr_src);
                        noc_async_read_barrier();
#endif
                        FILL_TILE_WITH_FIRST_COLUMN(cb_id_src);
                        cb_push_back(cb_id_src, onetile);
#else
                        cb_reserve_back(cb_id_src_b, onetile);
#if !SRC_SHARDED_B
                        uint32_t l1_write_addr_src_b = get_write_ptr(cb_id_src_b);
                        noc_async_read_tile(tile_offset_b + th, src_b, l1_write_addr_src_b);
                        noc_async_read_barrier();
#endif
                        FILL_TILE_WITH_FIRST_COLUMN_B(cb_id_src_b);
                        cb_push_back(cb_id_src_b, onetile);
#endif
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if SRC_BCAST_ROW_B
                            // read a tile from src_b
                            cb_reserve_back(cb_id_src_b, onetile);
#if !SRC_SHARDED_B
                            uint32_t l1_write_addr_b = get_write_ptr(cb_id_src_b);
                            noc_async_read_tile(tile_offset_b + tw, src_b, l1_write_addr_b);
                            noc_async_read_barrier();
#endif
                            FILL_TILE_WITH_FIRST_ROW_B(cb_id_src_b);
#if !SRC_SHARDED_B
                            cb_push_back(cb_id_src_b, onetile);
#endif
#else
                            // read a tile from src
                            cb_reserve_back(cb_id_src, onetile);
#if !SRC_SHARDED_B
                            uint32_t l1_write_addr = get_write_ptr(cb_id_src);
                            noc_async_read_tile(tile_offset + tw, src, l1_write_addr);
                            noc_async_read_barrier();
#endif
                            FILL_TILE_WITH_FIRST_ROW(cb_id_src);
#if !SRC_SHARDED_B
                            cb_push_back(cb_id_src, onetile);
#endif
#endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
                    }
#if !SRC_SHARDED
                    // same as following logically
                    // tile_offset += HtWt;
                    // tile_offset += next_c_shift;
                    tile_offset += c_stride;
#endif
#if !SRC_SHARDED_B
                    tile_offset_b += c_stride_b;
#endif
                }
#if !SRC_SHARDED
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_B
                tile_offset_b += next_n_shift_b;
#endif
            }
#if !SRC_SHARDED
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_B
            tile_offset_b += next_d_shift_b;
#endif
        }
#if !SRC_SHARDED
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_B
        tile_offset_b += next_nd_shift_b;
#endif
    }
}
