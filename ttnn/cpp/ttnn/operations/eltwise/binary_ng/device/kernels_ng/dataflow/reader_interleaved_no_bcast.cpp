// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(4);
    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t n_stride = get_arg_val<uint32_t>(6);
    const uint32_t c_stride = get_arg_val<uint32_t>(7);
    const uint32_t N = get_arg_val<uint32_t>(8);
    const uint32_t C = get_arg_val<uint32_t>(9);
    const uint32_t Ht = get_arg_val<uint32_t>(10);
    const uint32_t Wt = get_arg_val<uint32_t>(11);
    const uint32_t cND = get_arg_val<uint32_t>(12);  // collapsed dims > 4
    const uint32_t src_addr_b = get_arg_val<uint32_t>(13);
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(14);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(15);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t src_num_tiles_b = get_arg_val<uint32_t>(17);

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;

#if SRC_SHARDED
    cb_reserve_back(cb_id_src, src_num_tiles);
    cb_push_back(cb_id_src, src_num_tiles);
#else
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);
    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};
#endif
#if SRC_SHARDED_B
    cb_reserve_back(cb_id_src_b, src_num_tiles_b);
    cb_push_back(cb_id_src_b, src_num_tiles_b);
#else
    constexpr bool src_is_dram_b = get_compile_time_arg_val(2) == 1;
    const uint32_t src_tile_bytes_b = get_tile_size(cb_id_src_b);
    const DataFormat src_data_format_b = get_dataformat(cb_id_src_b);

    const InterleavedAddrGenFast<src_is_dram_b> src_b = {
        .bank_base_address = src_addr_b, .page_size = src_tile_bytes_b, .data_format = src_data_format_b};
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B
    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(1) == 1;
    const uint32_t HtWt = Ht * Wt;

    const uint32_t tiles_per_depth = N * C * HtWt;
    uint32_t start_d = start_tile_id / tiles_per_depth;  // ND index
    uint32_t start_remaining_1 = start_tile_id % tiles_per_depth;
    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_remaining_1 / tiles_per_batch;  // N index
    uint32_t start_remaining_2 = start_remaining_1 % tiles_per_batch;
    uint32_t tiles_per_channel = HtWt;
    uint32_t start_c = start_remaining_2 / tiles_per_channel;  // C index
    uint32_t start_t = start_remaining_2 % tiles_per_channel;  // tile index within HtWt
    uint32_t start_th = start_t / Wt;                          // H index
    uint32_t start_tw = start_t % Wt;                          // W index
    uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_d * nD_stride + start_n * n_stride + start_c * c_stride;
    tile_offset += start_th * Wt;
    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;
    uint32_t next_depth_shift = nD_stride - (n_stride * N);

    uint32_t tile_offset_b = start_d * nD_stride_b + start_n * n_stride_b + start_c * c_stride_b;
    tile_offset_b += start_th * Wt;
    uint32_t next_channel_shift_b = c_stride_b - HtWt;
    uint32_t next_batch_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_depth_shift_b = nD_stride_b - (n_stride_b * N);

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_d; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_n = 0) {
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
                        // read a tile from src_b
                        cb_reserve_back(cb_id_src_b, onetile);
                        uint32_t l1_write_addr_b = get_write_ptr(cb_id_src_b);
                        noc_async_read_tile(tile_offset_b + tw, src_b, l1_write_addr_b);
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B
                        noc_async_read_barrier();
#endif
#if !SRC_SHARDED
                        cb_push_back(cb_id_src, onetile);
#endif
#if !SRC_SHARDED_B
                        cb_push_back(cb_id_src_b, onetile);
#endif
                    }
                    if constexpr (!has_sharding) {
                        // next row of tiles should start at the first column
                        start_tw = 0;
                    }
                    tile_offset += Wt;
                    tile_offset_b += Wt;
                }
                tile_offset += next_channel_shift;
                tile_offset_b += next_channel_shift_b;
            }
            tile_offset += next_batch_shift;
            tile_offset_b += next_batch_shift_b;
        }
        tile_offset += next_depth_shift;
        tile_offset_b += next_depth_shift_b;
    }
#endif
}
