// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);  // collapsed dims > 5

    constexpr auto cb_id_src = tt::CBIndex::c_0;

#if SRC_SHARDED
    cb_reserve_back(cb_id_src, src_num_tiles);
    cb_push_back(cb_id_src, src_num_tiles);
#else
    constexpr uint32_t onetile = 1;
    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    constexpr bool has_sharding = get_compile_time_arg_val(src_args.next_compile_time_args_offset()) == 1;
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const auto src = TensorAccessor(src_args, src_addr, src_tile_bytes);
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

    // Use simple linear tile reading for treating sharded as interleaved
    for (uint32_t i = 0; i < dst_num_tiles; ++i) {
        cb_reserve_back(cb_id_src, onetile);
        uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
        noc_async_read_page(start_tile_id + i, src, l1_write_addr_src);
        noc_async_read_barrier();
        cb_push_back(cb_id_src, onetile);
    }
#endif
}
