// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const uint32_t packed_scalar = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(4);
    const uint32_t D = get_arg_val<uint32_t>(5);
    const uint32_t N = get_arg_val<uint32_t>(6);
    const uint32_t C = get_arg_val<uint32_t>(7);
    const uint32_t Ht = get_arg_val<uint32_t>(8);
    const uint32_t Wt = get_arg_val<uint32_t>(9);
    const uint32_t cND = get_arg_val<uint32_t>(10);  // collapsed dims > 5
    const uint32_t HtWt = Ht * Wt;

    constexpr auto cb_id_src = tt::CBIndex::c_1;
    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    constexpr uint32_t onetile = 1;

#if !DST_SHARDED
    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const auto dst = TensorAccessor(dst_args, dst_addr, dst_tile_bytes);
    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;

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
#endif
    // we only need to fill a tile with the scalar value once
    cb_reserve_back(cb_id_src, onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    const auto float_ptr = reinterpret_cast<const float*>(&packed_scalar);
    FILL_WITH_VALUE_FLOAT(cb_id_src, *float_ptr);
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_src, packed_scalar);
#endif
    cb_push_back(cb_id_src, onetile);

#if !DST_SHARDED
    // Use simple linear tile writing for treating sharded as interleaved
    for (uint32_t i = 0; i < dst_num_tiles; ++i) {
        cb_wait_front(cb_id_dst, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
        noc_async_write_page(start_tile_id + i, dst, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_dst, onetile);
    }
#endif
}
