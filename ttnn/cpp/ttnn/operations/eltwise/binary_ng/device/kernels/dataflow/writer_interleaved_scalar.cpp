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
    const uint32_t N = get_arg_val<uint32_t>(5);
    const uint32_t C = get_arg_val<uint32_t>(6);
    const uint32_t Ht = get_arg_val<uint32_t>(7);
    const uint32_t Wt = get_arg_val<uint32_t>(8);
    const uint32_t cND = get_arg_val<uint32_t>(9);  // collapsed dims > 4
    const uint32_t HtWt = Ht * Wt;

    constexpr auto cb_id_src = tt::CBIndex::c_1;
    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    constexpr uint32_t onetile = 1;

#if !DST_SHARDED
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const DataFormat dst_data_format = get_dataformat(cb_id_dst);

    const InterleavedAddrGenFast<dst_is_dram> dst = {
        .bank_base_address = dst_addr, .page_size = dst_tile_bytes, .data_format = dst_data_format};
#endif

    const uint32_t tiles_per_depth = N * C * HtWt;
    uint32_t start_d = start_tile_id / tiles_per_depth;  // ND index
    uint32_t start_remaining_1 = start_tile_id % tiles_per_depth;
    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_remaining_1 / tiles_per_batch;  // N index
    uint32_t start_remaining_2 = start_remaining_1 % tiles_per_batch;
    uint32_t tiles_per_channel = HtWt;
    uint32_t start_c = start_remaining_2 / tiles_per_channel;  // C index
    uint32_t start_t = start_remaining_2 % tiles_per_channel;

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
    uint32_t num_tiles_written = 0;
    for (uint32_t nd = start_d; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_n = 0) {
        for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
            for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_t = 0) {
                for (uint32_t t = start_t; t < HtWt && num_tiles_written < dst_num_tiles; ++t, ++num_tiles_written) {
                    // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                    cb_wait_front(cb_id_dst, onetile);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
                    noc_async_write_tile(start_tile_id + num_tiles_written, dst, l1_read_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_dst, onetile);
                }
            }
        }
    }
#endif
}
