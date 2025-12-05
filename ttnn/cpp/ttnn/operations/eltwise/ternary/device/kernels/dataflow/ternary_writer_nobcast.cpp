// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    // Additional arguments for width sharding support
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(3);
    const uint32_t D = get_arg_val<uint32_t>(4);
    const uint32_t N = get_arg_val<uint32_t>(5);
    const uint32_t C = get_arg_val<uint32_t>(6);
    const uint32_t Ht = get_arg_val<uint32_t>(7);
    const uint32_t Wt = get_arg_val<uint32_t>(8);
    const uint32_t cND = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1, 0>();

#if !DST_SHARDED
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;
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

    uint32_t num_tiles_written = 0;
    uint32_t dst_tile_offset = start_tile_id;

    for (uint32_t nd = start_nd; nd < cND && num_tiles_written < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_written < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_written < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_written < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_written < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < dst_num_tiles;
                             ++tw, ++num_tiles_written) {
                            cb_wait_front(cb_id_out, onetile);
                            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                            noc_async_write_page(dst_tile_offset + num_tiles_written, s, l1_read_addr);
                            noc_async_write_barrier();
                            cb_pop_front(cb_id_out, onetile);
                        }
                        if constexpr (has_sharding) {
                            // adjust the output tile offset since we had to skip parts of the row
                            dst_tile_offset += (Wt - dst_shard_width);
                        } else {
                            // otherwise, next row of tiles should start at the first column
                            start_tw = 0;
                        }
                    }
                }
            }
        }
    }
#endif
}
