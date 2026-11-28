// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#if COL_GROUP_B
#include "ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/col_group_walk.hpp"
#endif

void kernel_main() {
    uint32_t index = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(index++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(index++);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(index++);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(index++);
    const uint32_t D = get_arg_val<uint32_t>(index++);
    const uint32_t N = get_arg_val<uint32_t>(index++);
    const uint32_t C = get_arg_val<uint32_t>(index++);
    const uint32_t Ht = get_arg_val<uint32_t>(index++);
    const uint32_t Wt = get_arg_val<uint32_t>(index++);
    const uint32_t cND = get_arg_val<uint32_t>(index++);  // collapsed dims > 5

    constexpr uint32_t onetile = 1;

    constexpr auto cb_id_dst = tt::CBIndex::c_2;

    Noc noc;
    DataflowBuffer dfb_dst(cb_id_dst);

#if !DST_SHARDED
    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const uint32_t dst_tile_bytes = dfb_dst.get_entry_size();
    const auto dst = TensorAccessor(dst_args, dst_addr);
#endif

#if !DST_SHARDED
    constexpr bool has_sharding = get_compile_time_arg_val(dst_args.next_compile_time_args_offset()) == 1;
    const uint32_t HtWt = Ht * Wt;

#if COL_GROUP_B
    // Same column-group order as the reader; 0 = page order.
    const uint32_t group = get_arg_val<uint32_t>(10);
    if (group != 0) {
        uint32_t channel_base = start_tile_id / HtWt * HtWt;
        ColGroupWalk walk(start_tile_id % HtWt, Ht, Wt, group);
        for (uint32_t i = 0; i < dst_num_tiles; ++i) {
            dfb_dst.wait_front(onetile);
            noc.async_write(dfb_dst, dst, dst_tile_bytes, {}, {.page_id = channel_base + walk.th * Wt + walk.tw()});
            // Full barrier per tile: with many writes in flight, cores far from DRAM fall behind.
            noc.async_write_barrier();
            dfb_dst.pop_front(onetile);
            if (walk.next()) {
                channel_base += HtWt;
            }
        }
        return;
    }
#endif
#if ROTATE_START_BANK
    // Each core owns a contiguous range of tiles. When the range length is a multiple of the DRAM bank count, every
    // core starts on the same bank and all of them hit that bank in lockstep. Start core c on bank c % NUM_DRAM_BANKS
    // instead, and wrap to the head of the range at the end. Reader and writer use the same order.
    uint32_t rotation = 0;
    if (dst_num_tiles > 1) {
        const uint32_t core = start_tile_id / dst_num_tiles;
        rotation = (core % NUM_DRAM_BANKS + NUM_DRAM_BANKS - start_tile_id % NUM_DRAM_BANKS) % NUM_DRAM_BANKS;
        if (rotation >= dst_num_tiles) {
            rotation = 0;
        }
    }
#else
    constexpr uint32_t rotation = 0;
#endif
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const uint32_t range_start = pass == 0 ? start_tile_id + rotation : start_tile_id;
        const uint32_t range_tiles = pass == 0 ? dst_num_tiles - rotation : rotation;
        if (range_tiles == 0) {
            continue;
        }
        const uint32_t tiles_per_n = C * HtWt;
        const uint32_t tiles_per_d = N * tiles_per_n;
        const uint32_t tiles_per_nd = D * tiles_per_d;
        const uint32_t offset_nd = range_start % tiles_per_nd;
        const uint32_t offset_d = offset_nd % tiles_per_d;
        const uint32_t offset_n = offset_d % tiles_per_n;
        const uint32_t offset_c = offset_n % HtWt;
        uint32_t start_nd = range_start / tiles_per_nd;
        uint32_t start_d = offset_nd / tiles_per_d;
        uint32_t start_n = offset_d / tiles_per_n;
        uint32_t start_c = offset_n / HtWt;
        uint32_t start_th = offset_c / Wt;
        uint32_t start_tw = offset_c % Wt;
        uint32_t end_tw = has_sharding ? start_tw + dst_shard_width : Wt;

        uint32_t num_tiles_written = 0;
        uint32_t dst_tile_offset = range_start;

        for (uint32_t nd = start_nd; nd < cND && num_tiles_written < range_tiles; ++nd, start_d = 0) {
            for (uint32_t d = start_d; d < D && num_tiles_written < range_tiles; ++d, start_n = 0) {
                for (uint32_t n = start_n; n < N && num_tiles_written < range_tiles; ++n, start_c = 0) {
                    for (uint32_t c = start_c; c < C && num_tiles_written < range_tiles; ++c, start_th = 0) {
                        for (uint32_t th = start_th; th < Ht && num_tiles_written < range_tiles; ++th) {
                            for (uint32_t tw = start_tw; tw < end_tw && num_tiles_written < range_tiles;
                                 ++tw, ++num_tiles_written) {
#if !DST_SHARDED
                                //  write a tile to dst, since the dst shape is full, the tile offset simply grows
                                //  linearly
                                dfb_dst.wait_front(onetile);
                                noc.async_write(
                                    dfb_dst, dst, dst_tile_bytes, {}, {.page_id = dst_tile_offset + num_tiles_written});
                                noc.async_write_barrier();
                                dfb_dst.pop_front(onetile);
#endif
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
    }
#endif
}
