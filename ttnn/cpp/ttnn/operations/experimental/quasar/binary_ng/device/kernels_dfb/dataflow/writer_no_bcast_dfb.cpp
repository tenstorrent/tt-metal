// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for binary_ng's no-broadcast binary op.
//
// 1:1 mirror of the CircularBuffer writer kernels_ng/dataflow/writer_interleaved_no_bcast.cpp, with
// the CB->DFB API swap. The output DFB (out) is either:
//   - SHARDED  (#if DST_SHARDED): the DFB borrows the output's resident L1 shard and the compute
//     kernel packs directly into it, so there is no NoC write — the writer only drains the ring
//     credits (wait_front + pop_front) back to the compute producer.
//   - INTERLEAVED (#else): the DFB is an allocated ring the compute kernel fills; the writer drains
//     each tile to DRAM via a NoC write, addressed through a Metal 2.0 binding TensorAccessor(tensor::out).

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);  // collapsed dims > 5

    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_out(dfb::out);

#if DST_SHARDED
    // Output shard is written in place by the compute kernel; just drain the ring credits.
    dfb_out.wait_front(dst_num_tiles);
    dfb_out.pop_front(dst_num_tiles);
#else
    Noc noc;
    const uint32_t dst_tile_bytes = dfb_out.get_entry_size();
    const auto dst = TensorAccessor(tensor::out);

    // HAS_SHARDING (factory-set) mirrors the CB writer's sharded-output row handling.
    constexpr bool has_sharding = HAS_SHARDING;
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
                            //  write a tile to dst; for a full dst the tile offset grows linearly
                            dfb_out.wait_front(onetile);
                            noc.async_write(
                                dfb_out, dst, dst_tile_bytes, {}, {.page_id = dst_tile_offset + num_tiles_written});
                            noc.async_write_barrier();
                            dfb_out.pop_front(onetile);
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
