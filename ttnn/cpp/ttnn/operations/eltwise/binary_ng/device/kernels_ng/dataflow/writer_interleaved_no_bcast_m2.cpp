// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of writer_interleaved_no_bcast.cpp.
// Bound by BinaryNgDeviceOperation::ProgramSpecFactory (no-broadcast x tile x FPU x
// tensor-b-present x interleaved path). Logic is unchanged from the legacy writer; only the
// resource-access mechanism is converted to Metal 2.0 named bindings:
//   - CB id    -> dfb::dst
//   - dst address + TensorAccessorArgs -> ta::dst (address auto-injected)
//   - positional get_arg_val / get_compile_time_arg_val -> get_arg(args::...)
// The #if DST_SHARDED path is preserved verbatim; the ProgramSpecFactory only selects this
// kernel on the interleaved (unsharded) path, so DST_SHARDED is compiled out, but it is kept
// for faithfulness to the source.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
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

    Noc noc;
    CircularBuffer cb_dst(dfb::dst);

#if !DST_SHARDED
    const uint32_t dst_tile_bytes = cb_dst.get_tile_size();
    const auto dst = TensorAccessor(ta::dst);
#endif

#if !DST_SHARDED
    constexpr bool has_sharding = get_arg(args::has_sharding) == 1;
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
#if !DST_SHARDED
                            //  write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                            cb_dst.wait_front(onetile);
                            noc.async_write(
                                cb_dst, dst, dst_tile_bytes, {}, {.page_id = dst_tile_offset + num_tiles_written});
                            noc.async_write_barrier();
                            cb_dst.pop_front(onetile);
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
#endif
}
