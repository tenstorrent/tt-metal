// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of kernels/dataflow/reader_interleaved_no_bcast.cpp (the SINGLE-tensor reader
// used by the scalar-b path: a present, b absent -> only src is read; the scalar tile is filled
// by the writer). Bound by BinaryNgDeviceOperation::ProgramSpecFactory on the
//   no-broadcast x tile x FPU x scalar-b (no tensor b) x interleaved path.
// Logic is unchanged from the legacy reader; only the resource-access mechanism is converted to
// Metal 2.0 named bindings:
//   - CB id    -> dfb::src
//   - src address + TensorAccessorArgs -> ta::src (address auto-injected)
//   - positional get_arg_val / get_compile_time_arg_val -> get_arg(args::...)
// The #if SRC_SHARDED path is preserved verbatim; the ProgramSpecFactory only selects this kernel
// on the interleaved (unsharded) path, so SRC_SHARDED is compiled out, but it is kept for
// faithfulness to the source.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t src_num_tiles = get_arg(args::src_num_tiles);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t nD_stride = get_arg(args::nD_stride);
    const uint32_t d_stride = get_arg(args::d_stride);
    const uint32_t n_stride = get_arg(args::n_stride);
    const uint32_t c_stride = get_arg(args::c_stride);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);  // collapsed dims > 5

    Noc noc;
    CircularBuffer cb_src(dfb::src);

#if SRC_SHARDED
    cb_src.reserve_back(src_num_tiles);
    cb_src.push_back(src_num_tiles);
#else
    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_arg(args::has_sharding) == 1;
    const uint32_t src_tile_bytes = cb_src.get_tile_size();
    const auto src = TensorAccessor(ta::src);
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
    uint32_t tile_offset =
        start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt;
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            cb_src.reserve_back(onetile);
                            noc.async_read(
                                src, cb_src, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
                            cb_src.push_back(onetile);
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
                        tile_offset += Wt;
                    }
                    tile_offset += next_c_shift;
                }
                tile_offset += next_n_shift;
            }
            tile_offset += next_d_shift;
        }
        tile_offset += next_nd_shift;
    }
#endif
}
