// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) in0-only reader for binary_ng's tensor-SCALAR op.
//
// Copy of reader_no_bcast_dfb.cpp with every in1 element removed: `in1` is produced by
// writer_scalar_dfb.cpp (the scalar fill), so this reader must not produce `in1`. The entire in0
// path is kept untouched -- the borrow branch (SRC_SHARDED bulk publish), the interleaved NoC walk
// (TensorAccessor(tensor::in0)), the HAS_SHARDING row-wrap, and the in0 strides. The in1 DFB, its
// TensorAccessor, its per-tile read, and all *_b runtime args / tile_offset_b are gone.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
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
    DataflowBuffer dfb_in0(dfb::in0);

#if SRC_SHARDED
    dfb_in0.reserve_back(src_num_tiles);
    dfb_in0.push_back(src_num_tiles);
#else
    const uint32_t src_tile_bytes = dfb_in0.get_entry_size();
    const auto src = TensorAccessor(tensor::in0);
    constexpr uint32_t onetile = 1;
    // HAS_SHARDING (factory-set) mirrors the CB kernel's sharded-output row handling for an
    // interleaved operand: when set, each tile row spans only dst_shard_width tiles before wrapping.
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
                            dfb_in0.reserve_back(onetile);
                            noc.async_read(
                                src, dfb_in0, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
                            dfb_in0.push_back(onetile);
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
