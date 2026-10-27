// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t dst_num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tile_id = get_arg(args::start_id);
    // Strides and dimensions
    const uint32_t nD_stride = get_arg(args::nD_stride);
    const uint32_t d_stride = get_arg(args::d_stride);
    const uint32_t n_stride = get_arg(args::n_stride);
    const uint32_t c_stride = get_arg(args::c_stride);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);
    const uint32_t nD_stride_b = get_arg(args::nD_stride_b);
    const uint32_t d_stride_b = get_arg(args::d_stride_b);
    const uint32_t n_stride_b = get_arg(args::n_stride_b);
    const uint32_t c_stride_b = get_arg(args::c_stride_b);
    const uint32_t src1_num_tiles = get_arg(args::src_num_tiles_b);
    const uint32_t nD_stride_c = get_arg(args::nD_stride_c);
    const uint32_t d_stride_c = get_arg(args::d_stride_c);
    const uint32_t n_stride_c = get_arg(args::n_stride_c);
    const uint32_t c_stride_c = get_arg(args::c_stride_c);
    const uint32_t src2_num_tiles = get_arg(args::src_num_tiles_c);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t src0_num_tiles = get_arg(args::src_num_tiles);

    Noc noc;
    DataflowBuffer dfb0(dfb::in_a);
    DataflowBuffer dfb1(dfb::in_b);
    DataflowBuffer dfb2(dfb::in_c);

#if SRC_SHARDED_A
    dfb0.reserve_back(src0_num_tiles);
    dfb0.push_back(src0_num_tiles);
#else
    const uint32_t src0_tile_bytes = dfb0.get_entry_size();
    const auto src0 = TensorAccessor(tensor::in_a);
#endif
#if SRC_SHARDED_B
    dfb1.reserve_back(src1_num_tiles);
    dfb1.push_back(src1_num_tiles);
#else
    const uint32_t src1_tile_bytes = dfb1.get_entry_size();
    const auto src1 = TensorAccessor(tensor::in_b);
#endif
#if SRC_SHARDED_C
    dfb2.reserve_back(src2_num_tiles);
    dfb2.push_back(src2_num_tiles);
#else
    const uint32_t src2_tile_bytes = dfb2.get_entry_size();
    const auto src2 = TensorAccessor(tensor::in_c);
#endif
#if !SRC_SHARDED_A || !SRC_SHARDED_B || !SRC_SHARDED_C
    constexpr uint32_t onetile = 1;
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

    // Input tile offsets
    uint32_t tile_offset =
        start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt;
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b + start_th * Wt;
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t tile_offset_c =
        start_nd * nD_stride_c + start_d * d_stride_c + start_n * n_stride_c + start_c * c_stride_c + start_th * Wt;
    uint32_t next_c_shift_c = c_stride_c - HtWt;
    uint32_t next_n_shift_c = n_stride_c - c_stride_c * C;
    uint32_t next_d_shift_c = d_stride_c - n_stride_c * N;
    uint32_t next_nd_shift_c = nD_stride_c - d_stride_c * D;

    uint32_t num_tiles_read = 0;
    // Supports both nobcast and outer_bcast cases for TTT variant
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED_A
                            dfb0.reserve_back(onetile);
                            noc.async_read(
                                src0, dfb0, src0_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_B
                            dfb1.reserve_back(onetile);
                            noc.async_read(
                                src1, dfb1, src1_tile_bytes, {.page_id = tile_offset_b + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_C
                            dfb2.reserve_back(onetile);
                            noc.async_read(
                                src2, dfb2, src2_tile_bytes, {.page_id = tile_offset_c + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_A || !SRC_SHARDED_B || !SRC_SHARDED_C
                            noc.async_read_barrier();
#endif
#if !SRC_SHARDED_A
                            dfb0.push_back(onetile);
#endif
#if !SRC_SHARDED_B
                            dfb1.push_back(onetile);
#endif
#if !SRC_SHARDED_C
                            dfb2.push_back(onetile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
                        tile_offset += Wt;
                        tile_offset_b += Wt;
                        tile_offset_c += Wt;
                    }
                    tile_offset += next_c_shift;
                    tile_offset_b += next_c_shift_b;
                    tile_offset_c += next_c_shift_c;
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
                tile_offset_c += next_n_shift_c;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
            tile_offset_c += next_d_shift_c;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
        tile_offset_c += next_nd_shift_c;
    }
#endif
}
