// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // Standard first 2 arguments (matching column broadcast pattern)
    const uint32_t num_tiles = get_arg(args::num_tiles);  // num_tiles_per_core
    const uint32_t start_id = get_arg(args::start_id);    // start_tile_id

    // Predicate tensor parameters
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

    // True tensor parameters
    const uint32_t nD_stride_b = get_arg(args::nD_stride_b);
    const uint32_t d_stride_b = get_arg(args::d_stride_b);
    const uint32_t n_stride_b = get_arg(args::n_stride_b);
    const uint32_t c_stride_b = get_arg(args::c_stride_b);
    const uint32_t src_num_tiles_b = get_arg(args::src_num_tiles_b);

    // False tensor parameters
    const uint32_t nD_stride_c = get_arg(args::nD_stride_c);
    const uint32_t d_stride_c = get_arg(args::d_stride_c);
    const uint32_t n_stride_c = get_arg(args::n_stride_c);
    const uint32_t c_stride_c = get_arg(args::c_stride_c);
    const uint32_t src_num_tiles_c = get_arg(args::src_num_tiles_c);

    // Final parameters
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t src_num_tiles = get_arg(args::src_num_tiles);

    // For compatibility, map to old variable names
    const uint32_t start_tile_id = start_id;   // start tile id
    const uint32_t dst_num_tiles = num_tiles;  // num tiles per core

    Noc noc;
    DataflowBuffer dfb_a(dfb::in_a);  // predicate DFB
    DataflowBuffer dfb_b(dfb::in_b);  // true tensor DFB
    DataflowBuffer dfb_c(dfb::in_c);  // false tensor DFB

#if SRC_SHARDED_A
    dfb_a.reserve_back(src_num_tiles);
    dfb_a.push_back(src_num_tiles);
#else
    const uint32_t src_tile_bytes = dfb_a.get_tile_size();
    const auto src = TensorAccessor(tensor::in_a);
#endif
#if SRC_SHARDED_B
    dfb_b.reserve_back(src_num_tiles_b);
    dfb_b.push_back(src_num_tiles_b);
#else
    const uint32_t src_tile_bytes_b = dfb_b.get_tile_size();
    const auto src_b = TensorAccessor(tensor::in_b);
#endif
#if SRC_SHARDED_C
    dfb_c.reserve_back(src_num_tiles_c);
    dfb_c.push_back(src_num_tiles_c);
#else
    const uint32_t src_tile_bytes_c = dfb_c.get_tile_size();
    const auto src_c = TensorAccessor(tensor::in_c);
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

    // this is the INPUT tile offset
    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
#if !SRC_BCAST_A
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b;
#if !SRC_BCAST_B
    tile_offset_b += start_th * Wt;
#endif
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t tile_offset_c =
        start_nd * nD_stride_c + start_d * d_stride_c + start_n * n_stride_c + start_c * c_stride_c;
#if !SRC_BCAST_C
    tile_offset_c += start_th * Wt;
#endif
    uint32_t next_c_shift_c = c_stride_c - HtWt;
    uint32_t next_n_shift_c = n_stride_c - c_stride_c * C;
    uint32_t next_d_shift_c = d_stride_c - n_stride_c * N;
    uint32_t next_nd_shift_c = nD_stride_c - d_stride_c * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED_A
                            dfb_a.reserve_back(onetile);
                            noc.async_read(
                                src, dfb_a, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_B
                            // read a tile from src_b (true tensor)
                            dfb_b.reserve_back(onetile);
                            noc.async_read(
                                src_b, dfb_b, src_tile_bytes_b, {.page_id = tile_offset_b + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_C
                            // read a tile from src_c (false tensor)
                            dfb_c.reserve_back(onetile);
                            noc.async_read(
                                src_c, dfb_c, src_tile_bytes_c, {.page_id = tile_offset_c + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_A || !SRC_SHARDED_B || !SRC_SHARDED_C
                            noc.async_read_barrier();
#endif
#if SRC_BCAST_A && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW(dfb_a.get_write_ptr());
#endif
#if SRC_BCAST_B && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_B(dfb_b.get_write_ptr());
#endif
#if SRC_BCAST_C && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_C(dfb_c.get_write_ptr());
#endif
#if !SRC_SHARDED_A
                            dfb_a.push_back(onetile);
#endif
#if !SRC_SHARDED_B
                            dfb_b.push_back(onetile);
#endif
#if !SRC_SHARDED_C
                            dfb_c.push_back(onetile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
#if !SRC_BCAST_A
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_B
                        tile_offset_b += Wt;
#endif
#if !SRC_BCAST_C
                        tile_offset_c += Wt;
#endif
                    }
#if SRC_BCAST_A
                    // same as following logically
                    // tile_offset += HtWt;
                    // tile_offset += next_c_shift;
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#if SRC_BCAST_B
                    tile_offset_b += c_stride_b;
#else
                    tile_offset_b += next_c_shift_b;
#endif
#if SRC_BCAST_C
                    tile_offset_c += c_stride_c;
#else
                    tile_offset_c += next_c_shift_c;
#endif
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
