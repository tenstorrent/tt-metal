// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

// BCAST_LLK is only emitted by the host for the TTT variant; TTS/TST always take the
// software fill path. Default it here so the guards below do not rely on an undefined
// macro evaluating to 0.
#ifndef BCAST_LLK
#define BCAST_LLK 0
#endif

void kernel_main() {
    // Standard first 2 arguments
    const uint32_t num_tiles = get_arg(args::num_tiles);  // num_tiles_per_core
    const uint32_t start_id = get_arg(args::start_id);    // start_tile_id

    // Additional arguments for row broadcast
    const uint32_t nD_stride = get_arg(args::nD_stride);
    const uint32_t d_stride = get_arg(args::d_stride);
    const uint32_t n_stride = get_arg(args::n_stride);
    const uint32_t c_stride = get_arg(args::c_stride);
    const uint32_t D = get_arg(args::D);
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t cND = get_arg(args::cND);                      // collapsed dims > 5
    const uint32_t src_b_nD_stride = get_arg(args::nD_stride_b);  // TTS: true strides, TST: false strides
    const uint32_t src_b_d_stride = get_arg(args::d_stride_b);
    const uint32_t src_b_n_stride = get_arg(args::n_stride_b);
    const uint32_t src_b_c_stride = get_arg(args::c_stride_b);
    const uint32_t src_b_num_tiles = get_arg(args::src_num_tiles_b);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t src_num_tiles = get_arg(args::src_num_tiles);  // moved to end

    Noc noc;
    DataflowBuffer dfb_src(dfb::in_a);    // predicate
    DataflowBuffer dfb_src_b(dfb::in_b);  // TTS: true tensor, TST: false tensor

    const uint32_t src_tile_bytes = dfb_src.get_tile_size();
    const auto src = TensorAccessor(tensor::in_a);
    const uint32_t src_tile_bytes_b = dfb_src_b.get_tile_size();
    const auto src_b = TensorAccessor(tensor::in_b);

    constexpr uint32_t onetile = 1;
    const uint32_t HtWt = Ht * Wt;
    const uint32_t dst_num_tiles = num_tiles;

    const uint32_t tiles_per_n = C * HtWt;
    const uint32_t tiles_per_d = N * tiles_per_n;
    const uint32_t tiles_per_nd = D * tiles_per_d;
    const uint32_t offset_nd = start_id % tiles_per_nd;
    const uint32_t offset_d = offset_nd % tiles_per_d;
    const uint32_t offset_n = offset_d % tiles_per_n;
    const uint32_t offset_c = offset_n % HtWt;
    uint32_t start_nd = start_id / tiles_per_nd;
    uint32_t start_d = offset_nd / tiles_per_d;
    uint32_t start_n = offset_d / tiles_per_n;
    uint32_t start_c = offset_n / HtWt;
    uint32_t start_th = offset_c / Wt;
    uint32_t start_tw = offset_c % Wt;
    uint32_t end_tw = (dst_shard_width != 0) ? (start_tw + dst_shard_width) : Wt;

    // this is the INPUT tile offset for predicate
    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
#if !SRC_BCAST_A
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    // For src_b tensor (CB1) - use src_b tensor strides
    uint32_t tile_offset_b =
        start_nd * src_b_nD_stride + start_d * src_b_d_stride + start_n * src_b_n_stride + start_c * src_b_c_stride;
#if !SRC_BCAST_B
    tile_offset_b += start_th * Wt;
#endif
    uint32_t next_c_shift_b = src_b_c_stride - HtWt;
    uint32_t next_n_shift_b = src_b_n_stride - src_b_c_stride * C;
    uint32_t next_d_shift_b = src_b_d_stride - src_b_n_stride * N;
    uint32_t next_nd_shift_b = src_b_nD_stride - src_b_d_stride * D;

    // Main loop for reading tiles - TTT ROW BROADCAST PATTERN
    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th, start_tw = 0) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED_A
                            dfb_src.reserve_back(onetile);
                            noc.async_read(
                                src, dfb_src, src_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_B
                            // read a tile from src_b (TTS: true tensor, TST: false tensor)
                            dfb_src_b.reserve_back(onetile);
                            noc.async_read(
                                src_b,
                                dfb_src_b,
                                src_tile_bytes_b,
                                {.page_id = tile_offset_b + tw},
                                {.offset_bytes = 0});
#endif
#if !SRC_SHARDED_A || !SRC_SHARDED_B
                            noc.async_read_barrier();
#endif
#if SRC_BCAST_A && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW(dfb_src.get_write_ptr());
#endif
#if SRC_BCAST_B && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_B(dfb_src_b.get_write_ptr());
#endif
#if !SRC_SHARDED_A
                            dfb_src.push_back(onetile);
#endif
#if !SRC_SHARDED_B
                            dfb_src_b.push_back(onetile);
#endif
                        }
                        if (dst_shard_width == 0) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
#if !SRC_BCAST_A
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_B
                        tile_offset_b += Wt;
#endif
                    }
#if !SRC_SHARDED_A
#if SRC_BCAST_A
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#endif
#if !SRC_SHARDED_B
#if SRC_BCAST_B
                    tile_offset_b += src_b_c_stride;
#else
                    tile_offset_b += next_c_shift_b;
#endif
#endif
                }
#if !SRC_SHARDED_A
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_B
                tile_offset_b += next_n_shift_b;
#endif
            }
#if !SRC_SHARDED_A
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_B
            tile_offset_b += next_d_shift_b;
#endif
        }
#if !SRC_SHARDED_A
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_B
        tile_offset_b += next_nd_shift_b;
#endif
    }
}
