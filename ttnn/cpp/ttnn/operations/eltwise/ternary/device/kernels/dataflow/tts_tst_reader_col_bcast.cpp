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

void kernel_main() {
    // Standard first 2 arguments
    const uint32_t num_tiles = get_arg(args::num_tiles);  // num_tiles_per_core
    const uint32_t start_id = get_arg(args::start_id);    // start_tile_id

    // Additional arguments for column broadcast
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
    const uint32_t true_nD_stride = get_arg(args::nD_stride_b);
    const uint32_t true_d_stride = get_arg(args::d_stride_b);
    const uint32_t true_n_stride = get_arg(args::n_stride_b);
    const uint32_t true_c_stride = get_arg(args::c_stride_b);
    const uint32_t true_num_tiles = get_arg(args::src_num_tiles_b);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t src_num_tiles = get_arg(args::src_num_tiles);  // moved to end

// CB1 broadcast: For TTS it's true tensor, for TST it's false tensor
// So we check if either true or false tensor needs broadcasting
#define SRC_BCAST_CB1 (SRC_BCAST_B || SRC_BCAST_C)

    Noc noc;
    DataflowBuffer dfb_pred(dfb::in_a);
    DataflowBuffer dfb_true(dfb::in_b);

    const uint32_t src0_tile_bytes = dfb_pred.get_tile_size();
    const auto s0 = TensorAccessor(tensor::in_a);
    const uint32_t src1_tile_bytes = dfb_true.get_tile_size();
    const auto s1 = TensorAccessor(tensor::in_b);

    constexpr uint32_t onetile = 1;
    const uint32_t HtWt = Ht * Wt;

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

    // For true tensor (CB1) - use true tensor strides but predicate dimensions for offset
    uint32_t true_tile_offset =
        start_nd * true_nD_stride + start_d * true_d_stride + start_n * true_n_stride + start_c * true_c_stride;
#if !SRC_BCAST_CB1
    // Use predicate dimensions for offset calculation (same as TTT)
    true_tile_offset += start_th * Wt;
#endif
    uint32_t true_next_c_shift = true_c_stride - HtWt;                 // Use predicate HtWt
    uint32_t true_next_n_shift = true_n_stride - true_c_stride * C;    // Use predicate C
    uint32_t true_next_d_shift = true_d_stride - true_n_stride * N;    // Use predicate N
    uint32_t true_next_nd_shift = true_nD_stride - true_d_stride * D;  // Use predicate D

    // Main loop for reading tiles

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th) {
#if SRC_BCAST_A
                        dfb_pred.reserve_back(onetile);
#if !SRC_SHARDED_A
                        noc.async_read(
                            s0, dfb_pred, src0_tile_bytes, {.page_id = tile_offset + th}, {.offset_bytes = 0});
                        noc.async_read_barrier();
#endif
                        FILL_TILE_WITH_FIRST_COLUMN(dfb_pred.get_write_ptr());
                        dfb_pred.push_back(onetile);
#endif
#if SRC_BCAST_CB1
                        dfb_true.reserve_back(onetile);
#if !SRC_SHARDED_B
                        noc.async_read(
                            s1, dfb_true, src1_tile_bytes, {.page_id = true_tile_offset + th}, {.offset_bytes = 0});
                        noc.async_read_barrier();
#endif
                        FILL_TILE_WITH_FIRST_COLUMN_B(dfb_true.get_write_ptr());
                        dfb_true.push_back(onetile);
#endif

                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_BCAST_A
                            dfb_pred.reserve_back(onetile);
#if !SRC_SHARDED_A
                            noc.async_read(
                                s0, dfb_pred, src0_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
#endif
                            dfb_pred.push_back(onetile);
#endif
#if !SRC_BCAST_CB1
                            dfb_true.reserve_back(onetile);
#if !SRC_SHARDED_B
                            noc.async_read(
                                s1, dfb_true, src1_tile_bytes, {.page_id = true_tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
#endif
                            dfb_true.push_back(onetile);
#endif
                        }
                        // next row of tiles should start at the first column for non-sharded case
                        if (dst_shard_width == 0) {
                            start_tw = 0;
                        }
#if !SRC_BCAST_A && !SRC_SHARDED_A
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_CB1 && !SRC_SHARDED_B
                        true_tile_offset += Wt;
#endif
                    }
#if !SRC_SHARDED_A
#if SRC_BCAST_A
                    // same as following logically
                    // tile_offset += HtWt;
                    // tile_offset += next_c_shift;
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#endif
#if !SRC_SHARDED_B
#if SRC_BCAST_CB1
                    // For broadcast true tensor, use full stride
                    true_tile_offset += true_c_stride;
#else
                    // For non-broadcast true tensor, use incremental stride
                    true_tile_offset += true_next_c_shift;
#endif
#endif
                }
#if !SRC_SHARDED_A
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_B
                true_tile_offset += true_next_n_shift;
#endif
            }
#if !SRC_SHARDED_A
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_B
            true_tile_offset += true_next_d_shift;
#endif
        }
#if !SRC_SHARDED_A
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_B
        true_tile_offset += true_next_nd_shift;
#endif
    }
}
