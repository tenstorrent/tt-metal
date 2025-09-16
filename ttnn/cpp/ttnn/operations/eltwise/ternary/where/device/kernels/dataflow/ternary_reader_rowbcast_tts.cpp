// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // Standard first 5 arguments (same as ternary_reader_nobcast_tts.cpp)
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);  // predicate address
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);  // true tensor address
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);  // unused for TTS (but expected by kernel interface)
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);  // num_tiles_per_core
    const uint32_t start_id = get_arg_val<uint32_t>(4);   // start_tile_id

    // Additional arguments for row broadcast (args 5-26)
    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);  // collapsed dims > 5
    const uint32_t true_nD_stride = get_arg_val<uint32_t>(15);
    const uint32_t true_d_stride = get_arg_val<uint32_t>(16);
    const uint32_t true_n_stride = get_arg_val<uint32_t>(17);
    const uint32_t true_c_stride = get_arg_val<uint32_t>(18);
    const uint32_t true_num_tiles = get_arg_val<uint32_t>(19);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(25);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(26);  // moved to end

    constexpr auto cb_id_src = tt::CBIndex::c_0;    // predicate
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;  // true tensor (matches TTT pattern)

    // Compile-time args layout for TTS: 2 CB ids, then 2 TensorAccessorArgs blocks
    constexpr auto src0_args = TensorAccessorArgs<2>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src0_args, src0_addr, get_tile_size(cb_id_src));
    const auto src_b = TensorAccessor(src1_args, src1_addr, get_tile_size(cb_id_src_b));

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
#if !SRC_BCAST_PREDICATE
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    // For true tensor (CB1) - use true tensor strides
    uint32_t tile_offset_b =
        start_nd * true_nD_stride + start_d * true_d_stride + start_n * true_n_stride + start_c * true_c_stride;
#if !SRC_BCAST_TRUE
    tile_offset_b += start_th * Wt;
#endif
    uint32_t next_c_shift_b = true_c_stride - HtWt;
    uint32_t next_n_shift_b = true_n_stride - true_c_stride * C;
    uint32_t next_d_shift_b = true_d_stride - true_n_stride * N;
    uint32_t next_nd_shift_b = true_nD_stride - true_d_stride * D;

    // Main loop for reading tiles - TTT ROW BROADCAST PATTERN
    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th, start_tw = 0) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_SHARDED_PREDICATE
                            cb_reserve_back(cb_id_src, onetile);
                            uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
                            noc_async_read_tile(tile_offset + tw, src, l1_write_addr_src);
#endif
#if !SRC_SHARDED_TRUE
                            // read a tile from src_b (true tensor)
                            cb_reserve_back(cb_id_src_b, onetile);
                            uint32_t l1_write_addr_b = get_write_ptr(cb_id_src_b);
                            noc_async_read_tile(tile_offset_b + tw, src_b, l1_write_addr_b);
#endif
#if !SRC_SHARDED_PREDICATE || !SRC_SHARDED_TRUE
                            noc_async_read_barrier();
#endif
#if SRC_BCAST_PREDICATE && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW(cb_id_src);
#endif
#if SRC_BCAST_TRUE && !BCAST_LLK  // no sharding support for row bcast yet
                            FILL_TILE_WITH_FIRST_ROW_B(cb_id_src_b);
#endif
#if !SRC_SHARDED_PREDICATE
                            cb_push_back(cb_id_src, onetile);
#endif
#if !SRC_SHARDED_TRUE
                            cb_push_back(cb_id_src_b, onetile);
#endif
                        }
                        if (dst_shard_width == 0) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
#if !SRC_BCAST_PREDICATE
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_TRUE
                        tile_offset_b += Wt;
#endif
                    }
#if !SRC_SHARDED_PREDICATE
#if SRC_BCAST_PREDICATE
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#endif
#if !SRC_SHARDED_TRUE
#if SRC_BCAST_TRUE
                    tile_offset_b += true_c_stride;
#else
                    tile_offset_b += next_c_shift_b;
#endif
#endif
                }
#if !SRC_SHARDED_PREDICATE
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_TRUE
                tile_offset_b += next_n_shift_b;
#endif
            }
#if !SRC_SHARDED_PREDICATE
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_TRUE
            tile_offset_b += next_d_shift_b;
#endif
        }
#if !SRC_SHARDED_PREDICATE
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_TRUE
        tile_offset_b += next_nd_shift_b;
#endif
    }
}
