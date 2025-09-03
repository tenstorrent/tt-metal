// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // Standard first 5 arguments
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);      // predicate address
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);      // true_value tensor address or false_value tensor address
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);      // none
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);  // num_tiles_per_core
    const uint32_t start_tile_id = get_arg_val<uint32_t>(4);  // start_tile_id

    // Additional arguments for broadcast (args 5-26)
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
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(15);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t srcB_num_tiles = get_arg_val<uint32_t>(19);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(25);
    const uint32_t srcA_num_tiles = get_arg_val<uint32_t>(26);  // moved to end

    constexpr auto predicate_cb = get_compile_time_arg_val(0);
    constexpr auto src_b_cb = get_compile_time_arg_val(1);  // TTS: true_value is src_b; TST: false_value is src_b

    // Compile-time args layout mirrors no-bcast reader: 2 CB ids, then 2 TensorAccessorArgs blocks
    constexpr auto src0_args = TensorAccessorArgs<2>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    // #if SRC_SHARDED_A
    //     cb_reserve_back(predicate_cb, srcA_num_tiles);
    //     cb_push_back(predicate_cb, srcA_num_tiles);
    // #else
    const auto s0 = TensorAccessor(src0_args, src0_addr, get_tile_size(predicate_cb));
    // #endif
    // #if SRC_SHARDED_B
    //     cb_reserve_back(src_b_cb, srcB_num_tiles);
    //     cb_push_back(src_b_cb, srcB_num_tiles);
    // #else
    const auto s1 = TensorAccessor(src1_args, src1_addr, get_tile_size(src_b_cb));
    // #endif

    // #if !SRC_SHARDED_A || !SRC_SHARDED_B
    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = 0;  // TODO: remove this when sharding support is added
    // constexpr bool has_sharding = get_compile_time_arg_val(src2_args.next_compile_time_args_offset()) == 1;
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

    // this is the INPUT_A tile offset
    uint32_t tile_offset =
        start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt;
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    // this is the INPUT_B tile offset
    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b + start_th * Wt;
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
                            // #if !SRC_SHARDED_A
                            // read a tile from src_a
                            cb_reserve_back(predicate_cb, onetile);
                            uint32_t l1_write_addr_a = get_write_ptr(predicate_cb);
                            noc_async_read_tile(tile_offset + tw, s0, l1_write_addr_a);
                            // #endif
                            // #if !SRC_SHARDED_B
                            // read a tile from src_b
                            cb_reserve_back(src_b_cb, onetile);
                            uint32_t l1_write_addr_b = get_write_ptr(src_b_cb);
                            noc_async_read_tile(tile_offset_b + tw, s1, l1_write_addr_b);
                            // #endif

                            // #if !SRC_SHARDED_A || !SRC_SHARDED_B
                            noc_async_read_barrier();
                            // #endif
                            // #if !SRC_SHARDED_A
                            cb_push_back(predicate_cb, onetile);
                            // #endif
                            // #if !SRC_SHARDED_B
                            cb_push_back(src_b_cb, onetile);
                            // #endif
                        }
                        if constexpr (!has_sharding) {
                            // next row of tiles should start at the first column
                            start_tw = 0;
                        }
                        tile_offset += Wt;
                        tile_offset_b += Wt;
                    }
                    tile_offset += next_c_shift;
                    tile_offset_b += next_c_shift_b;
                }
                tile_offset += next_n_shift;
                tile_offset_b += next_n_shift_b;
            }
            tile_offset += next_d_shift;
            tile_offset_b += next_d_shift_b;
        }
        tile_offset += next_nd_shift;
        tile_offset_b += next_nd_shift_b;
    }
    // #endif
}
