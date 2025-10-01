// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // Standard first 5 arguments (same as ternary_reader_nobcast_ttt.cpp)
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);  // predicate address
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);  // true tensor address
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);  // false tensor address
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);  // num_tiles_per_core
    const uint32_t start_id = get_arg_val<uint32_t>(4);   // start_tile_id

    // Additional arguments for column broadcast (args 5-26)
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
    const uint32_t false_nD_stride = get_arg_val<uint32_t>(20);
    const uint32_t false_d_stride = get_arg_val<uint32_t>(21);
    const uint32_t false_n_stride = get_arg_val<uint32_t>(22);
    const uint32_t false_c_stride = get_arg_val<uint32_t>(23);
    const uint32_t false_num_tiles = get_arg_val<uint32_t>(24);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(25);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(26);  // moved to end

    // For compatibility, map to old variable names
    const uint32_t src_addr = src0_addr;       // predicate address
    const uint32_t true_addr = src1_addr;      // true tensor address
    const uint32_t false_addr = src2_addr;     // false tensor address
    const uint32_t start_tile_id = start_id;   // start tile id
    const uint32_t dst_num_tiles = num_tiles;  // num tiles per core

    constexpr auto predicate_cb = get_compile_time_arg_val(0);
    constexpr auto true_cb = get_compile_time_arg_val(1);
    constexpr auto false_cb = get_compile_time_arg_val(2);

    // Compile-time args layout mirrors no-bcast reader: 3 CB ids, then 3 TensorAccessorArgs blocks
    constexpr auto src0_args = TensorAccessorArgs<3>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto src2_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(src0_args, src_addr, get_tile_size(predicate_cb));
    const auto s1 = TensorAccessor(src1_args, true_addr, get_tile_size(true_cb));
    const auto s2 = TensorAccessor(src2_args, false_addr, get_tile_size(false_cb));

    constexpr uint32_t onetile = 1;
    const uint32_t HtWt = Ht * Wt;
    constexpr bool has_sharding = false;

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
#if !SRC_BCAST_PREDICATE
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t true_tile_offset =
        start_nd * true_nD_stride + start_d * true_d_stride + start_n * true_n_stride + start_c * true_c_stride;
#if !SRC_BCAST_TRUE
    true_tile_offset += start_th * Wt;
#endif
    uint32_t true_next_c_shift = true_c_stride - HtWt;
    uint32_t true_next_n_shift = true_n_stride - true_c_stride * C;
    uint32_t true_next_d_shift = true_d_stride - true_n_stride * N;
    uint32_t true_next_nd_shift = true_nD_stride - true_d_stride * D;

    // False tensor offset calculation (same as true tensor for now)
    uint32_t false_tile_offset =
        start_nd * false_nD_stride + start_d * false_d_stride + start_n * false_n_stride + start_c * false_c_stride;
#if !SRC_BCAST_FALSE
    false_tile_offset += start_th * Wt;
#endif
    uint32_t false_next_c_shift = false_c_stride - HtWt;
    uint32_t false_next_n_shift = false_n_stride - false_c_stride * C;
    uint32_t false_next_d_shift = false_d_stride - false_n_stride * N;
    uint32_t false_next_nd_shift = false_nD_stride - false_d_stride * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < dst_num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < dst_num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < dst_num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < dst_num_tiles; ++c, start_th = 0) {
                    // for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
#if SRC_BCAST_PREDICATE
                    cb_reserve_back(predicate_cb, onetile);
#if !SRC_SHARDED_PREDICATE
                    uint32_t l1_write_addr_predicate = get_write_ptr(predicate_cb);
                    noc_async_read_tile(tile_offset, s0, l1_write_addr_predicate);
                    noc_async_read_barrier();
#endif
                    FILL_TILE_WITH_FIRST_ELEMENT(predicate_cb);
                    cb_push_back(predicate_cb, onetile);
#endif
#if SRC_BCAST_TRUE
                    cb_reserve_back(true_cb, onetile);
#if !SRC_SHARDED_TRUE
                    uint32_t l1_write_addr_true = get_write_ptr(true_cb);
                    noc_async_read_tile(true_tile_offset, s1, l1_write_addr_true);
                    noc_async_read_barrier();
#endif
                    FILL_TILE_WITH_FIRST_ELEMENT_B(true_cb);
                    cb_push_back(true_cb, onetile);
#endif

                    // False tensor broadcast (same as true tensor)
#if SRC_BCAST_FALSE
                    cb_reserve_back(false_cb, onetile);
#if !SRC_SHARDED_FALSE
                    uint32_t l1_write_addr_false = get_write_ptr(false_cb);
                    noc_async_read_tile(false_tile_offset, s2, l1_write_addr_false);
                    noc_async_read_barrier();
#endif
                    FILL_TILE_WITH_FIRST_ELEMENT_C(false_cb);
                    cb_push_back(false_cb, onetile);
#endif
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < dst_num_tiles; ++th) {
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_BCAST_PREDICATE
                            cb_reserve_back(predicate_cb, onetile);
#if !SRC_SHARDED_PREDICATE
                            uint32_t l1_write_addr_predicate = get_write_ptr(predicate_cb);
                            noc_async_read_tile(tile_offset + tw, s0, l1_write_addr_predicate);
                            noc_async_read_barrier();
#endif
                            cb_push_back(predicate_cb, onetile);
#endif
#if !SRC_BCAST_TRUE
                            cb_reserve_back(true_cb, onetile);
#if !SRC_SHARDED_TRUE
                            uint32_t l1_write_addr_true = get_write_ptr(true_cb);
                            noc_async_read_tile(true_tile_offset + tw, s1, l1_write_addr_true);
                            noc_async_read_barrier();
#endif
                            cb_push_back(true_cb, onetile);
#endif

                            // False tensor non-broadcast (same as true tensor)
#if !SRC_BCAST_FALSE
                            cb_reserve_back(false_cb, onetile);
#if !SRC_SHARDED_FALSE
                            uint32_t l1_write_addr_false = get_write_ptr(false_cb);
                            noc_async_read_tile(false_tile_offset + tw, s2, l1_write_addr_false);
                            noc_async_read_barrier();
#endif
                            cb_push_back(false_cb, onetile);
#endif
                        }
                        // next row of tiles should start at the first column for non-sharded case
                        if constexpr (!has_sharding) {
                            start_tw = 0;
                        }
#if !SRC_BCAST_PREDICATE && !SRC_SHARDED_PREDICATE
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_TRUE && !SRC_SHARDED_TRUE
                        true_tile_offset += Wt;
#endif
#if !SRC_BCAST_FALSE && !SRC_SHARDED_FALSE
                        false_tile_offset += Wt;
#endif
                    }
#if !SRC_SHARDED_PREDICATE
#if SRC_BCAST_PREDICATE
                    // same as following logically
                    // tile_offset += HtWt;
                    // tile_offset += next_c_shift;
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#endif
#if !SRC_SHARDED_TRUE
#if SRC_BCAST_TRUE
                    true_tile_offset += true_c_stride;
#else
                    true_tile_offset += true_next_c_shift;
#endif
#endif
#if !SRC_SHARDED_FALSE
#if SRC_BCAST_FALSE
                    false_tile_offset += false_c_stride;
#else
                    false_tile_offset += false_next_c_shift;
#endif
#endif
                }
#if !SRC_SHARDED_PREDICATE
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_TRUE
                true_tile_offset += true_next_n_shift;
#endif
#if !SRC_SHARDED_FALSE
                false_tile_offset += false_next_n_shift;
#endif
            }
#if !SRC_SHARDED_PREDICATE
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_TRUE
            true_tile_offset += true_next_d_shift;
#endif
#if !SRC_SHARDED_FALSE
            false_tile_offset += false_next_d_shift;
#endif
        }
#if !SRC_SHARDED_PREDICATE
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_TRUE
        true_tile_offset += true_next_nd_shift;
#endif
#if !SRC_SHARDED_FALSE
        false_tile_offset += false_next_nd_shift;
#endif
    }
}
