// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);

    const uint32_t nD_stride_b = get_arg_val<uint32_t>(15);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t src_num_tiles_b = get_arg_val<uint32_t>(19);

    const uint32_t nD_stride_c = get_arg_val<uint32_t>(20);
    const uint32_t d_stride_c = get_arg_val<uint32_t>(21);
    const uint32_t n_stride_c = get_arg_val<uint32_t>(22);
    const uint32_t c_stride_c = get_arg_val<uint32_t>(23);
    const uint32_t src_num_tiles_c = get_arg_val<uint32_t>(24);

    const uint32_t dst_shard_width = get_arg_val<uint32_t>(25);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(26);

    const uint32_t start_tile_id = start_id;
    const uint32_t dst_num_tiles = num_tiles;

    constexpr auto predicate_cb = get_compile_time_arg_val(0);
    constexpr auto true_cb = get_compile_time_arg_val(1);
    constexpr auto false_cb = get_compile_time_arg_val(2);

    constexpr auto src0_args = TensorAccessorArgs<3, 0>();
    constexpr auto src1_args =
        TensorAccessorArgs<src0_args.next_compile_time_args_offset(), src0_args.next_common_runtime_args_offset()>();
    constexpr auto src2_args =
        TensorAccessorArgs<src1_args.next_compile_time_args_offset(), src1_args.next_common_runtime_args_offset()>();

    experimental::Noc noc;
    experimental::CircularBuffer cb_pred(predicate_cb);
    experimental::CircularBuffer cb_true(true_cb);
    experimental::CircularBuffer cb_false(false_cb);

#if SRC_SHARDED_A
    cb_pred.reserve_back(src_num_tiles);
    cb_pred.push_back(src_num_tiles);
#else
    const uint32_t src0_tile_bytes = get_tile_size(predicate_cb);
    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
#endif
#if SRC_SHARDED_B
    cb_true.reserve_back(src_num_tiles_b);
    cb_true.push_back(src_num_tiles_b);
#else
    const uint32_t src1_tile_bytes = get_tile_size(true_cb);
    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);
#endif
#if SRC_SHARDED_C
    cb_false.reserve_back(src_num_tiles_c);
    cb_false.push_back(src_num_tiles_c);
#else
    const uint32_t src2_tile_bytes = get_tile_size(false_cb);
    const auto s2 = TensorAccessor(src2_args, src2_addr, src2_tile_bytes);
#endif

    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(src2_args.next_compile_time_args_offset()) == 1;
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
    uint32_t end_tw = has_sharding ? (start_tw + dst_shard_width) : Wt;

    // Predicate offset: col/scalar bcast tensors skip start_th*Wt, row/full include it
    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
#if !SRC_BCAST_A && !SRC_ROW_BCAST_A
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    // True tensor offset
    uint32_t tile_offset_b =
        start_nd * nD_stride_b + start_d * d_stride_b + start_n * n_stride_b + start_c * c_stride_b;
#if !SRC_BCAST_B && !SRC_ROW_BCAST_B
    tile_offset_b += start_th * Wt;
#endif
    uint32_t next_c_shift_b = c_stride_b - HtWt;
    uint32_t next_n_shift_b = n_stride_b - c_stride_b * C;
    uint32_t next_d_shift_b = d_stride_b - n_stride_b * N;
    uint32_t next_nd_shift_b = nD_stride_b - d_stride_b * D;

    // False tensor offset
    uint32_t tile_offset_c =
        start_nd * nD_stride_c + start_d * d_stride_c + start_n * n_stride_c + start_c * c_stride_c;
#if !SRC_BCAST_C && !SRC_ROW_BCAST_C
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
                        // --- Col broadcast / scalar tensors: push one tile per row ---
#if SRC_BCAST_A
                        cb_pred.reserve_back(onetile);
#if !SRC_SHARDED_A
#if SRC_SCALAR_A
                        noc.async_read(s0, cb_pred, src0_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
#else
                        noc.async_read(
                            s0, cb_pred, src0_tile_bytes, {.page_id = tile_offset + th}, {.offset_bytes = 0});
#endif
                        noc.async_read_barrier();
#endif
#if SRC_SCALAR_A
                        FILL_TILE_WITH_FIRST_ELEMENT(predicate_cb);
#else
                        FILL_TILE_WITH_FIRST_COLUMN(predicate_cb);
#endif
                        cb_pred.push_back(onetile);
#endif

#if SRC_BCAST_B
                        cb_true.reserve_back(onetile);
#if !SRC_SHARDED_B
#if SRC_SCALAR_B
                        noc.async_read(s1, cb_true, src1_tile_bytes, {.page_id = tile_offset_b}, {.offset_bytes = 0});
#else
                        noc.async_read(
                            s1, cb_true, src1_tile_bytes, {.page_id = tile_offset_b + th}, {.offset_bytes = 0});
#endif
                        noc.async_read_barrier();
#endif
#if SRC_SCALAR_B
                        FILL_TILE_WITH_FIRST_ELEMENT_B(true_cb);
#else
                        FILL_TILE_WITH_FIRST_COLUMN_B(true_cb);
#endif
                        cb_true.push_back(onetile);
#endif

#if SRC_BCAST_C
                        cb_false.reserve_back(onetile);
#if !SRC_SHARDED_C
#if SRC_SCALAR_C
                        noc.async_read(s2, cb_false, src2_tile_bytes, {.page_id = tile_offset_c}, {.offset_bytes = 0});
#else
                        noc.async_read(
                            s2, cb_false, src2_tile_bytes, {.page_id = tile_offset_c + th}, {.offset_bytes = 0});
#endif
                        noc.async_read_barrier();
#endif
#if SRC_SCALAR_C
                        FILL_TILE_WITH_FIRST_ELEMENT_C(false_cb);
#else
                        FILL_TILE_WITH_FIRST_COLUMN_C(false_cb);
#endif
                        cb_false.push_back(onetile);
#endif
                        // --- Inner loop: row broadcast and full tensors ---
                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < dst_num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_BCAST_A
                            cb_pred.reserve_back(onetile);
#if !SRC_SHARDED_A
                            noc.async_read(
                                s0, cb_pred, src0_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
#endif
#if SRC_ROW_BCAST_A
                            FILL_TILE_WITH_FIRST_ROW(predicate_cb);
#endif
                            cb_pred.push_back(onetile);
#endif

#if !SRC_BCAST_B
                            cb_true.reserve_back(onetile);
#if !SRC_SHARDED_B
                            noc.async_read(
                                s1, cb_true, src1_tile_bytes, {.page_id = tile_offset_b + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
#endif
#if SRC_ROW_BCAST_B
                            FILL_TILE_WITH_FIRST_ROW_B(true_cb);
#endif
                            cb_true.push_back(onetile);
#endif

#if !SRC_BCAST_C
                            cb_false.reserve_back(onetile);
#if !SRC_SHARDED_C
                            noc.async_read(
                                s2, cb_false, src2_tile_bytes, {.page_id = tile_offset_c + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
#endif
#if SRC_ROW_BCAST_C
                            FILL_TILE_WITH_FIRST_ROW_C(false_cb);
#endif
                            cb_false.push_back(onetile);
#endif
                        }
                        if constexpr (!has_sharding) {
                            start_tw = 0;
                        }
                        // Advance tile offsets for next row: only full tensors advance by Wt
#if !SRC_BCAST_A && !SRC_ROW_BCAST_A && !SRC_SHARDED_A
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_B && !SRC_ROW_BCAST_B && !SRC_SHARDED_B
                        tile_offset_b += Wt;
#endif
#if !SRC_BCAST_C && !SRC_ROW_BCAST_C && !SRC_SHARDED_C
                        tile_offset_c += Wt;
#endif
                    }
                    // After all rows in a C block: advance to next C block
#if !SRC_SHARDED_A
#if SRC_BCAST_A || SRC_ROW_BCAST_A
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#endif
#if !SRC_SHARDED_B
#if SRC_BCAST_B || SRC_ROW_BCAST_B
                    tile_offset_b += c_stride_b;
#else
                    tile_offset_b += next_c_shift_b;
#endif
#endif
#if !SRC_SHARDED_C
#if SRC_BCAST_C || SRC_ROW_BCAST_C
                    tile_offset_c += c_stride_c;
#else
                    tile_offset_c += next_c_shift_c;
#endif
#endif
                }
#if !SRC_SHARDED_A
                tile_offset += next_n_shift;
#endif
#if !SRC_SHARDED_B
                tile_offset_b += next_n_shift_b;
#endif
#if !SRC_SHARDED_C
                tile_offset_c += next_n_shift_c;
#endif
            }
#if !SRC_SHARDED_A
            tile_offset += next_d_shift;
#endif
#if !SRC_SHARDED_B
            tile_offset_b += next_d_shift_b;
#endif
#if !SRC_SHARDED_C
            tile_offset_c += next_d_shift_c;
#endif
        }
#if !SRC_SHARDED_A
        tile_offset += next_nd_shift;
#endif
#if !SRC_SHARDED_B
        tile_offset_b += next_nd_shift_b;
#endif
#if !SRC_SHARDED_C
        tile_offset_c += next_nd_shift_c;
#endif
    }
}
