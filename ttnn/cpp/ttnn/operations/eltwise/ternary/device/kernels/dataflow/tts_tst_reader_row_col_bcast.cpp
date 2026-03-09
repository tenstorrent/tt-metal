// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);

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
    const uint32_t tensor_nD_stride = get_arg_val<uint32_t>(15);
    const uint32_t tensor_d_stride = get_arg_val<uint32_t>(16);
    const uint32_t tensor_n_stride = get_arg_val<uint32_t>(17);
    const uint32_t tensor_c_stride = get_arg_val<uint32_t>(18);
    const uint32_t tensor_num_tiles = get_arg_val<uint32_t>(19);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(25);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(26);

    constexpr auto predicate_cb = get_compile_time_arg_val(0);
    constexpr auto tensor_cb = get_compile_time_arg_val(1);

#define SRC_BCAST_CB1 (SRC_BCAST_B || SRC_BCAST_C)

    constexpr auto src0_args = TensorAccessorArgs<2, 0>();
    constexpr auto src1_args =
        TensorAccessorArgs<src0_args.next_compile_time_args_offset(), src0_args.next_common_runtime_args_offset()>();

    const auto s0 = TensorAccessor(src0_args, src0_addr, get_tile_size(predicate_cb));
    const auto s1 = TensorAccessor(src1_args, src1_addr, get_tile_size(tensor_cb));

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

    uint32_t tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride;
#if !SRC_BCAST_A && !SRC_ROW_BCAST_A
    tile_offset += start_th * Wt;
#endif
    uint32_t next_c_shift = c_stride - HtWt;
    uint32_t next_n_shift = n_stride - c_stride * C;
    uint32_t next_d_shift = d_stride - n_stride * N;
    uint32_t next_nd_shift = nD_stride - d_stride * D;

    uint32_t tensor_tile_offset =
        start_nd * tensor_nD_stride + start_d * tensor_d_stride + start_n * tensor_n_stride + start_c * tensor_c_stride;
#if !SRC_BCAST_CB1 && !SRC_ROW_BCAST_CB1
    tensor_tile_offset += start_th * Wt;
#endif
    uint32_t tensor_next_c_shift = tensor_c_stride - HtWt;
    uint32_t tensor_next_n_shift = tensor_n_stride - tensor_c_stride * C;
    uint32_t tensor_next_d_shift = tensor_d_stride - tensor_n_stride * N;
    uint32_t tensor_next_nd_shift = tensor_nD_stride - tensor_d_stride * D;

    uint32_t num_tiles_read = 0;
    for (uint32_t nd = start_nd; nd < cND && num_tiles_read < num_tiles; ++nd, start_d = 0) {
        for (uint32_t d = start_d; d < D && num_tiles_read < num_tiles; ++d, start_n = 0) {
            for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
                for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
                    for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th) {
#if SRC_BCAST_A
                        cb_reserve_back(predicate_cb, onetile);
                        uint32_t l1_addr_a = get_write_ptr(predicate_cb);
#if SRC_SCALAR_A
                        noc_async_read_page(tile_offset, s0, l1_addr_a);
                        noc_async_read_barrier();
                        FILL_TILE_WITH_FIRST_ELEMENT(predicate_cb);
#else
                        noc_async_read_page(tile_offset + th, s0, l1_addr_a);
                        noc_async_read_barrier();
                        FILL_TILE_WITH_FIRST_COLUMN(predicate_cb);
#endif
                        cb_push_back(predicate_cb, onetile);
#endif

#if SRC_BCAST_CB1
                        cb_reserve_back(tensor_cb, onetile);
                        uint32_t l1_addr_t = get_write_ptr(tensor_cb);
#if SRC_SCALAR_CB1
                        noc_async_read_page(tensor_tile_offset, s1, l1_addr_t);
                        noc_async_read_barrier();
                        FILL_TILE_WITH_FIRST_ELEMENT_B(tensor_cb);
#else
                        noc_async_read_page(tensor_tile_offset + th, s1, l1_addr_t);
                        noc_async_read_barrier();
                        FILL_TILE_WITH_FIRST_COLUMN_B(tensor_cb);
#endif
                        cb_push_back(tensor_cb, onetile);
#endif

                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_BCAST_A
                            cb_reserve_back(predicate_cb, onetile);
                            uint32_t l1_addr_a_inner = get_write_ptr(predicate_cb);
                            noc_async_read_page(tile_offset + tw, s0, l1_addr_a_inner);
                            noc_async_read_barrier();
#if SRC_ROW_BCAST_A
                            FILL_TILE_WITH_FIRST_ROW(predicate_cb);
#endif
                            cb_push_back(predicate_cb, onetile);
#endif

#if !SRC_BCAST_CB1
                            cb_reserve_back(tensor_cb, onetile);
                            uint32_t l1_addr_t_inner = get_write_ptr(tensor_cb);
                            noc_async_read_page(tensor_tile_offset + tw, s1, l1_addr_t_inner);
                            noc_async_read_barrier();
#if SRC_ROW_BCAST_CB1
                            FILL_TILE_WITH_FIRST_ROW_B(tensor_cb);
#endif
                            cb_push_back(tensor_cb, onetile);
#endif
                        }
                        if (dst_shard_width == 0) {
                            start_tw = 0;
                        }
#if !SRC_BCAST_A && !SRC_ROW_BCAST_A
                        tile_offset += Wt;
#endif
#if !SRC_BCAST_CB1 && !SRC_ROW_BCAST_CB1
                        tensor_tile_offset += Wt;
#endif
                    }
#if SRC_BCAST_A || SRC_ROW_BCAST_A
                    tile_offset += c_stride;
#else
                    tile_offset += next_c_shift;
#endif
#if SRC_BCAST_CB1 || SRC_ROW_BCAST_CB1
                    tensor_tile_offset += tensor_c_stride;
#else
                    tensor_tile_offset += tensor_next_c_shift;
#endif
                }
                tile_offset += next_n_shift;
                tensor_tile_offset += tensor_next_n_shift;
            }
            tile_offset += next_d_shift;
            tensor_tile_offset += tensor_next_d_shift;
        }
        tile_offset += next_nd_shift;
        tensor_tile_offset += tensor_next_nd_shift;
    }
}
