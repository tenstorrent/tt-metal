// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

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
    const uint32_t tensor_nD_stride = get_arg(args::nD_stride_b);
    const uint32_t tensor_d_stride = get_arg(args::d_stride_b);
    const uint32_t tensor_n_stride = get_arg(args::n_stride_b);
    const uint32_t tensor_c_stride = get_arg(args::c_stride_b);
    const uint32_t tensor_num_tiles = get_arg(args::src_num_tiles_b);
    const uint32_t dst_shard_width = get_arg(args::dst_shard_width);
    const uint32_t src_num_tiles = get_arg(args::src_num_tiles);

#define SRC_BCAST_CB1 (SRC_BCAST_B || SRC_BCAST_C)

    Noc noc;
    DataflowBuffer dfb_pred(dfb::in_a);
    DataflowBuffer dfb_tensor(dfb::in_b);

    const uint32_t src0_tile_bytes = dfb_pred.get_tile_size();
    const uint32_t src1_tile_bytes = dfb_tensor.get_tile_size();
    const auto s0 = TensorAccessor(tensor::in_a);
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
                        dfb_pred.reserve_back(onetile);
#if SRC_SCALAR_A
                        noc.async_read(s0, dfb_pred, src0_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
#else
                        noc.async_read(
                            s0, dfb_pred, src0_tile_bytes, {.page_id = tile_offset + th}, {.offset_bytes = 0});
#endif
                        noc.async_read_barrier();
#if SRC_SCALAR_A
                        FILL_TILE_WITH_FIRST_ELEMENT(dfb_pred.get_write_ptr());
#else
                        FILL_TILE_WITH_FIRST_COLUMN(dfb_pred.get_write_ptr());
#endif
                        dfb_pred.push_back(onetile);
#endif

#if SRC_BCAST_CB1
                        dfb_tensor.reserve_back(onetile);
#if SRC_SCALAR_CB1
                        noc.async_read(
                            s1, dfb_tensor, src1_tile_bytes, {.page_id = tensor_tile_offset}, {.offset_bytes = 0});
#else
                        noc.async_read(
                            s1, dfb_tensor, src1_tile_bytes, {.page_id = tensor_tile_offset + th}, {.offset_bytes = 0});
#endif
                        noc.async_read_barrier();
#if SRC_SCALAR_CB1
                        FILL_TILE_WITH_FIRST_ELEMENT_B(dfb_tensor.get_write_ptr());
#else
                        FILL_TILE_WITH_FIRST_COLUMN_B(dfb_tensor.get_write_ptr());
#endif
                        dfb_tensor.push_back(onetile);
#endif

                        for (uint32_t tw = start_tw; tw < end_tw && num_tiles_read < num_tiles;
                             ++tw, ++num_tiles_read) {
#if !SRC_BCAST_A
                            dfb_pred.reserve_back(onetile);
                            noc.async_read(
                                s0, dfb_pred, src0_tile_bytes, {.page_id = tile_offset + tw}, {.offset_bytes = 0});
                            noc.async_read_barrier();
#if SRC_ROW_BCAST_A
                            FILL_TILE_WITH_FIRST_ROW(dfb_pred.get_write_ptr());
#endif
                            dfb_pred.push_back(onetile);
#endif

#if !SRC_BCAST_CB1
                            dfb_tensor.reserve_back(onetile);
                            noc.async_read(
                                s1,
                                dfb_tensor,
                                src1_tile_bytes,
                                {.page_id = tensor_tile_offset + tw},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
#if SRC_ROW_BCAST_CB1
                            FILL_TILE_WITH_FIRST_ROW_B(dfb_tensor.get_write_ptr());
#endif
                            dfb_tensor.push_back(onetile);
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
