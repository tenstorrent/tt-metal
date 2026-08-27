// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_tile_id = get_arg(args::start_tile_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t n_stride = get_arg(args::n_stride);
    uint32_t c_stride = get_arg(args::c_stride);
    uint32_t N = get_arg(args::N);
    uint32_t C = get_arg(args::C);

    constexpr uint32_t onetile = 1;

    constexpr bool old_stat_is_fp32 = get_arg(args::old_stat_is_fp32) == 1;

    Noc noc;
    DataflowBuffer dfb_src(dfb::src);            // batch_var, read here for the compute kernel
    DataflowBuffer dfb_dst(dfb::dst);            // the op's output tensor
    DataflowBuffer dfb_old_mean(dfb::old_mean);  // old running mean; bound even when the tensor is absent
    DataflowBuffer dfb_old_var(dfb::old_var);    // old running var; bound even when the tensor is absent
    DataflowBuffer dfb_new_mean(dfb::new_mean);  // updated running mean, as the compute kernel leaves it
    DataflowBuffer dfb_new_var(dfb::new_var);    // updated running var, likewise
    auto old_running_mean = construct_nullable_tensor(tensor::running_mean);
    auto old_running_var = construct_nullable_tensor(tensor::running_var);

    const uint32_t src_tile_bytes = dfb_src.get_entry_size();
    const auto src = TensorAccessor(tensor::batch_var);

    const uint32_t dst_tile_bytes = dfb_dst.get_entry_size();
    const auto dst = TensorAccessor(tensor::output);

    const uint32_t old_running_mean_tile_bytes = dfb_old_mean.get_entry_size();
    const uint32_t old_running_var_tile_bytes = dfb_old_var.get_entry_size();

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;
    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_t = 0) {
            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // read a tile from src
                dfb_src.reserve_back(onetile);
                noc.async_read(src, dfb_src, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_src.push_back(onetile);

                with_nullable_token(tensor::running_mean, [&](auto const&) {
                        // read data
                        dfb_old_mean.reserve_back(onetile);
                        noc.async_read(
                            old_running_mean,
                            dfb_old_mean,
                            old_running_mean_tile_bytes,
                            {.page_id = tile_offset},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                        if constexpr (old_stat_is_fp32) {
                            fill_tile_with_first_element<float>(dfb_old_mean.get_write_ptr());
                        } else {
                            fill_tile_with_first_element_bfloat16(dfb_old_mean.get_write_ptr());
                        }
                        dfb_old_mean.push_back(onetile);

                        // write data
                        dfb_new_mean.wait_front(onetile);
                        noc.async_write(
                            dfb_new_mean,
                            old_running_mean,
                            old_running_mean_tile_bytes,
                            {.offset_bytes = 0},
                            {.page_id = tile_offset});
                        noc.async_write_barrier();
                        dfb_new_mean.pop_front(onetile);
                    });

                with_nullable_token(tensor::running_var, [&](auto const&) {
                        // read data
                        dfb_old_var.reserve_back(onetile);
                        noc.async_read(
                            old_running_var,
                            dfb_old_var,
                            old_running_var_tile_bytes,
                            {.page_id = tile_offset},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                        if constexpr (old_stat_is_fp32) {
                            fill_tile_with_first_element<float>(dfb_old_var.get_write_ptr());
                        } else {
                            fill_tile_with_first_element_bfloat16(dfb_old_var.get_write_ptr());
                        }
                        dfb_old_var.push_back(onetile);

                        // write data
                        dfb_new_var.wait_front(onetile);
                        noc.async_write(
                            dfb_new_var,
                            old_running_var,
                            old_running_var_tile_bytes,
                            {.offset_bytes = 0},
                            {.page_id = tile_offset});
                        noc.async_write_barrier();
                        dfb_new_var.pop_front(onetile);
                    });
                ++tile_offset;

                // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                dfb_dst.wait_front(onetile);
                noc.async_write(
                    dfb_dst, dst, dst_tile_bytes, {.offset_bytes = 0}, {.page_id = start_tile_id + num_tiles_written});
                noc.async_write_barrier();
                dfb_dst.pop_front(onetile);
            }
            tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}
