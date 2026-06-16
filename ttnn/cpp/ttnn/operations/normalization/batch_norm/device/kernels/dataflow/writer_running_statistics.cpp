// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);               // batch_var
    uint32_t old_running_mean_addr = get_arg_val<uint32_t>(1);  // old running_mean
    uint32_t old_running_var_addr = get_arg_val<uint32_t>(2);   // old running_var
    uint32_t dst_addr = get_arg_val<uint32_t>(3);               // output
    uint32_t start_tile_id = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t HtWt = get_arg_val<uint32_t>(6);
    uint32_t n_stride = get_arg_val<uint32_t>(7);
    uint32_t c_stride = get_arg_val<uint32_t>(8);
    uint32_t N = get_arg_val<uint32_t>(9);
    uint32_t C = get_arg_val<uint32_t>(10);

    constexpr uint32_t onetile = 1;

    constexpr bool old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool old_running_var_has_value = get_compile_time_arg_val(1) == 1;
    constexpr auto cb_id_src = get_compile_time_arg_val(2);
    constexpr auto cb_id_dst = get_compile_time_arg_val(3);
    constexpr auto cb_id_old_running_mean = get_compile_time_arg_val(4);
    constexpr auto cb_id_old_running_var = get_compile_time_arg_val(5);
    constexpr auto cb_id_updated_running_mean = get_compile_time_arg_val(6);
    constexpr auto cb_id_updated_running_var = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto old_running_mean_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto old_running_var_args = TensorAccessorArgs<old_running_mean_args.next_compile_time_args_offset()>();
    constexpr bool old_stat_is_fp32 =
        get_compile_time_arg_val(old_running_var_args.next_compile_time_args_offset()) == 1;

    Noc noc;
    CircularBuffer cb_src(cb_id_src);
    CircularBuffer cb_dst(cb_id_dst);
    CircularBuffer cb_old_mean(cb_id_old_running_mean);
    CircularBuffer cb_old_var(cb_id_old_running_var);
    CircularBuffer cb_new_mean(cb_id_updated_running_mean);
    CircularBuffer cb_new_var(cb_id_updated_running_var);

    const uint32_t src_tile_bytes = cb_src.get_tile_size();
    const auto src = TensorAccessor(src_args, src_addr);

    const uint32_t dst_tile_bytes = cb_dst.get_tile_size();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t old_running_mean_tile_bytes = cb_old_mean.get_tile_size();
    const auto old_running_mean = TensorAccessor(old_running_mean_args, old_running_mean_addr);

    const uint32_t old_running_var_tile_bytes = cb_old_var.get_tile_size();
    const auto old_running_var = TensorAccessor(old_running_var_args, old_running_var_addr);

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
                cb_src.reserve_back(onetile);
                noc.async_read(src, cb_src, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_src.push_back(onetile);

                if constexpr (old_running_mean_has_value) {
                    // read data
                    cb_old_mean.reserve_back(onetile);
                    noc.async_read(
                        old_running_mean,
                        cb_old_mean,
                        old_running_mean_tile_bytes,
                        {.page_id = tile_offset},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    if constexpr (old_stat_is_fp32) {
                        fill_tile_with_first_element<float>(cb_old_mean.get_write_ptr());
                    } else {
                        fill_tile_with_first_element_bfloat16(cb_old_mean.get_write_ptr());
                    }
                    cb_old_mean.push_back(onetile);

                    // write data
                    cb_new_mean.wait_front(onetile);
                    noc.async_write(
                        cb_new_mean,
                        old_running_mean,
                        old_running_mean_tile_bytes,
                        {.offset_bytes = 0},
                        {.page_id = tile_offset});
                    noc.async_write_barrier();
                    cb_new_mean.pop_front(onetile);
                }

                if constexpr (old_running_var_has_value) {
                    // read data
                    cb_old_var.reserve_back(onetile);
                    noc.async_read(
                        old_running_var,
                        cb_old_var,
                        old_running_var_tile_bytes,
                        {.page_id = tile_offset},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    if constexpr (old_stat_is_fp32) {
                        fill_tile_with_first_element<float>(cb_old_var.get_write_ptr());
                    } else {
                        fill_tile_with_first_element_bfloat16(cb_old_var.get_write_ptr());
                    }
                    cb_old_var.push_back(onetile);

                    // write data
                    cb_new_var.wait_front(onetile);
                    noc.async_write(
                        cb_new_var,
                        old_running_var,
                        old_running_var_tile_bytes,
                        {.offset_bytes = 0},
                        {.page_id = tile_offset});
                    noc.async_write_barrier();
                    cb_new_var.pop_front(onetile);
                }
                ++tile_offset;

                // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                cb_dst.wait_front(onetile);
                noc.async_write(
                    cb_dst, dst, dst_tile_bytes, {.offset_bytes = 0}, {.page_id = start_tile_id + num_tiles_written});
                noc.async_write_barrier();
                cb_dst.pop_front(onetile);
            }
            tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}
