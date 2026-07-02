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
    uint32_t src_addr = get_arg_val<uint32_t>(0);        // batch_mean
    uint32_t batch_var_addr = get_arg_val<uint32_t>(1);  // batch_var
    uint32_t weight_addr = get_arg_val<uint32_t>(2);     // weight
    uint32_t bias_addr = get_arg_val<uint32_t>(3);       // bias
    uint32_t dst_addr = get_arg_val<uint32_t>(4);        // output
    uint32_t start_tile_id = get_arg_val<uint32_t>(5);
    uint32_t num_tiles = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t n_stride = get_arg_val<uint32_t>(8);
    uint32_t c_stride = get_arg_val<uint32_t>(9);
    uint32_t N = get_arg_val<uint32_t>(10);
    uint32_t C = get_arg_val<uint32_t>(11);

    constexpr uint32_t onetile = 1;

    // batch_mean
    constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool bias_has_value = get_compile_time_arg_val(1) == 1;
    constexpr auto cb_id_src = get_compile_time_arg_val(2);
    constexpr auto cb_id_dst = get_compile_time_arg_val(3);
    constexpr auto cb_id_batch_var = get_compile_time_arg_val(4);
    constexpr auto cb_id_weight = get_compile_time_arg_val(5);
    constexpr auto cb_id_bias = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto batch_var_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<batch_var_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr bool batch_stat_is_fp32 = get_compile_time_arg_val(bias_args.next_compile_time_args_offset()) == 1;
    constexpr bool param_is_fp32 = get_compile_time_arg_val(bias_args.next_compile_time_args_offset() + 1) == 1;

    Noc noc;
    CircularBuffer cb_src(cb_id_src);
    CircularBuffer cb_dst(cb_id_dst);
    CircularBuffer cb_batch_var(cb_id_batch_var);
    CircularBuffer cb_weight(cb_id_weight);
    CircularBuffer cb_bias(cb_id_bias);

    const uint32_t src_tile_bytes = cb_src.get_tile_size();
    const auto src = TensorAccessor(src_args, src_addr);

    // output
    const uint32_t dst_tile_bytes = cb_dst.get_tile_size();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    // batch_var
    const uint32_t batch_var_tile_bytes = cb_batch_var.get_tile_size();
    const auto batch_var = TensorAccessor(batch_var_args, batch_var_addr);

    // weight
    const uint32_t weight_tile_bytes = cb_weight.get_tile_size();
    const auto weight = TensorAccessor(weight_args, weight_addr);

    // bias
    const uint32_t bias_tile_bytes = cb_bias.get_tile_size();
    const auto bias = TensorAccessor(bias_args, bias_addr);

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    // Input tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_t = 0) {
            // read a tile from src
            cb_src.reserve_back(onetile);
            noc.async_read(src, cb_src, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (batch_stat_is_fp32) {
                fill_tile_with_first_element<float>(cb_src.get_write_ptr());
            } else {
                fill_tile_with_first_element_bfloat16(cb_src.get_write_ptr());
            }
            cb_src.push_back(onetile);

            // read a tile from batch variance
            cb_batch_var.reserve_back(onetile);
            noc.async_read(
                batch_var, cb_batch_var, batch_var_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (batch_stat_is_fp32) {
                fill_tile_with_first_element<float>(cb_batch_var.get_write_ptr());
            } else {
                fill_tile_with_first_element_bfloat16(cb_batch_var.get_write_ptr());
            }
            cb_batch_var.push_back(onetile);

            if constexpr (weight_has_value) {  // read a tile from weight tensor
                cb_weight.reserve_back(onetile);
                noc.async_read(weight, cb_weight, weight_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(cb_weight.get_write_ptr());
                } else {
                    fill_tile_with_first_element_bfloat16(cb_weight.get_write_ptr());
                }
                cb_weight.push_back(onetile);
            }

            if constexpr (bias_has_value) {  // read a tile from bias tensor
                cb_bias.reserve_back(onetile);
                noc.async_read(bias, cb_bias, bias_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(cb_bias.get_write_ptr());
                } else {
                    fill_tile_with_first_element_bfloat16(cb_bias.get_write_ptr());
                }
                cb_bias.push_back(onetile);
            }

            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // write a tile to dst
                cb_dst.wait_front(onetile);
                noc.async_write(
                    cb_dst, dst, dst_tile_bytes, {.offset_bytes = 0}, {.page_id = start_tile_id + num_tiles_written});
                noc.async_write_barrier();
                cb_dst.pop_front(onetile);
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
