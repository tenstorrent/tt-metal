// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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

    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const auto src = TensorAccessor(src_args, src_addr, src_tile_bytes);

    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const auto dst = TensorAccessor(dst_args, dst_addr, dst_tile_bytes);

    // batch_var
    const uint32_t batch_var_tile_bytes = get_tile_size(cb_id_batch_var);
    const auto batch_var = TensorAccessor(batch_var_args, batch_var_addr, batch_var_tile_bytes);

    // weight
    const uint32_t weight_tile_bytes = get_tile_size(cb_id_weight);
    const auto weight = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);

    // bias
    const uint32_t bias_tile_bytes = get_tile_size(cb_id_bias);
    const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_id_src_obj(cb_id_src);
    experimental::CircularBuffer cb_id_dst_obj(cb_id_dst);
    experimental::CircularBuffer cb_id_batch_var_obj(cb_id_batch_var);
    experimental::CircularBuffer cb_id_weight_obj(cb_id_weight);
    experimental::CircularBuffer cb_id_bias_obj(cb_id_bias);

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
            cb_id_src_obj.reserve_back(onetile);
            noc.async_read(src, cb_id_src_obj, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (batch_stat_is_fp32) {
                fill_tile_with_first_element<float>(cb_id_src);
            } else {
                fill_tile_with_first_element_bfloat16(cb_id_src);
            }
            cb_id_src_obj.push_back(onetile);

            // read a tile from batch variance
            cb_id_batch_var_obj.reserve_back(onetile);
            noc.async_read(
                batch_var, cb_id_batch_var_obj, batch_var_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (batch_stat_is_fp32) {
                fill_tile_with_first_element<float>(cb_id_batch_var);
            } else {
                fill_tile_with_first_element_bfloat16(cb_id_batch_var);
            }
            cb_id_batch_var_obj.push_back(onetile);

            if constexpr (weight_has_value) {  // read a tile from weight tensor
                cb_id_weight_obj.reserve_back(onetile);
                noc.async_read(
                    weight, cb_id_weight_obj, weight_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(cb_id_weight);
                } else {
                    fill_tile_with_first_element_bfloat16(cb_id_weight);
                }
                cb_id_weight_obj.push_back(onetile);
            }

            if constexpr (bias_has_value) {  // read a tile from bias tensor
                cb_id_bias_obj.reserve_back(onetile);
                noc.async_read(bias, cb_id_bias_obj, bias_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if constexpr (param_is_fp32) {
                    fill_tile_with_first_element<float>(cb_id_bias);
                } else {
                    fill_tile_with_first_element_bfloat16(cb_id_bias);
                }
                cb_id_bias_obj.push_back(onetile);
            }

            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // write a tile to dst
                cb_id_dst_obj.wait_front(onetile);
                noc.async_write(
                    cb_id_dst_obj,
                    dst,
                    dst_tile_bytes,
                    {.offset_bytes = 0},
                    {.page_id = start_tile_id + num_tiles_written});
                noc.async_write_barrier();
                cb_id_dst_obj.pop_front(onetile);
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
