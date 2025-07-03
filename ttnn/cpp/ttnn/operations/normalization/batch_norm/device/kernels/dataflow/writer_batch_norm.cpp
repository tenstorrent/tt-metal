// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

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
    constexpr auto cb_id_src = get_compile_time_arg_val(7);
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);

    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    // output
    constexpr auto cb_id_dst = get_compile_time_arg_val(8);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const DataFormat dst_data_format = get_dataformat(cb_id_dst);

    const InterleavedAddrGenFast<dst_is_dram> dst = {
        .bank_base_address = dst_addr, .page_size = dst_tile_bytes, .data_format = dst_data_format};

    // batch_var
    constexpr auto cb_id_batch_var = get_compile_time_arg_val(9);
    constexpr bool batch_var_is_dram = get_compile_time_arg_val(2) == 1;
    const uint32_t batch_var_tile_bytes = get_tile_size(cb_id_batch_var);
    const DataFormat batch_var_data_format = get_dataformat(cb_id_batch_var);

    const InterleavedAddrGenFast<batch_var_is_dram> batch_var = {
        .bank_base_address = batch_var_addr, .page_size = batch_var_tile_bytes, .data_format = batch_var_data_format};

    // weight
    constexpr auto cb_id_weight = get_compile_time_arg_val(10);
    constexpr bool weight_is_dram = get_compile_time_arg_val(3) == 1;
    const uint32_t weight_tile_bytes = get_tile_size(cb_id_weight);
    const DataFormat weight_data_format = get_dataformat(cb_id_weight);

    const InterleavedAddrGenFast<weight_is_dram> weight = {
        .bank_base_address = weight_addr, .page_size = weight_tile_bytes, .data_format = weight_data_format};

    // bias
    constexpr auto cb_id_bias = get_compile_time_arg_val(11);
    constexpr bool bias_is_dram = get_compile_time_arg_val(4) == 1;
    const uint32_t bias_tile_bytes = get_tile_size(cb_id_bias);
    const DataFormat bias_data_format = get_dataformat(cb_id_bias);

    const InterleavedAddrGenFast<bias_is_dram> bias = {
        .bank_base_address = bias_addr, .page_size = bias_tile_bytes, .data_format = bias_data_format};

    constexpr bool weight_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool bias_has_value = get_compile_time_arg_val(6) == 1;

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
            cb_reserve_back(cb_id_src, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_src);
            noc_async_read_tile(tile_offset, src, l1_write_addr);
            noc_async_read_barrier();
            FILL_TILE_WITH_FIRST_ELEMENT(cb_id_src);
            cb_push_back(cb_id_src, onetile);

            // read a tile from batch variance
            cb_reserve_back(cb_id_batch_var, onetile);
            uint32_t l1_batch_var_write_addr = get_write_ptr(cb_id_batch_var);
            noc_async_read_tile(tile_offset, batch_var, l1_batch_var_write_addr);
            noc_async_read_barrier();
            FILL_TILE_WITH_FIRST_ELEMENT(cb_id_batch_var);
            cb_push_back(cb_id_batch_var, onetile);

            if constexpr (weight_has_value) {  // read a tile from weight tensor
                cb_reserve_back(cb_id_weight, onetile);
                uint32_t l1_weight_write_addr = get_write_ptr(cb_id_weight);
                noc_async_read_tile(tile_offset, weight, l1_weight_write_addr);
                noc_async_read_barrier();
                FILL_TILE_WITH_FIRST_ELEMENT(cb_id_weight);
                cb_push_back(cb_id_weight, onetile);
            }

            if constexpr (bias_has_value) {  // read a tile from bias tensor
                cb_reserve_back(cb_id_bias, onetile);
                uint32_t l1_bias_write_addr = get_write_ptr(cb_id_bias);
                noc_async_read_tile(tile_offset, bias, l1_bias_write_addr);
                noc_async_read_barrier();
                FILL_TILE_WITH_FIRST_ELEMENT(cb_id_bias);
                cb_push_back(cb_id_bias, onetile);
            }

            for (uint32_t t = start_t; t < HtWt && num_tiles_written < num_tiles; ++t, ++num_tiles_written) {
                // write a tile to dst
                cb_wait_front(cb_id_dst, onetile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_dst);
                noc_async_write_tile(start_tile_id + num_tiles_written, dst, l1_read_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_id_dst, onetile);
            }
            tile_offset += c_stride;
        }
        tile_offset += next_batch_shift;
    }
}
