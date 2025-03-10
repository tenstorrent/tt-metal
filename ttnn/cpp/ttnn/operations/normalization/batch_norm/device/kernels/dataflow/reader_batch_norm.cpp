// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    const auto eps = get_arg_val<uint32_t>(0);
    uint32_t src_addr = get_arg_val<uint32_t>(1);  // input tensor
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t HtWt = get_arg_val<uint32_t>(4);
    uint32_t n_stride = get_arg_val<uint32_t>(5);
    uint32_t c_stride = get_arg_val<uint32_t>(6);
    uint32_t N = get_arg_val<uint32_t>(7);
    uint32_t C = get_arg_val<uint32_t>(8);
    uint32_t n_stride_stat = get_arg_val<uint32_t>(9);
    uint32_t c_stride_stat = get_arg_val<uint32_t>(10);
    uint32_t batch_var_addr = get_arg_val<uint32_t>(11);   // batch_var
    uint32_t weight_addr = get_arg_val<uint32_t>(12);      // weight
    uint32_t bias_addr = get_arg_val<uint32_t>(13);        // bias
    uint32_t batch_mean_addr = get_arg_val<uint32_t>(14);  // batch_mean

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr auto cb_id_src = get_compile_time_arg_val(1);
    constexpr uint32_t onetile = 1;

    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const DataFormat src_data_format = get_dataformat(cb_id_src);
    const InterleavedAddrGenFast<src_is_dram> src = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    uint32_t tiles_per_batch = HtWt * C;
    uint32_t start_n = start_tile_id / tiles_per_batch;
    uint32_t start_remaining = start_tile_id % tiles_per_batch;
    uint32_t start_c = start_remaining / HtWt;
    uint32_t start_t = start_remaining % HtWt;

    constexpr auto cb_id_eps = get_compile_time_arg_val(2);

    union {
        float f;
        uint32_t u;
    } scalar;
    scalar.u = eps;
    cb_reserve_back(cb_id_eps, onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    FILL_WITH_VALUE_FLOAT(cb_id_eps, scalar.f);
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);
#endif
    cb_push_back(cb_id_eps, onetile);

    // Input tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;

    // Inputs stats offset
    uint32_t tile_offset_stat = start_n * n_stride_stat + start_c * c_stride_stat;
    uint32_t next_batch_shift_stat = n_stride_stat - c_stride_stat * C;

    uint32_t next_channel_shift = c_stride - HtWt;
    uint32_t next_batch_shift = n_stride - c_stride * C;

    // batch_mean
    constexpr auto cb_id_batch_mean = get_compile_time_arg_val(12);
    constexpr bool batch_mean_is_dram = get_compile_time_arg_val(11) == 1;
    const uint32_t batch_mean_tile_bytes = get_tile_size(cb_id_batch_mean);
    const DataFormat batch_mean_data_format = get_dataformat(cb_id_batch_mean);

    const InterleavedAddrGenFast<batch_mean_is_dram> batch_mean = {
        .bank_base_address = batch_mean_addr,
        .page_size = batch_mean_tile_bytes,
        .data_format = batch_mean_data_format};

    // batch_var
    constexpr auto cb_id_batch_var = get_compile_time_arg_val(4);
    constexpr bool batch_var_is_dram = get_compile_time_arg_val(3) == 1;
    const uint32_t batch_var_tile_bytes = get_tile_size(cb_id_batch_var);
    const DataFormat batch_var_data_format = get_dataformat(cb_id_batch_var);

    const InterleavedAddrGenFast<batch_var_is_dram> batch_var = {
        .bank_base_address = batch_var_addr, .page_size = batch_var_tile_bytes, .data_format = batch_var_data_format};

    // weight
    constexpr auto cb_id_weight = get_compile_time_arg_val(6);
    constexpr bool weight_is_dram = get_compile_time_arg_val(5) == 1;
    const uint32_t weight_tile_bytes = get_tile_size(cb_id_weight);
    const DataFormat weight_data_format = get_dataformat(cb_id_weight);

    const InterleavedAddrGenFast<weight_is_dram> weight = {
        .bank_base_address = weight_addr, .page_size = weight_tile_bytes, .data_format = weight_data_format};

    constexpr bool weight_has_value = get_compile_time_arg_val(7) == 1;

    // bias
    constexpr auto cb_id_bias = get_compile_time_arg_val(8);
    constexpr bool bias_is_dram = get_compile_time_arg_val(9) == 1;
    const uint32_t bias_tile_bytes = get_tile_size(cb_id_bias);
    const DataFormat bias_data_format = get_dataformat(cb_id_bias);

    const InterleavedAddrGenFast<bias_is_dram> bias = {
        .bank_base_address = bias_addr, .page_size = bias_tile_bytes, .data_format = bias_data_format};

    constexpr bool bias_has_value = get_compile_time_arg_val(10) == 1;

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_t = 0) {
            // read a tile from batch_mean, batch variance
            cb_reserve_back(cb_id_batch_mean, onetile);
            cb_reserve_back(cb_id_batch_var, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_batch_mean);
            uint32_t l1_batch_var_write_addr = get_write_ptr(cb_id_batch_var);
            noc_async_read_tile(tile_offset_stat, batch_mean, l1_write_addr);
            noc_async_read_tile(tile_offset_stat, batch_var, l1_batch_var_write_addr);
            noc_async_read_barrier();
            FILL_TILE_WITH_FIRST_ELEMENT(cb_id_batch_mean);
            FILL_TILE_WITH_FIRST_ELEMENT(cb_id_batch_var);
            cb_push_back(cb_id_batch_mean, onetile);
            cb_push_back(cb_id_batch_var, onetile);

            if constexpr (weight_has_value) {  // read a tile from weight tensor
                cb_reserve_back(cb_id_weight, onetile);
                uint32_t l1_weight_write_addr = get_write_ptr(cb_id_weight);
                noc_async_read_tile(tile_offset_stat, weight, l1_weight_write_addr);
                noc_async_read_barrier();
                FILL_TILE_WITH_FIRST_ELEMENT(cb_id_weight);
                cb_push_back(cb_id_weight, onetile);
            }

            if constexpr (bias_has_value) {  // read a tile from bias tensor
                cb_reserve_back(cb_id_bias, onetile);
                uint32_t l1_bias_write_addr = get_write_ptr(cb_id_bias);
                noc_async_read_tile(tile_offset_stat, bias, l1_bias_write_addr);
                noc_async_read_barrier();
                FILL_TILE_WITH_FIRST_ELEMENT(cb_id_bias);
                cb_push_back(cb_id_bias, onetile);
            }

            for (uint32_t t = start_t; t < HtWt && num_tiles_read < num_tiles; ++t, ++num_tiles_read, ++tile_offset) {
                cb_reserve_back(cb_id_src, onetile);
                uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
                noc_async_read_tile(tile_offset, src, l1_write_addr_src);
                noc_async_read_barrier();
                cb_push_back(cb_id_src, onetile);
            }
            tile_offset += next_channel_shift;
            tile_offset_stat += c_stride_stat;
        }
        tile_offset += next_batch_shift;
        tile_offset_stat += next_batch_shift_stat;
    }
}
