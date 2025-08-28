// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t cos_addr = get_arg_val<uint32_t>(0);
    uint32_t sin_addr = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);
    uint32_t start_row_id = get_arg_val<uint32_t>(3);
    uint32_t cos_sin_start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_input_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(4);
    constexpr uint16_t scalar_value = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t HtWt = get_compile_time_arg_val(8);
    constexpr uint32_t half_Wt_size = get_compile_time_arg_val(9);
    constexpr auto cos_args = TensorAccessorArgs<10>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;

    cb_reserve_back(input_cb_id, num_rows * Wt);
    cb_push_back(input_cb_id, num_rows * Wt);
    uint64_t input_l1_read_addr = get_noc_addr(get_read_ptr(input_cb_id));

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

    // Fill tile with zeros
    const uint32_t scalar_tile_bytes = get_tile_size(scalar_cb_id);
    cb_reserve_back(scalar_cb_id, onetile);
    uint32_t l1_zeros_addr_in_scalar = get_write_ptr(scalar_cb_id);
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    cb_push_back(scalar_cb_id, onetile);

    uint32_t cos_sin_curr_id = cos_sin_start_id;

#ifdef DECODE_MODE
    cb_reserve_back(sin_cb_id, Wt);
    cb_reserve_back(cos_cb_id, Wt);
    uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
    uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
    for (uint32_t i = 0; i < Wt; i++) {
        noc_async_read_tile(cos_sin_curr_id, s2, sin_l1_write_addr);
        noc_async_read_tile(cos_sin_curr_id, s1, cos_l1_write_addr);
        cos_sin_curr_id++;
        sin_l1_write_addr += sin_tile_bytes;
        cos_l1_write_addr += cos_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(sin_cb_id, Wt);
    cb_push_back(cos_cb_id, Wt);
#else
    uint32_t ht = start_row_id;
#endif

    uint32_t Wt_size = half_Wt_size + half_Wt_size;
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_reserve_back(rotated_input_cb_id, Wt);
        uint32_t rotated_input_l1_write_addr = get_write_ptr(rotated_input_cb_id);
        noc_async_read(input_l1_read_addr + half_Wt_size, rotated_input_l1_write_addr, half_Wt_size);
        noc_async_read(input_l1_read_addr, rotated_input_l1_write_addr + half_Wt_size, half_Wt_size);
        input_l1_read_addr += Wt_size;
        noc_async_read_barrier();
        cb_push_back(rotated_input_cb_id, Wt);

#ifndef DECODE_MODE
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_reserve_back(sin_cb_id, onetile);
            uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
            noc_async_read_tile(cos_sin_curr_id, s2, sin_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(sin_cb_id, onetile);

            cb_reserve_back(cos_cb_id, onetile);
            uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
            noc_async_read_tile(cos_sin_curr_id, s1, cos_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cos_cb_id, onetile);
            cos_sin_curr_id++;
        }
        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
#endif
    }
}
