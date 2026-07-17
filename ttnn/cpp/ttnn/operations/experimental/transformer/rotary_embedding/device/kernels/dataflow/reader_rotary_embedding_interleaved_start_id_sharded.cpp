// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

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

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_rotated_input(rotated_input_cb_id);
    CircularBuffer cb_cos(cos_cb_id);
    CircularBuffer cb_sin(sin_cb_id);
    CircularBuffer cb_scalar(scalar_cb_id);

    cb_input.reserve_back(num_rows * Wt);
    cb_input.push_back(num_rows * Wt);
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address.
    uint64_t input_l1_read_addr = get_noc_addr(cb_input.get_read_ptr());

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr);

    // Fill tile with zeros
    const uint32_t scalar_tile_bytes = get_tile_size(scalar_cb_id);
    cb_scalar.reserve_back(onetile);
    uint32_t l1_zeros_addr_in_scalar = cb_scalar.get_write_ptr();
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    cb_scalar.push_back(onetile);

    uint32_t cos_sin_curr_id = cos_sin_start_id;

#ifdef DECODE_MODE
    cb_sin.reserve_back(Wt);
    cb_cos.reserve_back(Wt);
    uint32_t sin_l1_write_addr = cb_sin.get_write_ptr();
    uint32_t cos_l1_write_addr = cb_cos.get_write_ptr();
    for (uint32_t i = 0; i < Wt; i++) {
        noc.async_read(s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
        noc.async_read(s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
        cos_sin_curr_id++;
        sin_l1_write_addr += sin_tile_bytes;
        cos_l1_write_addr += cos_tile_bytes;
    }
    noc.async_read_barrier();
    cb_sin.push_back(Wt);
    cb_cos.push_back(Wt);
#else
    uint32_t ht = start_row_id;
#endif

    uint32_t Wt_size = half_Wt_size + half_Wt_size;
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_rotated_input.reserve_back(Wt);
        uint32_t rotated_input_l1_write_addr = cb_rotated_input.get_write_ptr();
        // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address.
        noc_async_read(input_l1_read_addr + half_Wt_size, rotated_input_l1_write_addr, half_Wt_size);
        // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address.
        noc_async_read(input_l1_read_addr, rotated_input_l1_write_addr + half_Wt_size, half_Wt_size);
        input_l1_read_addr += Wt_size;
        noc.async_read_barrier();
        cb_rotated_input.push_back(Wt);

#ifndef DECODE_MODE
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_sin.reserve_back(onetile);
            uint32_t sin_l1_write_addr = cb_sin.get_write_ptr();
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            cb_sin.push_back(onetile);

            cb_cos.reserve_back(onetile);
            uint32_t cos_l1_write_addr = cb_cos.get_write_ptr();
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            cb_cos.push_back(onetile);
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
