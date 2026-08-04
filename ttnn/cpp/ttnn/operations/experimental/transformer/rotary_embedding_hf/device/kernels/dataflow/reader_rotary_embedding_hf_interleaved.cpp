// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t cos_addr = get_arg_val<uint32_t>(1);
    uint32_t sin_addr = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t start_row_id = get_arg_val<uint32_t>(5);
    uint32_t cos_sin_start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_input_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(4);
    constexpr uint16_t scalar_value = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t HtWt = get_compile_time_arg_val(8);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(9);
    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr auto cos_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const auto s0 = TensorAccessor(src_args, src_addr, input_tile_bytes);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_rotated_input(rotated_input_cb_id);
    CircularBuffer cb_cos(cos_cb_id);
    CircularBuffer cb_sin(sin_cb_id);
    CircularBuffer cb_scalar(scalar_cb_id);

    // Fill tile with scalar value (-1)
    const uint32_t scalar_tile_bytes = get_tile_size(scalar_cb_id);
    cb_scalar.reserve_back(onetile);
    uint32_t l1_zeros_addr_in_scalar = cb_scalar.get_write_ptr();
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    cb_scalar.push_back(onetile);

    uint32_t input_curr_id = start_id;
    uint32_t rotated_input_curr_id = start_id + half_Wt;
    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_rotated_input.reserve_back(onetile);
            uint32_t rotated_input_l1_write_addr = cb_rotated_input.get_write_ptr();
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(rotated_input_l1_write_addr),
                input_tile_bytes,
                {.page_id = rotated_input_curr_id},
                {});
            noc.async_read_barrier();
            cb_rotated_input.push_back(onetile);
            rotated_input_curr_id++;

            cb_sin.reserve_back(onetile);
            uint32_t sin_l1_write_addr = cb_sin.get_write_ptr();
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            cb_sin.push_back(onetile);

            cb_input.reserve_back(onetile);
            uint32_t input_l1_write_addr = cb_input.get_write_ptr();
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(input_l1_write_addr), input_tile_bytes, {.page_id = input_curr_id}, {});
            noc.async_read_barrier();
            cb_input.push_back(onetile);
            input_curr_id++;

            cb_cos.reserve_back(onetile);
            uint32_t cos_l1_write_addr = cb_cos.get_write_ptr();
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            cb_cos.push_back(onetile);
            cos_sin_curr_id++;

            if (j == half_Wt - 1) {
                rotated_input_curr_id -= Wt;
            }
        }
        rotated_input_curr_id += Wt;
        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }
}
