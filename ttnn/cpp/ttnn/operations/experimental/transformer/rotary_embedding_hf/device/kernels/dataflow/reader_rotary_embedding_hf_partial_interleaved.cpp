// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Partial-rotary variant of reader_rotary_embedding_hf_interleaved.cpp (cos/sin narrower than the input, prefill,
// interleaved): each input row has ROT_WT_IN tiles; the first Wt (= cos width) tiles are rotated exactly as the full
// kernel does (rotate_half within those Wt tiles), the remaining ROT_WT_IN - Wt tiles are passed through untouched via
// the PASS_CB circular buffer (read in one batch per row) to the writer.

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
    uint32_t start_id = get_arg_val<uint32_t>(4);  // first input tile of this core (row_start * ROT_WT_IN)
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
    constexpr uint32_t Wt_in = ROT_WT_IN;
    constexpr uint32_t Pt = Wt_in - Wt;
    constexpr uint32_t pass_cb_id = PASS_CB;

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
    CircularBuffer cb_pass(pass_cb_id);

    cb_scalar.reserve_back(onetile);
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scalar.get_write_ptr());
    scalar_buffer[0] = scalar_value;
    cb_scalar.push_back(onetile);

    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;
    uint32_t row_base = start_id;

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            const uint32_t rot_j = j < half_Wt ? j + half_Wt : j - half_Wt;
            cb_rotated_input.reserve_back(onetile);
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(cb_rotated_input.get_write_ptr()),
                input_tile_bytes,
                {.page_id = row_base + rot_j},
                {});
            cb_sin.reserve_back(onetile);
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(cb_sin.get_write_ptr()), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            cb_input.reserve_back(onetile);
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(cb_input.get_write_ptr()), input_tile_bytes, {.page_id = row_base + j}, {});
            cb_cos.reserve_back(onetile);
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cb_cos.get_write_ptr()), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
            cb_rotated_input.push_back(onetile);
            cb_sin.push_back(onetile);
            cb_input.push_back(onetile);
            cb_cos.push_back(onetile);
            cos_sin_curr_id++;
        }
        // pass-through tiles of this row, one batch
        cb_pass.reserve_back(Pt);
        uint32_t pass_addr = cb_pass.get_write_ptr();
        for (uint32_t p = 0; p < Pt; ++p) {
            noc.async_read(s0, CoreLocalMem<uint32_t>(pass_addr), input_tile_bytes, {.page_id = row_base + Wt + p}, {});
            pass_addr += input_tile_bytes;
        }
        noc.async_read_barrier();
        cb_pass.push_back(Pt);

        row_base += Wt_in;
        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }
}
