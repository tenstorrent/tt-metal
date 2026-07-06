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

    uint32_t argrt = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t n_heads = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr bool freq_per_head = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t cos_Ht = get_compile_time_arg_val(8);
    constexpr uint32_t sin_Ht = get_compile_time_arg_val(9);
    constexpr uint32_t rotary_Ht = get_compile_time_arg_val(10);
    constexpr auto input_args = TensorAccessorArgs<11>();
    constexpr auto cos_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const auto s0 = TensorAccessor(input_args, src_addr);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr);

    const auto s3 = TensorAccessor(trans_mat_args, trans_mat_addr);

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_cos(cos_cb_id);
    CircularBuffer cb_sin(sin_cb_id);
    CircularBuffer cb_trans_mat(trans_mat_cb_id);

    uint32_t trans_mat_curr_idx = 0;

    // Read transformation matrix in CB (only once, because it will be reused)
    cb_trans_mat.reserve_back(onetile);
    uint32_t trans_mat_l1_write_addr = cb_trans_mat.get_write_ptr();
    noc.async_read(
        s3,
        CoreLocalMem<uint32_t>(trans_mat_l1_write_addr),
        get_tile_size(trans_mat_cb_id),
        {.page_id = trans_mat_curr_idx},
        {});
    noc.async_read_barrier();
    cb_trans_mat.push_back(onetile);

    /*
        Read a ublock of tiles from src to CB, and then push the ublock to unpacker

        For example:
            num_rows_per_core = 1 * 8 * 128 * 128 // 128 // 32 = 32
            Ht = 4
            Wt = 4
    */

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        uint32_t sin_l1_write_addr = 0;
        uint32_t cos_l1_write_addr = 0;
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            cb_sin.reserve_back(my_cos_sin_tiles);
            cb_cos.reserve_back(my_cos_sin_tiles);
            sin_l1_write_addr = cb_sin.get_write_ptr();
            cos_l1_write_addr = cb_cos.get_write_ptr();
        }
#endif

        // To make sure the sin/cos row are read only once
        uint32_t sin_cos_row_cnt = 0;
        bool done_sin_cos = false;

        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
#if RELOAD_IMPL == 1
                cb_sin.reserve_back(Wt);
                cb_cos.reserve_back(Wt);
                uint32_t sin_l1_write_addr = cb_sin.get_write_ptr();
                uint32_t cos_l1_write_addr = cb_cos.get_write_ptr();
#endif

                cb_input.reserve_back(Wt);
                uint32_t input_l1_write_addr = cb_input.get_write_ptr();
                uint32_t input_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                uint32_t cos_curr_idx;
                uint32_t sin_curr_idx;
                if constexpr (freq_per_head) {
                    cos_curr_idx = head_num * cos_Ht * Wt + seq_tile * Wt;
                    sin_curr_idx = head_num * sin_Ht * Wt + seq_tile * Wt;
                } else {
                    cos_curr_idx = seq_tile * Wt;
                    sin_curr_idx = seq_tile * Wt;
                }
                for (uint32_t j = 0; j < Wt; ++j) {
                    // Read input into CB
                    noc.async_read(
                        s0,
                        CoreLocalMem<uint32_t>(input_l1_write_addr),
                        input_tile_bytes,
                        {.page_id = input_curr_idx},
                        {});
                    input_curr_idx++;
                    input_l1_write_addr += input_tile_bytes;

                    if (!done_sin_cos) {
                        noc.async_read(
                            s2,
                            CoreLocalMem<uint32_t>(sin_l1_write_addr),
                            sin_tile_bytes,
                            {.page_id = sin_curr_idx},
                            {});
                        noc.async_read(
                            s1,
                            CoreLocalMem<uint32_t>(cos_l1_write_addr),
                            cos_tile_bytes,
                            {.page_id = cos_curr_idx},
                            {});
                        sin_curr_idx++;
                        cos_curr_idx++;
                        sin_l1_write_addr += sin_tile_bytes;
                        cos_l1_write_addr += cos_tile_bytes;
                    }
                }

                noc.async_read_barrier();
                cb_input.push_back(Wt);
#if RELOAD_IMPL == 1
                cb_sin.push_back(Wt);
                cb_cos.push_back(Wt);
#else

                if (!done_sin_cos) {
                    cb_sin.push_back(Wt);
                    cb_cos.push_back(Wt);

                    // Update sin_cos_row_cnt
                    sin_cos_row_cnt++;

                    if (sin_cos_row_cnt == my_rotary_seq_tiles) {
                        done_sin_cos = true;
                    }
                }
#endif
            }
        }
    }
}
