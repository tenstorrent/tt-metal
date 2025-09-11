// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
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
    constexpr auto input_args = TensorAccessorArgs<8>();
    constexpr auto cos_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    const uint32_t my_seq_tiles = seq_t_end - seq_t_start;
    const uint32_t my_cos_sin_tiles = my_seq_tiles * Wt;

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const auto s0 = TensorAccessor(input_args, src_addr, input_tile_bytes);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const auto s1 = TensorAccessor(cos_args, cos_addr, cos_tile_bytes);

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto s2 = TensorAccessor(sin_args, sin_addr, sin_tile_bytes);

    const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);
    const auto s3 = TensorAccessor(trans_mat_args, trans_mat_addr, trans_mat_tile_bytes);

    uint32_t trans_mat_curr_idx = 0;

    // Read transformation matrix in CB (only once, because it will be reused)
    cb_reserve_back(trans_mat_cb_id, onetile);
    uint32_t trans_mat_l1_write_addr = get_write_ptr(trans_mat_cb_id);
    noc_async_read_tile(trans_mat_curr_idx, s3, trans_mat_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(trans_mat_cb_id, onetile);

    /*
        Read a ublock of tiles from src to CB, and then push the ublock to unpacker

        For example:
            num_rows_per_core = 1 * 8 * 128 * 128 // 128 // 32 = 32
            Ht = 4
            Wt = 4
    */

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        cb_reserve_back(sin_cb_id, my_cos_sin_tiles);
        cb_reserve_back(cos_cb_id, my_cos_sin_tiles);
        uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
        uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
#endif

        // To make sure the sin/cos row are read only once
        uint32_t sin_cos_row_cnt = 0;
        bool done_sin_cos = false;

        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
#if RELOAD_IMPL == 1
                cb_reserve_back(sin_cb_id, Wt);
                cb_reserve_back(cos_cb_id, Wt);
                uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
                uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
#endif

                cb_reserve_back(input_cb_id, Wt);
                uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
                uint32_t input_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                uint32_t cos_sin_curr_idx;
                if constexpr (freq_per_head) {
                    cos_sin_curr_idx = head_num * Ht * Wt + seq_tile * Wt;
                } else {
                    cos_sin_curr_idx = seq_tile * Wt;
                }
                for (uint32_t j = 0; j < Wt; ++j) {
                    // Read input into CB
                    noc_async_read_tile(input_curr_idx, s0, input_l1_write_addr);
                    input_curr_idx++;
                    input_l1_write_addr += input_tile_bytes;

                    if (!done_sin_cos) {
                        // Read sin into CB
                        noc_async_read_tile(cos_sin_curr_idx, s2, sin_l1_write_addr);
                        sin_l1_write_addr += sin_tile_bytes;

                        // Read cos into CB
                        noc_async_read_tile(cos_sin_curr_idx, s1, cos_l1_write_addr);
                        cos_l1_write_addr += cos_tile_bytes;

                        cos_sin_curr_idx++;
                    }
                }

                noc_async_read_barrier();
                cb_push_back(input_cb_id, Wt);
#if RELOAD_IMPL == 1
                cb_push_back(sin_cb_id, Wt);
                cb_push_back(cos_cb_id, Wt);
#else

                if (!done_sin_cos) {
                    cb_push_back(sin_cb_id, Wt);
                    cb_push_back(cos_cb_id, Wt);

                    // Update sin_cos_row_cnt
                    sin_cos_row_cnt++;

                    if (sin_cos_row_cnt == my_seq_tiles) {
                        done_sin_cos = true;
                    }
                }
#endif
            }
        }
    }
}
