// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void zero_tile_at(uint32_t l1_write_addr, uint32_t tile_bytes) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = 0;
    }
}

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_zero = get_compile_time_arg_val(1);
    constexpr uint32_t n_heads = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t rotary_Ht = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const uint32_t zero_tile_bytes = get_tile_size(cb_id_zero);
    const auto s = TensorAccessor(dst_args, dst_addr);

    cb_reserve_back(cb_id_zero, Wt);
    uint32_t zero_l1_write_addr = get_write_ptr(cb_id_zero);
    for (uint32_t j = 0; j < Wt; j++) {
        zero_tile_at(zero_l1_write_addr, zero_tile_bytes);
        zero_l1_write_addr += zero_tile_bytes;
    }
    cb_push_back(cb_id_zero, Wt);
    cb_wait_front(cb_id_zero, Wt);

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                uint32_t output_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                const bool write_rotary_output = seq_tile < rotary_Ht;
                if (write_rotary_output) {
                    cb_wait_front(cb_id_out, Wt);
                }

                uint32_t l1_read_addr = write_rotary_output ? get_read_ptr(cb_id_out) : get_read_ptr(cb_id_zero);
                const uint32_t l1_read_stride = write_rotary_output ? tile_bytes : zero_tile_bytes;
                for (uint32_t j = 0; j < Wt; j++) {
                    noc_async_write_tile(output_curr_idx, s, l1_read_addr);
                    l1_read_addr += l1_read_stride;
                    output_curr_idx++;
                }
                noc_async_write_barrier();

                if (write_rotary_output) {
                    cb_pop_front(cb_id_out, Wt);
                }
            }
        }
    }

    cb_pop_front(cb_id_zero, Wt);
}
