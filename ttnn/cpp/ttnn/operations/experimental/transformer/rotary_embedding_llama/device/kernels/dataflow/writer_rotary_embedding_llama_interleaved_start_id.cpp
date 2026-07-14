// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE void zero_tile_at(uint32_t l1_write_addr, uint32_t tile_bytes) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = 0;
    }
}

void kernel_main() {
    Noc noc;

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

    CircularBuffer cb_out(cb_id_out);
    CircularBuffer cb_zero(cb_id_zero);

    cb_zero.reserve_back(Wt);
    uint32_t zero_l1_write_addr = cb_zero.get_write_ptr();
    for (uint32_t j = 0; j < Wt; j++) {
        zero_tile_at(zero_l1_write_addr, zero_tile_bytes);
        zero_l1_write_addr += zero_tile_bytes;
    }
    cb_zero.push_back(Wt);
    cb_zero.wait_front(Wt);

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                uint32_t output_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                const bool write_rotary_output = seq_tile < rotary_Ht;
                if (write_rotary_output) {
                    cb_out.wait_front(Wt);
                }

                uint32_t l1_read_addr = write_rotary_output ? cb_out.get_read_ptr() : cb_zero.get_read_ptr();
                const uint32_t l1_read_stride = write_rotary_output ? tile_bytes : zero_tile_bytes;
                const uint32_t write_bytes = write_rotary_output ? tile_bytes : zero_tile_bytes;
                for (uint32_t j = 0; j < Wt; j++) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l1_read_addr), s, write_bytes, {}, {.page_id = output_curr_idx});
                    l1_read_addr += l1_read_stride;
                    output_curr_idx++;
                }
                noc.async_write_barrier();

                if (write_rotary_output) {
                    cb_out.pop_front(Wt);
                }
            }
        }
    }

    cb_zero.pop_front(Wt);
}
