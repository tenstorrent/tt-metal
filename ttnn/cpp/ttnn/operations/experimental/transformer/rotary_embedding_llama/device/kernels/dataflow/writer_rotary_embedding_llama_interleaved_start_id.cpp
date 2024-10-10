// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {

    uint32_t argrt = 0;
    uint32_t dst_addr  = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t n_heads = get_compile_time_arg_val(2);
    constexpr uint32_t Wt =  get_compile_time_arg_val(3);
    constexpr uint32_t Ht =  get_compile_time_arg_val(4);


    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                uint32_t output_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                cb_wait_front(cb_id_out, Wt);
                // Write a row
                uint32_t l1_read_addr = get_read_ptr(cb_id_out);
                for (uint32_t j = 0; j < Wt; j++) {
                    noc_async_write_tile(output_curr_idx, s, l1_read_addr);
                    l1_read_addr += tile_bytes;
                    output_curr_idx++;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_id_out, Wt);
            }
        }
    }
}
