// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    uint32_t head_size                     = get_arg_val<uint32_t>(0);
    uint32_t num_q_heads                   = get_arg_val<uint32_t>(1);
    uint32_t num_q_heads_per_core          = get_arg_val<uint32_t>(2);
    uint32_t remote_q_head_start_idx       = get_arg_val<uint32_t>(3);
    uint32_t start_q_x                     = get_arg_val<uint32_t>(4);
    uint32_t start_q_y                     = get_arg_val<uint32_t>(5);
    uint32_t q_base_addr                   = get_arg_val<uint32_t>(6);
    uint32_t q_start_addr                  = get_arg_val<uint32_t>(7);
    uint32_t q_offset                      = get_arg_val<uint32_t>(8);
    constexpr uint32_t cb_id_q_out         = get_compile_time_arg_val(0);

    bool read_kv_heads                     = get_arg_val<uint32_t>(9);

    uint32_t num_x                         = get_arg_val<uint32_t>(18);
    tt_l1_ptr uint32_t * in0_mcast_noc_x          = (tt_l1_ptr uint32_t*)(get_arg_addr(19));
    tt_l1_ptr uint32_t * in0_mcast_noc_y          = (tt_l1_ptr uint32_t*)(get_arg_addr(19 + num_x));

    uint32_t q_x = start_q_x;
    uint32_t q_y = start_q_y;
    uint32_t remote_q_head_idx = remote_q_head_start_idx;
    uint64_t q_read_addr = get_noc_addr(in0_mcast_noc_x[q_x], in0_mcast_noc_y[q_y], q_start_addr);
    uint32_t q_write_addr = get_write_ptr(cb_id_q_out) + q_offset;

    for (uint32_t q = 0; q < num_q_heads; ++q) {
        // Q
        noc_async_read(q_read_addr, q_write_addr, head_size);
        q_read_addr += head_size;
        q_write_addr += head_size;
        remote_q_head_idx++;
        if (remote_q_head_idx == num_q_heads_per_core) {
            remote_q_head_idx = 0;
            q_x++;
            if (q_x == num_x) {
                q_x = 0;
                q_y++;
                q_read_addr = get_noc_addr(in0_mcast_noc_x[q_x], in0_mcast_noc_y[q_y], q_base_addr);
            }
        }
        noc_async_read_barrier();
    }

    if (read_kv_heads) {
        uint32_t num_kv_heads                  = get_arg_val<uint32_t>(10);
        uint32_t num_kv_heads_per_core         = get_arg_val<uint32_t>(11);
        uint32_t remote_kv_head_start_idx      = get_arg_val<uint32_t>(12);
        uint32_t start_kv_x                    = get_arg_val<uint32_t>(13);
        uint32_t start_kv_y                    = get_arg_val<uint32_t>(14);
        uint32_t kv_base_addr                  = get_arg_val<uint32_t>(15);
        uint32_t kv_start_addr                 = get_arg_val<uint32_t>(16);
        uint32_t num_kv_tiles                  = get_arg_val<uint32_t>(17);
        constexpr uint32_t cb_id_kv_out        = get_compile_time_arg_val(1);

        uint32_t kv_x = start_kv_x;
        uint32_t kv_y = start_kv_y;
        uint32_t remote_kv_head_idx = remote_kv_head_start_idx;
        uint64_t kv_read_addr = get_noc_addr(in0_mcast_noc_x[kv_x], in0_mcast_noc_y[kv_y], kv_start_addr);
        cb_reserve_back(cb_id_kv_out, num_kv_tiles);
        uint32_t kv_write_addr = get_write_ptr(cb_id_kv_out);

        // K or V
        for (uint32_t kv = 0; kv < num_kv_heads; ++kv) {
            noc_async_read(kv_read_addr, kv_write_addr, head_size);
            kv_read_addr += head_size;
            kv_write_addr += head_size;
            remote_kv_head_idx++;
            if (remote_kv_head_idx == num_kv_heads_per_core) {
                remote_kv_head_idx = 0;
                kv_x++;
                if (kv_x == num_x) {
                    kv_x = 0;
                    kv_y++;
                }
                kv_read_addr = get_noc_addr(in0_mcast_noc_x[kv_x], in0_mcast_noc_y[kv_y], kv_base_addr);
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_id_kv_out, num_kv_tiles);
    }
}
