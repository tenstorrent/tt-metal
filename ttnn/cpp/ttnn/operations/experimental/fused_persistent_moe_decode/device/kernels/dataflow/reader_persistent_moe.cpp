// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr    = get_arg_val<uint32_t>(0);
    uint32_t topk_idx_addr = get_arg_val<uint32_t>(1);
    uint32_t topk_wt_addr  = get_arg_val<uint32_t>(2);
    uint32_t w1_addr     = get_arg_val<uint32_t>(3);
    uint32_t w3_addr     = get_arg_val<uint32_t>(4);
    uint32_t w2_addr     = get_arg_val<uint32_t>(5);
    uint32_t num_tiles   = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_w1 = tt::CB::c_in1;
    constexpr uint32_t cb_id_w3 = tt::CB::c_in2;
    constexpr uint32_t cb_id_w2 = tt::CB::c_in3;
    constexpr uint32_t cb_id_idx = tt::CB::c_in4;
    constexpr uint32_t cb_id_wt = tt::CB::c_in5;

    constexpr bool is_dram = true;
    uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    uint32_t w1_tile_bytes = get_tile_size(cb_id_w1);
    uint32_t w3_tile_bytes = get_tile_size(cb_id_w3);
    uint32_t w2_tile_bytes = get_tile_size(cb_id_w2);
    uint32_t idx_tile_bytes = get_tile_size(cb_id_idx);
    uint32_t wt_tile_bytes = get_tile_size(cb_id_wt);

    const InterleavedAddrGenFast<is_dram> s0 = { .bank_base_address = src_addr, .page_size = in0_tile_bytes, .data_format = get_dataformat(cb_id_in0) };
    const InterleavedAddrGenFast<is_dram> s_idx = { .bank_base_address = topk_idx_addr, .page_size = idx_tile_bytes, .data_format = get_dataformat(cb_id_idx) };
    const InterleavedAddrGenFast<is_dram> s_wt = { .bank_base_address = topk_wt_addr, .page_size = wt_tile_bytes, .data_format = get_dataformat(cb_id_wt) };
    const InterleavedAddrGenFast<is_dram> s_w1 = { .bank_base_address = w1_addr, .page_size = w1_tile_bytes, .data_format = get_dataformat(cb_id_w1) };
    const InterleavedAddrGenFast<is_dram> s_w3 = { .bank_base_address = w3_addr, .page_size = w3_tile_bytes, .data_format = get_dataformat(cb_id_w3) };
    const InterleavedAddrGenFast<is_dram> s_w2 = { .bank_base_address = w2_addr, .page_size = w2_tile_bytes, .data_format = get_dataformat(cb_id_w2) };

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Read input
        cb_reserve_back(cb_id_in0, 1);
        noc_async_read_tile(i, s0, get_write_ptr(cb_id_in0));
        
        // Read topk
        cb_reserve_back(cb_id_idx, 1);
        noc_async_read_tile(i, s_idx, get_write_ptr(cb_id_idx));

        cb_reserve_back(cb_id_wt, 1);
        noc_async_read_tile(i, s_wt, get_write_ptr(cb_id_wt));

        // Read weights (dummy implementation, fetches 1 tile per weight tensor)
        cb_reserve_back(cb_id_w1, 1);
        noc_async_read_tile(i % 256, s_w1, get_write_ptr(cb_id_w1));

        cb_reserve_back(cb_id_w3, 1);
        noc_async_read_tile(i % 256, s_w3, get_write_ptr(cb_id_w3));

        cb_reserve_back(cb_id_w2, 1);
        noc_async_read_tile(i % 256, s_w2, get_write_ptr(cb_id_w2));

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_idx, 1);
        cb_push_back(cb_id_wt, 1);
        cb_push_back(cb_id_w1, 1);
        cb_push_back(cb_id_w3, 1);
        cb_push_back(cb_id_w2, 1);
    }
}
