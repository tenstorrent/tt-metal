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
    uint32_t k           = get_arg_val<uint32_t>(7);
    uint32_t w1_expert_tiles = get_arg_val<uint32_t>(8);
    uint32_t w3_expert_tiles = get_arg_val<uint32_t>(9);
    uint32_t w2_expert_tiles = get_arg_val<uint32_t>(10);
    uint32_t idx_num_tiles   = get_arg_val<uint32_t>(11);

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
        noc_async_read_tile(i % idx_num_tiles, s_idx, get_write_ptr(cb_id_idx));

        cb_reserve_back(cb_id_wt, 1);
        noc_async_read_tile(i % idx_num_tiles, s_wt, get_write_ptr(cb_id_wt));

        noc_async_read_barrier();
        
        // Save the pointer to the indices before we push and advance the write pointer!
        volatile tt_l1_ptr uint16_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_idx));

        // Push these immediately so compute can start if it wants
        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_idx, 1);
        cb_push_back(cb_id_wt, 1);

        for (uint32_t j = 0; j < k; j++) {
            uint16_t expert_idx = idx_ptr[8 + j]; // +8 to skip the 16-byte TileHeader

            // W1
            uint32_t w1_rem = w1_expert_tiles;
            uint32_t w1_off = 0;
            while (w1_rem > 0) {
                uint32_t chunk = w1_rem > 256 ? 256 : w1_rem;
                cb_reserve_back(cb_id_w1, chunk);
                uint32_t l1 = get_write_ptr(cb_id_w1);
                for (uint32_t t = 0; t < chunk; t++) {
                    noc_async_read_tile(expert_idx * w1_expert_tiles + w1_off + t, s_w1, l1);
                    l1 += w1_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_w1, chunk);
                w1_rem -= chunk;
                w1_off += chunk;
            }

            // W3
            uint32_t w3_rem = w3_expert_tiles;
            uint32_t w3_off = 0;
            while (w3_rem > 0) {
                uint32_t chunk = w3_rem > 256 ? 256 : w3_rem;
                cb_reserve_back(cb_id_w3, chunk);
                uint32_t l1 = get_write_ptr(cb_id_w3);
                for (uint32_t t = 0; t < chunk; t++) {
                    noc_async_read_tile(expert_idx * w3_expert_tiles + w3_off + t, s_w3, l1);
                    l1 += w3_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_w3, chunk);
                w3_rem -= chunk;
                w3_off += chunk;
            }

            // W2
            uint32_t w2_rem = w2_expert_tiles;
            uint32_t w2_off = 0;
            while (w2_rem > 0) {
                uint32_t chunk = w2_rem > 256 ? 256 : w2_rem;
                cb_reserve_back(cb_id_w2, chunk);
                uint32_t l1 = get_write_ptr(cb_id_w2);
                for (uint32_t t = 0; t < chunk; t++) {
                    noc_async_read_tile(expert_idx * w2_expert_tiles + w2_off + t, s_w2, l1);
                    l1 += w2_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_w2, chunk);
                w2_rem -= chunk;
                w2_off += chunk;
            }
        }
    }
}
