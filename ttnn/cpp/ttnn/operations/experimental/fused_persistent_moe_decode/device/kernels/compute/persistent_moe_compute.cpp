// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/sfpu_compute_api.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t k = get_arg_val<uint32_t>(1);
    uint32_t w1_expert_tiles = get_arg_val<uint32_t>(2); 
    uint32_t w3_expert_tiles = get_arg_val<uint32_t>(3);
    uint32_t w2_expert_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // X
    constexpr uint32_t cb_id_w1 = tt::CB::c_in1;  // W1
    constexpr uint32_t cb_id_w3 = tt::CB::c_in2;  // W3
    constexpr uint32_t cb_id_w2 = tt::CB::c_in3;  // W2
    constexpr uint32_t cb_id_idx = tt::CB::c_in4; // TopK Indices
    constexpr uint32_t cb_id_wt = tt::CB::c_in5;  // TopK Weights
    
    constexpr uint32_t cb_id_out = tt::CB::c_out0; // Output

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for inputs from reader (the input tokens and routing information)
        cb_wait_front(cb_id_in0, 1);
        cb_wait_front(cb_id_idx, 1);
        cb_wait_front(cb_id_wt, 1);
        
        cb_reserve_back(cb_id_out, 1);
        
        for (uint32_t j = 0; j < k; j++) {
            // W1 chunks
            uint32_t w1_rem = w1_expert_tiles;
            while (w1_rem > 0) {
                uint32_t chunk = w1_rem > 256 ? 256 : w1_rem;
                cb_wait_front(cb_id_w1, chunk);
                // Math goes here
                cb_pop_front(cb_id_w1, chunk);
                w1_rem -= chunk;
            }

            // W3 chunks
            uint32_t w3_rem = w3_expert_tiles;
            while (w3_rem > 0) {
                uint32_t chunk = w3_rem > 256 ? 256 : w3_rem;
                cb_wait_front(cb_id_w3, chunk);
                // Math goes here
                cb_pop_front(cb_id_w3, chunk);
                w3_rem -= chunk;
            }

            // W2 chunks
            uint32_t w2_rem = w2_expert_tiles;
            while (w2_rem > 0) {
                uint32_t chunk = w2_rem > 256 ? 256 : w2_rem;
                cb_wait_front(cb_id_w2, chunk);
                // Math goes here
                cb_pop_front(cb_id_w2, chunk);
                w2_rem -= chunk;
            }
        }

        // Just output zeros to bypass math for the boundary test
        acquire_dst();
        copy_tile_to_dst_init_short();
        copy_tile(cb_id_in0, 0, 0); // copy in0 tile 0 to dst 0 to emit something
        pack_tile(0, cb_id_out);
        release_dst();

        cb_push_back(cb_id_out, 1);

        // Pop inputs for next token
        cb_pop_front(cb_id_in0, 1);
        cb_pop_front(cb_id_idx, 1);
        cb_pop_front(cb_id_wt, 1);
    }
}
}