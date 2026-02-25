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

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // X
    constexpr uint32_t cb_id_w1 = tt::CB::c_in1;  // W1
    constexpr uint32_t cb_id_w3 = tt::CB::c_in2;  // W3
    constexpr uint32_t cb_id_w2 = tt::CB::c_in3;  // W2
    constexpr uint32_t cb_id_idx = tt::CB::c_in4; // TopK Indices (unused in compute, reader handles routing)
    constexpr uint32_t cb_id_wt = tt::CB::c_in5;  // TopK Weights
    
    // Intermediates
    constexpr uint32_t cb_id_interm0 = tt::CB::c_intermed0;
    constexpr uint32_t cb_id_interm1 = tt::CB::c_intermed1;
    
    constexpr uint32_t cb_id_out = tt::CB::c_out0; // Output

    mm_init(cb_id_in0, cb_id_w1, cb_id_interm0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for inputs from reader
        cb_wait_front(cb_id_in0, 1);
        cb_wait_front(cb_id_w1, 1);
        cb_wait_front(cb_id_w3, 1);
        cb_wait_front(cb_id_w2, 1);
        cb_wait_front(cb_id_idx, 1);
        cb_wait_front(cb_id_wt, 1);
        
        // --------------------------------------------------------------------
        // Step 1: Gate = SiLU(X * W1)
        // --------------------------------------------------------------------
        cb_reserve_back(cb_id_interm0, 1);
        tile_regs_acquire();
        
        matmul_tiles(cb_id_in0, cb_id_w1, 0, 0, 0, false);
        
        // Apply SiLU
        copy_tile_to_dst_init_short();
        silu_tile_init();
        silu_tile(0);
        
        pack_tile(0, cb_id_interm0);
        tile_regs_commit();
        tile_regs_wait();
        cb_push_back(cb_id_interm0, 1);
        tile_regs_release();
        
        // --------------------------------------------------------------------
        // Step 2: Up = X * W3
        // --------------------------------------------------------------------
        cb_reserve_back(cb_id_interm1, 1);
        tile_regs_acquire();
        
        mm_init_short(cb_id_in0, cb_id_w3);
        matmul_tiles(cb_id_in0, cb_id_w3, 0, 0, 0, false);
        
        pack_tile(0, cb_id_interm1);
        tile_regs_commit();
        tile_regs_wait();
        cb_push_back(cb_id_interm1, 1);
        tile_regs_release();

        // --------------------------------------------------------------------
        // Step 3: FF = Gate * Up
        // --------------------------------------------------------------------
        cb_wait_front(cb_id_interm0, 1);
        cb_wait_front(cb_id_interm1, 1);
        
        // We can reuse interm0 for the FF output
        cb_reserve_back(cb_id_interm0, 1);
        tile_regs_acquire();
        
        mul_tiles_init(cb_id_interm0, cb_id_interm1);
        mul_tiles(cb_id_interm0, cb_id_interm1, 0, 0, 0);
        
        pack_tile(0, cb_id_interm0);
        tile_regs_commit();
        tile_regs_wait();
        cb_push_back(cb_id_interm0, 1);
        tile_regs_release();
        
        cb_pop_front(cb_id_interm0, 1); // pop original Gate
        cb_pop_front(cb_id_interm1, 1); // pop original Up

        // --------------------------------------------------------------------
        // Step 4: Out = FF * W2
        // --------------------------------------------------------------------
        cb_wait_front(cb_id_interm0, 1); // FF is now at front of interm0
        
        cb_reserve_back(cb_id_interm1, 1);
        tile_regs_acquire();
        
        mm_init_short(cb_id_interm0, cb_id_w2);
        matmul_tiles(cb_id_interm0, cb_id_w2, 0, 0, 0, false);
        
        pack_tile(0, cb_id_interm1);
        tile_regs_commit();
        tile_regs_wait();
        cb_push_back(cb_id_interm1, 1);
        tile_regs_release();
        
        cb_pop_front(cb_id_interm0, 1); // pop FF
        
        // --------------------------------------------------------------------
        // Step 5: Final = Out * topk_weight
        // --------------------------------------------------------------------
        cb_wait_front(cb_id_interm1, 1);
        
        cb_reserve_back(cb_id_out, 1);
        tile_regs_acquire();
        
        mul_tiles_init(cb_id_interm1, cb_id_wt);
        // Note: topk_weight is a scalar tile, we need a scalar broadcast multiply here if it's packed that way,
        // but assuming it's a replicated tile for now.
        mul_tiles(cb_id_interm1, cb_id_wt, 0, 0, 0);
        
        pack_tile(0, cb_id_out);
        tile_regs_commit();
        tile_regs_wait();
        cb_push_back(cb_id_out, 1);
        tile_regs_release();
        
        cb_pop_front(cb_id_interm1, 1); // pop Out

        // --------------------------------------------------------------------
        // Pop inputs for next token
        // --------------------------------------------------------------------
        cb_pop_front(cb_id_in0, 1);
        cb_pop_front(cb_id_w1, 1);
        cb_pop_front(cb_id_w3, 1);
        cb_pop_front(cb_id_w2, 1);
        cb_pop_front(cb_id_idx, 1);
        cb_pop_front(cb_id_wt, 1);
        
        // Reset mm_init for the next token's Gate step
        mm_init_short(cb_id_in0, cb_id_w1);
    }
}
}