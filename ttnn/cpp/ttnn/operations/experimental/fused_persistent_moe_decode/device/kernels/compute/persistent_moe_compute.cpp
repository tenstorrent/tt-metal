#include <stdint.h>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t k = get_arg_val<uint32_t>(1);
    uint32_t w1_expert_tiles = get_arg_val<uint32_t>(2); 
    uint32_t w3_expert_tiles = get_arg_val<uint32_t>(3);
    uint32_t w2_expert_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // X [1 x 32]
    constexpr uint32_t cb_id_w1 = tt::CB::c_in1;  // W1 [32 x 16]
    constexpr uint32_t cb_id_w3 = tt::CB::c_in2;  // W3 [32 x 16]
    constexpr uint32_t cb_id_w2 = tt::CB::c_in3;  // W2 [16 x 32]
    constexpr uint32_t cb_id_idx = tt::CB::c_in4; // TopK Indices
    constexpr uint32_t cb_id_wt = tt::CB::c_in5;  // TopK Weights
    
    constexpr uint32_t cb_id_intermed0 = tt::CB::c_intermed0;
    constexpr uint32_t cb_id_intermed1 = tt::CB::c_intermed1;
    
    constexpr uint32_t cb_id_out = tt::CB::c_out0; // Output [1 x 32]

    cb_wait_front(cb_id_in0, 32);
    cb_wait_front(cb_id_idx, 1);
    cb_wait_front(cb_id_wt, 1);
    
    cb_reserve_back(cb_id_out, 32);

    for (uint32_t j = 0; j < k; j++) {
        // W1 chunks -> cb_id_intermed0 [1x16]
        // Full init needed at start of loop (especially after W2 block 1 which outputted to cb_id_out)
        mm_init(cb_id_in0, cb_id_w1, cb_id_intermed0);
        acquire_dst();
        uint32_t w1_rem = w1_expert_tiles; // 512
        uint32_t in_idx = 0;
        while (w1_rem > 0) {
            uint32_t chunk = 32;
            cb_wait_front(cb_id_w1, chunk);
            for(uint32_t r = 0; r < 2; r++) { // chunk is 2 rows of 16
                for(uint32_t c = 0; c < 16; c++) {
                    matmul_tiles(cb_id_in0, cb_id_w1, in_idx, r * 16 + c, c);
                }
                in_idx++;
            }
            cb_pop_front(cb_id_w1, chunk);
            w1_rem -= chunk;
        }
        cb_reserve_back(cb_id_intermed0, 16);
        for(uint32_t c = 0; c < 16; c++) { pack_tile(c, cb_id_intermed0, c); }
        cb_push_back(cb_id_intermed0, 16);
        release_dst();

        // W3 chunks -> cb_id_intermed1 [1x16]
        // We only changed the second operand, so we can use short init, but to be safe since out cb changed, let's use full mm_init
        mm_init(cb_id_in0, cb_id_w3, cb_id_intermed1);
        acquire_dst();
        uint32_t w3_rem = w3_expert_tiles;
        in_idx = 0;
        while (w3_rem > 0) {
            uint32_t chunk = 32;
            cb_wait_front(cb_id_w3, chunk);
            for(uint32_t r = 0; r < 2; r++) {
                for(uint32_t c = 0; c < 16; c++) {
                    matmul_tiles(cb_id_in0, cb_id_w3, in_idx, r * 16 + c, c);
                }
                in_idx++;
            }
            cb_pop_front(cb_id_w3, chunk);
            w3_rem -= chunk;
        }
        cb_reserve_back(cb_id_intermed1, 16);
        for(uint32_t c = 0; c < 16; c++) { pack_tile(c, cb_id_intermed1, c); }
        cb_push_back(cb_id_intermed1, 16);
        release_dst();

        // SiLU + MUL: intermed0 = silu(intermed0) * intermed1
        cb_wait_front(cb_id_intermed0, 16);
        cb_wait_front(cb_id_intermed1, 16);
        mul_tiles_init(cb_id_intermed0, cb_id_intermed1);
        acquire_dst();
        for(uint32_t c = 0; c < 16; c++) {
            mul_tiles(cb_id_intermed0, cb_id_intermed1, c, c, c);
        }
        // pop intermed0, push it back
        cb_pop_front(cb_id_intermed0, 16);
        cb_reserve_back(cb_id_intermed0, 16);
        for(uint32_t c = 0; c < 16; c++) { pack_tile(c, cb_id_intermed0, c); }
        cb_push_back(cb_id_intermed0, 16);
        release_dst();
        cb_pop_front(cb_id_intermed1, 16);

        // W2 block 0 (left 16 cols)
        cb_wait_front(cb_id_intermed0, 16);
        // Must do full mm_init because we are transitioning from eltwise mul back to matmul!
        mm_init(cb_id_intermed0, cb_id_w2, cb_id_out);
        acquire_dst();
        uint32_t w2_rem = 256;
        in_idx = 0;
        while (w2_rem > 0) {
            uint32_t chunk = 32;
            cb_wait_front(cb_id_w2, chunk);
            for(uint32_t r = 0; r < 2; r++) {
                for(uint32_t c = 0; c < 16; c++) {
                    matmul_tiles(cb_id_intermed0, cb_id_w2, in_idx, r * 16 + c, c);
                }
                in_idx++;
            }
            cb_pop_front(cb_id_w2, chunk);
            w2_rem -= chunk;
        }
        for(uint32_t c = 0; c < 16; c++) { pack_tile(c, cb_id_out, c); }
        release_dst();

        // W2 block 1 (right 16 cols)
        // Can use mm_init_short or mm_init. Let's use mm_init to be safe.
        mm_init(cb_id_intermed0, cb_id_w2, cb_id_out);
        acquire_dst();
        w2_rem = 256;
        in_idx = 0;
        while (w2_rem > 0) {
            uint32_t chunk = 32;
            cb_wait_front(cb_id_w2, chunk);
            for(uint32_t r = 0; r < 2; r++) {
                for(uint32_t c = 0; c < 16; c++) {
                    matmul_tiles(cb_id_intermed0, cb_id_w2, in_idx, r * 16 + c, c);
                }
                in_idx++;
            }
            cb_pop_front(cb_id_w2, chunk);
            w2_rem -= chunk;
        }
        for(uint32_t c = 0; c < 16; c++) { pack_tile(c, cb_id_out, 16 + c); }
        release_dst();
        
        cb_pop_front(cb_id_intermed0, 16);
    }
    
    cb_push_back(cb_id_out, 32);

    // Pop inputs for next token
    cb_pop_front(cb_id_in0, 32);
    cb_pop_front(cb_id_idx, 1);
    cb_pop_front(cb_id_wt, 1);
}