// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint_tensix.h"

// Need these headers for running SFPU on PACK thread
#ifdef TRISC_PACK
#include "swiglu_sfpu.h"
#endif

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;
    constexpr auto cb_debug = tt::CBIndex::c_5;
    constexpr auto cb_debug2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;

    // Constants for MoEGPT
    // GPT-OSS: K=2880 -> 90 tiles height, N=2880 -> 90 tiles
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_ring::NUM_W0_W1_TILES_H;  // 90
    constexpr uint32_t num_w2_tiles_h = moe_gpt_ring::NUM_W2_TILES_H;        // 90

    const uint32_t num_w0_w1_tiles_w = moe_gpt_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];  // 7 or 8
    const uint32_t num_w2_tiles_w = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];                    // 7 or 8

    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_gpt_ring::W0_W1_TXNS_PER_BLOCK;           // 2
    constexpr uint32_t w0_w1_tiles_per_txn = moe_gpt_ring::W0_W1_TILES_PER_TXN;             // 10
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 10 * 2 = 20
    // blocks to read for 2 output element tiles (covers full K height, 4 tiles wide per block):
    // 4 * (90 / 10) / 2 = 4 * 9 / 2 = 18
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile =
        4 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;  // 18
    // blocks per expert = 18 * 8 / 2 = 72
    constexpr uint32_t w0_w1_blocks_per_expert = moe_gpt_ring::W0_B0_W1_B1_BLOCKS_PER_EXPERT;

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = moe_gpt_ring::W2_TXNS_PER_BLOCK;                     // 2
    constexpr uint32_t w2_tiles_per_txn = moe_gpt_ring::W2_TILES_PER_TXN;                       // 10
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;               // 10 * 2 = 20
    constexpr uint32_t w2_txns_h = (num_w2_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;  // 90 / 10 = 9
    // blocks for 4 output tiles: 4 * 9 / 2 = 18
    constexpr uint32_t w2_blocks_per_four_mm2_tile = 4 * w2_txns_h / w2_txns_per_block;  // 18
    constexpr uint32_t w2_blocks_per_expert = moe_gpt_ring::W2_B2_BLOCKS_PER_EXPERT;     // 36

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_gpt_ring::NUM_A2A_ITERS_A;  // 2

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_gpt_ring::NUM_CORES;  // 12

    // The number of tiles to send in each step (max of 7/8 = 8)
    constexpr uint32_t tiles_per_step = moe_gpt_ring::IN2_TILES_PER_STEP_A;  // 8

    //-------------------------------------------------------------------------
    // Compute
    //-------------------------------------------------------------------------
    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_s2c_in2);

    // Unpacker B is for input/activation and eltwise inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for W0,W1 and W2, so Bf4_b
    reconfig_data_format_srca(cb_r2c_w0_w1);

    // Initialize matmul for W0
    mm_block_init(cb_s2c_in, cb_r2c_w0_w1, cb_s2c_in2, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);

    // Initialize SFPU for GPT-OSS SwiGLU activation
    PACK((llk_math_eltwise_binary_sfpu_swiglu_init<true>()));

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    uint32_t in0_offset_per_expert = 0;
    uint32_t out_offset_per_expert = 0;
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        //---------------------------------------------------------------------
        // Cross-expert boundary synchronization
        // For expert > 0, wait for dm1 to confirm that our predecessor's last
        // A2A write (which targets buf 0) has completed, so it's safe to
        // overwrite buf 0 with this expert's SwiGLU output.
        //---------------------------------------------------------------------
        if (expert_id > 0) {
            cb_wait_front(cb_w2c_rdy, 1);
            cb_pop_front(cb_w2c_rdy, 1);
        }

        //---------------------------------------------------------------------
        // Compute in @ {W0,W1}
        // GPT-OSS: 8 tiles/core (max), processed 2 at a time -> 4 iterations
        //---------------------------------------------------------------------
        for (uint32_t tile_id = 0; tile_id < tiles_per_step; tile_id += 2) {
            uint32_t in0_index =
                expert_id * num_w0_w1_tiles_h;  // expert 0: 0, expert 1: 90, expert 2: 180, expert 3: 270

            tile_regs_acquire();
            uint32_t k_tracker = 0;
            // clean up 13 later 91/7 = 13
            for (uint32_t block_id = 0; block_id < 13; ++block_id) {
                cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
                cb_push_back(cb_debug, 1);
                cb_reserve_back(cb_debug, 1);
                cb_wait_front(cb_debug, 1);
                cb_pop_front(cb_debug, 1);
                // DPRINT << "new block "<< block_id << ENDL();
                for (uint32_t k = 0; k < w0_w1_tiles_per_block; k += 4) {
                    // DPRINT <<"k dim: " << k_tracker <<ENDL();

                    // UNPACK(tt::compute::common::print_full_tile(cb_s2c_in,in0_index , true));
                    // in0_index++;
                    // copy_tile_init(cb_s2c_in);
                    // copy_tile(cb_r2c_w0_w1, k,0);
                    // // copy_tile_init(cb_r2c_w0_w1);
                    // copy_tile(cb_r2c_w0_w1, k +1,1);
                    // copy_tile(cb_s2c_in, in0_index,2);
                    // DPRINT << "W0: "<<ENDL();
                    // dprint_tensix_dest_reg(0);
                    // DPRINT << "W1: "<<ENDL();
                    // dprint_tensix_dest_reg(1);
                    // DPRINT << "in: "<<ENDL();
                    // dprint_tensix_dest_reg(2);
                    if (k_tracker == num_w0_w1_tiles_h) {
                        break;
                    }
                    // matmul_block(
                    //     cb_s2c_in,
                    //     cb_r2c_w0_w1,
                    //     in0_index++,
                    //     /*in1_index=*/k,
                    //     /*idst=*/0,
                    //     /*transpose=*/false,
                    //     /*ct_dim=*/4,
                    //     /*rt_dim=*/1,
                    //     /*kt_dim=*/1);
                    // k_tracker++;
                }
                if (k_tracker == num_w0_w1_tiles_h) {
                    // add matmul bias logic here
                }
                cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
            }

            tile_regs_commit();

            // The below is equivalent to tile_regs_wait(), but we stall CFG as well, so that the succeeding
            // TT_SETC16 instruction is also stalled until math thread is done with these dest registers.
            PACK(TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO));

            // Make SFPU access the appropriate half of the destination registers
            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

            //---------------------------------------------------------------------
            // Apply GPT-OSS SwiGLU activation
            // SwiGLU: (clamp(up)+1) * clamp(gate) * sigmoid(alpha * clamp(gate))
            // Dest layout: [gate0, up0, gate1, up1] at tile indices [0, 1, 2, 3]
            //---------------------------------------------------------------------
            PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(0, 1, 0)));
            PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(2, 3, 2)));

            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

            pack_tile</*out_of_order_output=*/true>(0, cb_s2c_in2, /*output_tile_index=*/tile_id);
            pack_tile</*out_of_order_output=*/true>(2, cb_s2c_in2, /*output_tile_index=*/tile_id + 1);
            tile_regs_release();
        }

        // Signal to DM1 that the output from this core is ready
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);

        //---------------------------------------------------------------------
        // Compute in2 @ W2 (in pairs of 4)
        // GPT-OSS: W2 height=90, 18 blocks per iter, 2 iters
        //---------------------------------------------------------------------
        uint32_t out_tile_index =
            expert_id * num_w0_w1_tiles_h;  // expert 0: 0, expert 1: 90, expert 2: 180, expert 3: 270
        for (uint32_t iter = 0; iter < num_a2a_iters; ++iter) {
            uint32_t dm1_step = 0;
            uint32_t dm1_tiles_remaining = moe_gpt_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
            cb_wait_front(cb_w2c_rdy, 1);

            // 6-buffer cycling: each A2A step uses buf (step % 6).
            // Matches dm1's scheme where step S reads from buf S % 6.
            uint32_t in2_buf = 0, in2_offset = 0, in2_index = 0;

            tile_regs_acquire();
            uint32_t k_tracker = 0;

            // 90/13
            for (uint32_t block_id = 0; block_id < 13; ++block_id) {
                cb_wait_front(cb_r2c_w2, w2_tiles_per_block);

                for (uint32_t k = 0; k < w2_tiles_per_block; k += 4) {
                    if (k_tracker == num_w0_w1_tiles_h) {
                        break;
                    }
                    if (dm1_tiles_remaining == 0) {
                        // cb_pop_front(cb_w2c_rdy, 1);
                        // cb_wait_front(cb_w2c_rdy, 1);
                        // dm1_tiles_remaining =
                        // moe_gpt_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][++dm1_step]; in2_buf = (in2_buf
                        // >= 5) ? 0 : in2_buf + 1;  // 6 buffers: cycle 0..5 in2_offset = in2_buf * tiles_per_step;
                        // in2_index = in2_offset;
                    }
                    // dm1_tiles_remaining--;
                    DPRINT << "k dim: " << k_tracker << ENDL();

                    UNPACK(tt::compute::common::print_full_tile(cb_r2c_w2, k, true));
                    // matmul_block(
                    //     cb_s2c_in2,
                    //     cb_r2c_w2,
                    //     in2_index++,
                    //     /*in1_index=*/k,
                    //     /*idst=*/0,
                    //     /*transpose=*/false,
                    //     /*ct_dim=*/4,
                    //     /*rt_dim=*/1,
                    //     /*kt_dim=*/1);
                    k_tracker++;
                }
                if (k_tracker == num_w0_w1_tiles_h) {
                    // add matmul bias logic here
                }
                cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
            }

            // Pop the last cb_w2c_rdy entry that was waited on but not popped.
            // Each iter starts with cb_wait_front(cb_w2c_rdy) and has 11 internal
            // pop+wait transitions for 12 total steps. Without this final pop,
            // cb_w2c_rdy stays full and dm1 deadlocks on cb_reserve_back.
            cb_pop_front(cb_w2c_rdy, 1);

            tile_regs_commit();

            tile_regs_wait();
            // Pack this in-place for now.
            pack_tile</*out_of_order_output=*/true>(0, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
            pack_tile</*out_of_order_output=*/true>(1, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
            pack_tile</*out_of_order_output=*/true>(2, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
            pack_tile</*out_of_order_output=*/true>(3, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
            tile_regs_release();
        }
    }  // end for (expert_id)

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w2, w2_tiles_per_block);
    cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
}
