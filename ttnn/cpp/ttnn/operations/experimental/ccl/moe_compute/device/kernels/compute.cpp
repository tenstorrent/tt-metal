// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ring_common.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

// Need these headers for running SFPU on PACK thread
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
#include "llk_math_eltwise_binary_sfpu_binop.h"
#endif

FORCE_INLINE
void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    WAYPOINT("NSMW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) < val);
    WAYPOINT("NSMD");
}

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // For synchronization with tilize cores
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_s2c_in = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;
    constexpr auto cb_w2c_md = tt::CBIndex::c_5;

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_1;

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = moe_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_ring::W2_TILES_PER_CORE_A[ring_core_id];

    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile =
        4 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;  // 32
    constexpr uint32_t w0_w1_blocks_per_expert =
        w0_w1_blocks_per_two_elt_tile * moe_ring::IN2_TILES_PER_STEP_A /
        2;  // 32 * 3 = 96
            // 2 * num_w0_w1_tiles_w * num_w0_w1_tiles_h / w0_w1_tiles_per_block;  // (5|6 * 224) / 28 = 80|96

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = moe_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;               // 14 * 2 = 28
    constexpr uint32_t w2_txns_h = (num_w2_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;  // 5 (round up)
    constexpr uint32_t w2_blocks_per_four_mm2_tile = 4 * w2_txns_h / w2_txns_per_block;         // 4 * 5 / 2 = 10
    constexpr uint32_t w2_blocks_per_expert = moe_ring::W2_BLOCKS_PER_EXPERT;

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_ring::NUM_A2A_ITERS_A;

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_ring::NUM_CORES;

    // The number of tiles to send in each step
    // We send 6 tiles in each step, even though some cores in some steps may have only 5 valid ones
    constexpr uint32_t tiles_per_step = moe_ring::IN2_TILES_PER_STEP_A;  // max(num_w0_w1_tiles_w)

    //-------------------------------------------------------------------------
    // Compute
    //-------------------------------------------------------------------------
    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_s2c_in2);

    // Unpacker B is for input/activation and eltiwse inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for W0,W1 and W2, so Bf4_b
    reconfig_data_format_srca(cb_r2c_w0_w1);

    // Initialize matmul for W0
    mm_block_init(cb_s2c_in, cb_r2c_w0_w1, cb_s2c_in2, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);

    // Initialize SFPU for SILU and eltwise multiply
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Receive num_tokens_per_expert from the tilize cores, wait for signal from writer that the data has arrived
    // We also use this CB to transfer (from the writer to compute) 2 semaphore addresses:
    // - 0: address of semaphore used to send metadata (number of tokens per expert)
    // - 1: address of semaphore used to notify matmuls cores that tilized chunks have arrived
    cb_wait_front(cb_w2c_md, 2);
    cb_pop_front(cb_w2c_md, 2);
    volatile tt_l1_ptr uint32_t* cb_w2c_md_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_w2c_md, 0));

    // Precompute NUM_CHUNKS_PER_EXPERT
    volatile tt_l1_ptr uint32_t* metadata_ready_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_w2c_md_read_ptr[0]);
    uint32_t encoded_metadata_value = *metadata_ready_semaphore_ptr;

    constexpr uint32_t BITS_PER_EXPERT = 10;
    constexpr uint32_t EXPERT_MASK = 0x3FFu;
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = (encoded_metadata_value >> (1 + BITS_PER_EXPERT * expert_id)) & EXPERT_MASK;
        NUM_CHUNKS_PER_EXPERT[expert_id] = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    }

    // Value we wait on that indicates the next chunk of tiles have arrived from the tilize cores
    uint32_t matmul_chunk_ready_semaphore_wait_value = 1;
    uint32_t matmul_chunk_ready_semaphore_addr = cb_w2c_md_read_ptr[1];

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------

    // This decides which half of the buffer will have the valid data sent by tilize cores
    bool use_second_half_buffer = false;

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];
        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            //---------------------------------------------------------------------
            // Compute in @ {W0,W1}
            //---------------------------------------------------------------------

            // Wait for next chunk of tiles to arrive from the tilize cores
            // Min to allow tilize cores to send increment for second expert
            // while first expert still being processed
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(matmul_chunk_ready_semaphore_addr),
                matmul_chunk_ready_semaphore_wait_value++);

            for (uint32_t tile_id = 0; tile_id < tiles_per_step; tile_id += 2) {
                uint32_t in0_index = use_second_half_buffer ? num_w0_w1_tiles_h : 0;

                tile_regs_acquire();
                for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_two_elt_tile; ++block_id) {
                    cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);

                    for (uint32_t k = 0; k < w0_w1_tiles_per_block; k += 4) {
                        matmul_block(
                            cb_s2c_in,
                            cb_r2c_w0_w1,
                            in0_index++,
                            /*in1_index=*/k,
                            /*idst=*/0,
                            /*transpose=*/false,
                            /*ct_dim=*/4,
                            /*rt_dim=*/1,
                            /*kt_dim=*/1);
                    }
                    cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
                }

                tile_regs_commit();

                // The below is equivalent to tile_regs_wait(), but we stall CFG as well, so that the succeeding
                // TT_SETC16 instruction is also stalled until math thread is done with these dest registers.
                TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK),
                    p_stall::STALL_ON_ZERO);

                // Make SFPU access the appropriate half of the destination registers
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                //---------------------------------------------------------------------
                // Apply SILU activation and then eltwise multiply
                //---------------------------------------------------------------------
                PACK((llk_math_eltwise_unary_sfpu_silu<true, false>(0)));
                PACK((llk_math_eltwise_unary_sfpu_silu<true, false>(2)));

                PACK((llk_math_eltwise_binary_sfpu_binop<true, ckernel::BinaryOp::MUL>(0, 1, 0)));
                PACK((llk_math_eltwise_binary_sfpu_binop<true, ckernel::BinaryOp::MUL>(2, 3, 2)));

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
            //---------------------------------------------------------------------
            uint32_t out_tile_index = use_second_half_buffer ? num_w0_w1_tiles_h : 0;
            for (uint32_t iter = 0; iter < num_a2a_iters; ++iter) {
                uint32_t dm1_step = 0;
                uint32_t dm1_tiles_remaining = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
                cb_wait_front(cb_w2c_rdy, 1);

                uint32_t in2_offset = 0, in2_index = 0;

                tile_regs_acquire();

                for (uint32_t block_id = 0; block_id < w2_blocks_per_four_mm2_tile; ++block_id) {
                    cb_wait_front(cb_r2c_w2, w2_tiles_per_block);

                    for (uint32_t k = 0; k < w2_tiles_per_block; k += 4) {
                        // The last block has only 4 tiles of interest, so we exit early.
                        if ((block_id == (w2_blocks_per_four_mm2_tile - 1)) && (k == 4)) {
                            cb_pop_front(cb_w2c_rdy, 1);
                            break;
                        }

                        if (dm1_tiles_remaining == 0) {
                            cb_pop_front(cb_w2c_rdy, 1);
                            cb_wait_front(cb_w2c_rdy, 1);
                            dm1_tiles_remaining = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][++dm1_step];
                            in2_offset = (in2_offset == tiles_per_step) ? 0 : tiles_per_step;
                            in2_index = in2_offset;
                        }
                        dm1_tiles_remaining--;

                        matmul_block(
                            cb_s2c_in2,
                            cb_r2c_w2,
                            in2_index++,
                            /*in1_index=*/k,
                            /*idst=*/0,
                            /*transpose=*/false,
                            /*ct_dim=*/4,
                            /*rt_dim=*/1,
                            /*kt_dim=*/1);
                    }
                    cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
                }

                tile_regs_commit();

                tile_regs_wait();
                // Pack this in-place for now.
                pack_tile</*out_of_order_output=*/true>(0, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
                pack_tile</*out_of_order_output=*/true>(1, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
                pack_tile</*out_of_order_output=*/true>(2, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
                pack_tile</*out_of_order_output=*/true>(3, cb_c2s_out, /*output_tile_index=*/out_tile_index++);
                tile_regs_release();
            }

            // Toggle the buffer to use
            use_second_half_buffer = !use_second_half_buffer;
        }  // end for (chunk)
    }  // end for (expert_id)

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w2, w2_tiles_per_block);
    cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
}
