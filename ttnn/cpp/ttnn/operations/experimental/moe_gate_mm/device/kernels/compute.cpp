// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/transpose_wh.h"
#include "top2.h"
#include "top4.h"
#include "mask.h"
#include "top8.h"

#include "compute_kernel_api/eltwise_unary/fill.h"
#include "api/debug/dprint_tensix.h"
#include "compute_kernel_api/tile_move_copy.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#include "ckernel_template.h"
#include "lltt.h"
#endif

#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "ckernel_template.h"
#endif

//=============================================================================
// Add row 0 from CB tile to all 32 rows in DST using MOP
// Uses: Zero SrcA + acc_to_dest mode: DST = 0 + broadcast_row + DST
//=============================================================================

#ifdef TRISC_UNPACK
inline void add_row_to_dst_unpack_init() {
    using namespace ckernel;
    using namespace ckernel::unpacker;

    // Record instructions to replay buffer:
    // [0]: Zero SrcA
    // [1]: Set dvalid for SrcA
    lltt::record(0, 2);
    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_SET_DVALID);

    const uint32_t zero_srca_dvalid = lltt::replay_insn(0, 2);

    // SrcB unpack with Z increment and dvalid
    const uint32_t unpack_srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    const uint32_t srcb_clear_z = TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001);

    // MOP structure matches standard ROW broadcast: outerloop=2, innerloop=2
    // Each iteration: (unpack_srcb, zero_srca_dvalid) - matches (srcB, srcA) pattern
    // end_op resets SrcB Z counter between halves
    ckernel_template tmp(2, 2, unpack_srcb, zero_srca_dvalid);
    tmp.set_end_op(srcb_clear_z);
    tmp.program();
}

inline void add_row_to_dst_unpack(uint32_t row_cb, uint32_t tile_idx) {
    using namespace ckernel;
    using namespace ckernel::unpacker;

    uint32_t op_id = get_operand_id(row_cb);
    uint32_t addr =
        get_local_cb_interface(op_id).fifo_rd_ptr - 1 + get_local_cb_interface(op_id).fifo_page_size * tile_idx;

    // Reset SrcB Z/W counters
    TTI_SETADCZW(0b010, 0, 0, 0, 0, 0b1111);

    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);

    // Set SrcB base address (SrcA is zeroed, no address needed)
    const uint32_t upk1_reg =
        (unp_cfg_context == 0) ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32;
    cfg[upk1_reg] = addr;

    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run the MOP (processes all 4 faces)
    ckernel::ckernel_template::run();

    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
inline void add_row_to_dst_math_init() {
    using namespace ckernel;

    // Configure address modifiers for ROW broadcast
    // SrcA: increment by 8 (8 rows per op)
    // SrcB: no increment (same row broadcast to all)
    // Dest: increment by 8
    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);

    // MOP structure: outerloop=4 (faces), innerloop=2 (8 rows × 2 = 16 rows per face)
    const uint32_t elwadd_op = TT_OP_ELWADD(0, 1 /*acc_to_dest*/, p_elwise::SRCB_BCAST_ROW, ADDR_MOD_0, 0);
    const uint32_t clr_src = TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);

    ckernel_template tmp(4, 2, elwadd_op);
    tmp.set_end_op(clr_src);
    tmp.program();

    // Disable clearing dvalid on SrcA read (we're using zeroed SrcA)
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void add_row_to_dst_math(uint32_t dst_idx) {
    using namespace ckernel;

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_idx);

    // Run the MOP (processes all 4 faces)
    ckernel::ckernel_template::run();

    math::clear_dst_reg_addr();
}
#endif

// Combined init - call once before using add_row_to_dst
inline void add_row_to_dst_init() {
    UNPACK((add_row_to_dst_unpack_init()));
    MATH((add_row_to_dst_math_init()));
}

// Main function - adds row 0 from CB tile to all rows in DST
inline void add_row_to_dst(uint32_t row_cb, uint32_t tile_idx, uint32_t dst_idx) {
    UNPACK((add_row_to_dst_unpack(row_cb, tile_idx)));
    MATH((add_row_to_dst_math(dst_idx)));
}

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto partial_semaphore = get_arg_val<uint32_t>(argidx++);
    const auto is_send_core = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto collector_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto collector_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5 = tt::CBIndex::c_7;

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = is_send_core ? (2 * 72) : (2 * 76 + 1);
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 8;
    constexpr uint32_t w_tiles_per_txn = noc_packet_size / 2048;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t w_num_blocks = num_w_tiles_h * num_w_tiles_w / w_tiles_per_block;
    const uint32_t w_remaining_tiles = (num_w_tiles_h * num_w_tiles_w) % w_tiles_per_block;
    const uint32_t w_last_block_txns = (w_remaining_tiles + w_tiles_per_txn - 1) / w_tiles_per_txn;  // Ceiling division
    const uint32_t w_tiles_per_block_last = w_remaining_tiles - 1;
    const uint32_t bias_tile_index = w_tiles_per_block_last;  // Bias is the last tile in the last block

    //-------------------------------------------------------------------------
    // Compute configuration
    //-------------------------------------------------------------------------
    // Pack is configured to Float16_b
    pack_reconfig_data_format(cb_s2c_out);

    // Unpacker B is for input/activation, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for weight, so Float16_b
    reconfig_data_format_srca(cb_r2c_w);

    if (is_send_core) {
        // Initialize matmul: input @ weight -> output
        mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/2, /*rt_dim=*/1, /*kt_dim=*/1);

        //-------------------------------------------------------------------------
        // Compute: input @ 2 weights -> 2 outputss
        //-------------------------------------------------------------------------
        tile_regs_acquire();

        uint32_t tile_index = 2 * 76;
        for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
            cb_wait_front(cb_r2c_w, w_tiles_per_block);

            for (uint32_t tile_id = 0; tile_id < w_tiles_per_block; tile_id += 2) {
                // Perform matmul: 1 input tile @ 2 weight tiles
                matmul_block(
                    cb_s2c_in,
                    cb_r2c_w,
                    /*in0_index=*/tile_index++,
                    /*in1_index=*/tile_id,
                    /*idst=*/0,
                    /*transpose=*/false,
                    /*ct_dim=*/2,
                    /*rt_dim=*/1,
                    /*kt_dim=*/1);
            }
            cb_pop_front(cb_r2c_w, w_tiles_per_block);
        }

        // Last block
        cb_wait_front(cb_r2c_w, w_tiles_per_block);
        for (uint32_t tile_id = 0; tile_id < w_tiles_per_block_last; tile_id += 2) {
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/tile_index++,
                /*in1_index=*/tile_id,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/2,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }
        cb_pop_front(cb_r2c_w, w_tiles_per_block);

        tile_regs_commit();

        tile_regs_wait();

        cb_reserve_back(cb_c2w_rdy, 1);
        // Since neighbor1 is farther, we send it first (dst 1)
        pack_tile</*out_of_order_output=*/true>(1, cb_s2c_out, /*output_tile_index=*/0);
        cb_push_back(cb_c2w_rdy, 1);

        cb_reserve_back(cb_c2w_rdy, 1);
        pack_tile</*out_of_order_output=*/true>(0, cb_s2c_out, /*output_tile_index=*/0);
        cb_push_back(cb_c2w_rdy, 1);

        tile_regs_release();
    } else {
        // Initialize matmul: input @ weight -> output
        mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/1);

        //-------------------------------------------------------------------------
        // Compute: input @ weight -> output
        //-------------------------------------------------------------------------
        tile_regs_acquire();

        uint32_t tile_index = 0;
        for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
            cb_wait_front(cb_r2c_w, w_tiles_per_block);

            for (uint32_t tile_id = 0; tile_id < w_tiles_per_block; ++tile_id) {
                // Perform matmul: 1 input tile @ 1 weight tile
                matmul_block(
                    cb_s2c_in,
                    cb_r2c_w,
                    /*in0_index=*/tile_index++,
                    /*in1_index=*/tile_id,
                    /*idst=*/0,
                    /*transpose=*/false,
                    /*ct_dim=*/1,
                    /*rt_dim=*/1,
                    /*kt_dim=*/1);
            }
            cb_pop_front(cb_r2c_w, w_tiles_per_block);
        }

        // Last block
        cb_wait_front(cb_r2c_w, w_tiles_per_block);
        for (uint32_t tile_id = 0; tile_id < w_tiles_per_block_last; ++tile_id) {
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/tile_index++,
                /*in1_index=*/tile_id,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/1,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w2c_in2);

        // Wait for the partial to come, add it
        cb_wait_front(cb_w2c_in2, 1);
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w2c_in2, 0, 0);
        cb_pop_front(cb_w2c_in2, 1);

        //-------------------------------------------------------------------------
        // Sigmoid
        //-------------------------------------------------------------------------

        // Sigmoid the output
        sigmoid_tile_init();
        sigmoid_tile(0);

        //-------------------------------------------------------------------------
        // Retain a copy
        //-------------------------------------------------------------------------
        // Retain this copy for final scores (raw scores)
        copy_dest_values_init();
        copy_dest_values(1, 0);

        //-------------------------------------------------------------------------
        // Add bias
        //-------------------------------------------------------------------------
        add_row_to_dst_init();
        add_row_to_dst(cb_r2c_w, bias_tile_index, 0);

        //-------------------------------------------------------------------------
        // Sum of top2 scores for this group
        //-------------------------------------------------------------------------
        // Pack the output and bring it back as transposed
        tile_regs_commit();
        tile_regs_wait();

        // Send the bias adjusted scores to be sent over to the collector core
        cb_reserve_back(cb_s2c_out, 1);
        pack_tile(0, cb_s2c_out);
        cb_push_back(cb_s2c_out, 1);

        // Pack the raw scores to be sent over to the collector core
        cb_reserve_back(cb_w2c_in5, 1);
        pack_tile(1, cb_w2c_in5);
        cb_push_back(cb_w2c_in5, 1);

        // Signal to DM1 that both are ready
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);

        tile_regs_release();

        cb_wait_front(cb_s2c_out, 1);
        tile_regs_acquire();

        // Transpose
        transpose_wh_init_short(cb_s2c_out);
        transpose_wh_tile(cb_s2c_out, 0, 0);

        // Sum the top-2 of the output
        sum_top2_tile_init();
        sum_top2_tile(0);

        tile_regs_commit();

        cb_reserve_back(cb_c2w_rdy, 1);

        tile_regs_wait();
        // Pack output tile
        pack_tile(0, cb_c2w_rdy);
        tile_regs_release();
        cb_push_back(cb_c2w_rdy, 1);

        cb_pop_front(cb_r2c_w, w_tiles_per_block);

        //-------------------------------------------------------------------------
        // Collector core
        //-------------------------------------------------------------------------
        if (core_id == 7) {
            // I am collecting, let us wait for everyone else to finish sending their data to me
            cb_wait_front(cb_w2c_in3, 8);

            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_w2c_in3);

            // Copy the group scores
            copy_tile(cb_w2c_in3, 7, 7);

            //-------------------------------------------------------------------------
            // Top 4 groups for each token
            //-------------------------------------------------------------------------
            top4_tile_init();
            top4_tile(7);

            //-------------------------------------------------------------------------
            // Adjusted scores -> Copy to mask and find top8 experts for each token
            //-------------------------------------------------------------------------
            mask_group_init();
            for (uint32_t i = 0; i < 7; i++) {
                copy_tile(cb_w2c_in3, i, i);
            }
            mask_group<0>(0);
            mask_group<1>(1);
            mask_group<2>(2);
            mask_group<3>(3);
            mask_group<4>(4);
            mask_group<5>(5);
            mask_group<6>(6);

            // My own data
            copy_tile(cb_s2c_out, 0, 7);
            mask_group<7>(7);

            top8_bitonic_tile_init();
            top8_bitonic_tile(0);

            cb_pop_front(cb_w2c_in3, 8);
            //-------------------------------------------------------------------------
            // Raw scores -> Use it to normalize and pack
            //-------------------------------------------------------------------------
            cb_wait_front(cb_w2c_in4, 7);

            cb_pop_front(cb_w2c_in4, 7);

            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
        cb_pop_front(cb_s2c_out, 1);
    }
}
