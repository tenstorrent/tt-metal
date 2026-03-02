// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for moe_gpt_fused - same logic as moe_gpt compute.cpp
// Performs: SwiGLU(input @ W0, input @ W1) @ W2 via ring A2A

#include "moe_gpt_fused_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack_untilize.h"

#ifdef TRISC_PACK
#include "swiglu_sfpu.h"
#endif

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");

    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);

    // CBs (same as moe_gpt)
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;
    constexpr auto cb_c2s_out = tt::CBIndex::c_14;  // untilized ROW_MAJOR output

    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_fused_ring::NUM_W0_W1_TILES_H;  // 90
    constexpr uint32_t num_w2_tiles_h = moe_gpt_fused_ring::NUM_W2_TILES_H;        // 90

    const uint32_t num_w0_w1_tiles_w = moe_gpt_fused_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_gpt_fused_ring::W2_TILES_PER_CORE_A[ring_core_id];

    // W0/W1 constants
    constexpr uint32_t w0_w1_txns_per_block = moe_gpt_fused_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_gpt_fused_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;
    constexpr uint32_t w0_w1_blocks_per_two_elt_tile =
        4 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;

    // W2 constants
    constexpr uint32_t w2_txns_per_block = moe_gpt_fused_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_gpt_fused_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;
    constexpr uint32_t w2_txns_h = (num_w2_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;
    constexpr uint32_t w2_blocks_per_four_mm2_tile = 4 * w2_txns_h / w2_txns_per_block;

    // Ring setup
    constexpr uint32_t num_a2a_iters = moe_gpt_fused_ring::NUM_A2A_ITERS_A;
    constexpr uint32_t tiles_per_step = moe_gpt_fused_ring::IN2_TILES_PER_STEP_A;

    //-------------------------------------------------------------------------
    // Initialize compute
    //-------------------------------------------------------------------------
    pack_reconfig_data_format(cb_s2c_in2);
    reconfig_data_format_srcb(cb_s2c_in);
    reconfig_data_format_srca(cb_r2c_w0_w1);

    mm_block_init(cb_s2c_in, cb_r2c_w0_w1, cb_s2c_in2, false, 4, 1, 1);

    PACK((llk_math_eltwise_binary_sfpu_swiglu_init<true>()));

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Cross-expert boundary sync
        if (expert_id > 0) {
            cb_wait_front(cb_w2c_rdy, 1);
            cb_pop_front(cb_w2c_rdy, 1);
        }

        // Compute input @ {W0, W1} with SwiGLU
        // Input is shared at c_1[0..89] for all experts (no per-expert copy)
        for (uint32_t tile_id = 0; tile_id < tiles_per_step; tile_id += 2) {
            uint32_t in0_index = 0;  // shared input always at position 0

            tile_regs_acquire();
            for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_two_elt_tile; ++block_id) {
                cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);

                for (uint32_t k = 0; k < w0_w1_tiles_per_block; k += 4) {
                    matmul_block(cb_s2c_in, cb_r2c_w0_w1, in0_index++, k, 0, false, 4, 1, 1);
                }
                cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
            }

            tile_regs_commit();

            TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO);

            PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

            // SwiGLU: (clamp(up)+1) * clamp(gate) * sigmoid(alpha * clamp(gate))
            PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(0, 1, 0)));
            PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(2, 3, 2)));

            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

            pack_tile<true>(0, cb_s2c_in2, tile_id);
            pack_tile<true>(2, cb_s2c_in2, tile_id + 1);
            tile_regs_release();
        }

        // Signal dm1 that SwiGLU output is ready
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);

        // Compute intermediate @ W2
        // Output is untilized into c_14 (ROW_MAJOR) for combine core writes
        constexpr uint32_t source_width_tiles = moe_gpt_fused_ring::SOURCE_WIDTH_TILES;  // 8
        cb_reserve_back(cb_c2s_out, moe_gpt_fused_ring::TOKENS_PER_CHUNK);               // 32 pages

        for (uint32_t iter = 0; iter < num_a2a_iters; ++iter) {
            uint32_t dm1_step = 0;
            uint32_t dm1_tiles_remaining = moe_gpt_fused_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
            cb_wait_front(cb_w2c_rdy, 1);

            uint32_t in2_buf = 0, in2_offset = 0, in2_index = 0;

            tile_regs_acquire();

            for (uint32_t block_id = 0; block_id < w2_blocks_per_four_mm2_tile; ++block_id) {
                cb_wait_front(cb_r2c_w2, w2_tiles_per_block);

                for (uint32_t k = 0; k < w2_tiles_per_block; k += 4) {
                    if (dm1_tiles_remaining == 0) {
                        cb_pop_front(cb_w2c_rdy, 1);
                        cb_wait_front(cb_w2c_rdy, 1);
                        dm1_tiles_remaining =
                            moe_gpt_fused_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][++dm1_step];
                        in2_buf = (in2_buf >= 5) ? 0 : in2_buf + 1;
                        in2_offset = in2_buf * tiles_per_step;
                        in2_index = in2_offset;
                    }
                    dm1_tiles_remaining--;

                    matmul_block(cb_s2c_in2, cb_r2c_w2, in2_index++, k, 0, false, 4, 1, 1);
                }
                cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
            }

            cb_pop_front(cb_w2c_rdy, 1);

            tile_regs_commit();

            tile_regs_wait();
            pack_untilize_dest_init</*block_ct_dim=*/4, /*full_ct_dim=*/source_width_tiles>(cb_c2s_out);
            pack_untilize_dest</*block_ct_dim=*/4, /*full_ct_dim=*/source_width_tiles>(
                cb_c2s_out, /*block_rt_dim=*/1, /*block_c_index=*/iter);
            pack_untilize_uninit(cb_c2s_out);
            tile_regs_release();
        }

        cb_push_back(cb_c2s_out, moe_gpt_fused_ring::TOKENS_PER_CHUNK);  // 32 pages

        // Reinit packer for next expert's SwiGLU pack_tile calls
        pack_reconfig_data_format(cb_s2c_in2);
    }

    // Drain pipeline
    cb_wait_front(cb_r2c_w2, w2_tiles_per_block);
    cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
}
