// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_kloop_helpers.hpp"

#include "bias_bcast_sfpu.h"
#include "top2_sum_sfpu.h"
#include "top4_sfpu.h"
#include "top8_sfpu.h"
#include "top8_merge_sfpu.h"

// ── Functors threaded into compute_kernel_lib::matmul_kloop_pack ───────────

// Send-core pack body. Two separate cb_reserve+pack_tile+cb_push for the two
// DST tiles (DST 1 first because neighbor1 is farther — send order matters).
template <uint32_t OutCbId, uint32_t SignalCbId>
struct MoEGateMMSendPack {
    ALWI void operator()() const {
        tile_regs_wait();

        cb_reserve_back(SignalCbId, 1);
        // Since neighbor1 is farther, we send it first (dst 1)
        pack_tile</*out_of_order_output=*/true>(1, OutCbId, /*output_tile_index=*/0);
        cb_push_back(SignalCbId, 1);

        cb_reserve_back(SignalCbId, 1);
        pack_tile</*out_of_order_output=*/true>(0, OutCbId, /*output_tile_index=*/0);
        cb_push_back(SignalCbId, 1);
    }
};

// Non-send-core post-K compute body. MATH-thread SFPU sequence on DST after
// the matmul accumulator is built: partial-add → sigmoid → copy_dest (raw
// scores retention) → bias add. Modifies DST so must run before commit.
template <uint32_t PartialCbId, uint32_t W1CbId>
struct MoEGateMMNonSendPostK {
    uint32_t bias_tile_index;

    ALWI void operator()() const {
        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(PartialCbId);
        cb_wait_front(PartialCbId, 1);
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(PartialCbId, 0, 0);
        cb_pop_front(PartialCbId, 1);

        sigmoid_tile_init();
        sigmoid_tile(0);

        // Retain a copy for final scores (raw scores).
        copy_dest_values_init();
        copy_dest_values(0, 1);

        // Add bias.
        copy_tile_init(W1CbId);
        copy_tile(W1CbId, bias_tile_index, 2);
        add_bias_init();
        add_bias(0);
    }
};

// Non-send-core pack body. Packs DST 0 (bias-adjusted scores) to OutCbId and
// DST 1 (raw scores retained via copy_dest above) to RawScoresCbId.
template <uint32_t OutCbId, uint32_t RawScoresCbId>
struct MoEGateMMNonSendPack {
    ALWI void operator()() const {
        tile_regs_wait();

        // Bias-adjusted scores → cb_s2c_out
        cb_reserve_back(OutCbId, 1);
        pack_tile</*out_of_order_output=*/true>(0, OutCbId, /*output_tile_index=*/0);
        cb_push_back(OutCbId, 1);

        // Raw scores → cb_w2c_in3
        cb_reserve_back(RawScoresCbId, 1);
        pack_tile(1, RawScoresCbId);
        cb_push_back(RawScoresCbId, 1);
    }
};

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t column_id = get_named_compile_time_arg_val("column_id");

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
    const auto raw_scores_semaphore = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5 = tt::CBIndex::c_7;
    constexpr auto cb_w2c_in6 = tt::CBIndex::c_8;
    constexpr auto cb_w2c_in7 = tt::CBIndex::c_9;

    // Aliases
    constexpr auto cb_w2c_in8 = tt::CBIndex::c_6;

    // Buffer wrappers consumed by matmul_kloop_pack (which owns the K-loop
    // cb_wait/pop on in1 and the FMA stride). Pack-target CBs (cb_c2w_rdy,
    // cb_s2c_out, cb_w2c_in3) are managed by each call's pack_body lambda.
    experimental::CircularBuffer in0_buf(cb_s2c_in);
    experimental::CircularBuffer in1_buf(cb_r2c_w);

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
    // Collector core
    //-------------------------------------------------------------------------
    constexpr uint32_t COLLECTOR_CORE_ID = 7;

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
        // Compute: input @ 2 weights -> 2 outputs
        //-------------------------------------------------------------------------
        // matmul_kloop_pack Form 2: K-loop + custom pack body, partial last
        // block (w_tiles_per_block_last < w_tiles_per_block).
        using KStep = KStepDefault<experimental::CircularBuffer>;
        using PackBody = MoEGateMMSendPack<cb_s2c_out, cb_c2w_rdy>;
        KStep k_step{in0_buf, in1_buf, /*in0_index=*/2 * 76, /*transpose=*/false};
        matmul_kloop_pack(
            in1_buf,
            SegmentedKLoopShape::of(
                /*num_blocks=*/w_num_blocks,
                /*tiles_per_block=*/w_tiles_per_block,
                /*ct_dim=*/2,
                /*rt_dim=*/1,
                /*kt_dim=*/1,
                /*last_block_tiles=*/w_tiles_per_block_last),
            k_step,
            PackBody{});

        return;
    }

    // -------------------------------------------------------------------------
    // Rest of the 8 cores do more
    // -------------------------------------------------------------------------

    // Initialize matmul: input @ weight -> output
    mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/1);

    //-------------------------------------------------------------------------
    // Compute: input @ weight -> output (with mid-DST SFPU: partial-add, sigmoid,
    // retain raw scores via copy_dest, bias add)
    //-------------------------------------------------------------------------
    // matmul_kloop_pack Form 3: K-loop + post-K compute body + custom pack
    // body. last_block_no_pop=true because the bias copy_tile below reads
    // cb_r2c_w at bias_tile_index in the (still-fronted) last block; the
    // trailing cb_pop_front fires after the matmul_kloop_pack scope, once
    // the bias is consumed.
    using NonSendKStep = KStepDefault<experimental::CircularBuffer>;
    using NonSendPostK = MoEGateMMNonSendPostK<cb_w2c_in2, cb_r2c_w>;
    using NonSendPack = MoEGateMMNonSendPack<cb_s2c_out, cb_w2c_in3>;
    NonSendKStep k_step{in0_buf, in1_buf, /*in0_index=*/0, /*transpose=*/false};
    matmul_kloop_pack(
        in1_buf,
        SegmentedKLoopShape::of(
            /*num_blocks=*/w_num_blocks,
            /*tiles_per_block=*/w_tiles_per_block,
            /*ct_dim=*/1,
            /*rt_dim=*/1,
            /*kt_dim=*/1,
            /*last_block_tiles=*/w_tiles_per_block_last,
            /*last_block_no_pop=*/true),
        k_step,
        NonSendPack{},
        NonSendPostK{/*bias_tile_index=*/bias_tile_index});

    tile_regs_acquire();

    // Transpose
    cb_wait_front(cb_s2c_out, 1);
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
    // Non-collector cores
    //-------------------------------------------------------------------------
    if (core_id != COLLECTOR_CORE_ID) {
        // Wait for the group masks
        cb_wait_front(cb_w2c_in5, 1);

        tile_regs_acquire();

        // Get the adjusted scores
        transpose_wh_init_short(cb_s2c_out);
        transpose_wh_tile(cb_s2c_out, 0, 0);
        cb_pop_front(cb_s2c_out, 1);

        // Get the group masks
        copy_tile_init(cb_w2c_in5);
        copy_tile(cb_w2c_in5, 0, 2);

        // Get top 8 from adjusted scores, and mask them
        top8_tile_init();
        top8_tile(/*tile_index=*/core_id, /*dst_index=*/0);

        cb_pop_front(cb_w2c_in5, 1);
        cb_reserve_back(cb_w2c_in8, 1);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_w2c_in8);
        tile_regs_release();
        cb_push_back(cb_w2c_in8, 1);
    }

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    if (core_id == COLLECTOR_CORE_ID) {
        // I am collecting, let us wait for everyone else to finish sending their data to me
        cb_wait_front(cb_w2c_in4, 1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_w2c_in4);

        // Copy the group scores
        copy_tile(cb_w2c_in4, 0, 0);

        //-------------------------------------------------------------------------
        // Top 4 groups for each token
        //-------------------------------------------------------------------------
        top4_tile_init();
        top4_tile(0);

        // Pack this out for other cores to get the group masks
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_w2c_in5, 1);
        pack_tile</*out_of_order_output=*/true>(0, cb_w2c_in5, /*output_tile_index=*/0);
        tile_regs_release();
        cb_push_back(cb_w2c_in5, 1);

        // Get top 8 from adjusted scores, and mask them
        tile_regs_acquire();
        transpose_wh_init_short(cb_s2c_out);
        transpose_wh_tile(cb_s2c_out, 0, 0);
        cb_pop_front(cb_s2c_out, 1);

        copy_tile_init(cb_w2c_in5);
        copy_tile(cb_w2c_in5, 0, 2);

        top8_tile_init();
        top8_tile(/*tile_index=*/core_id, /*dst_index=*/0);

        cb_pop_front(cb_w2c_in4, 1);

        // Wait for sorted top-8 from all other cores
        cb_wait_front(cb_w2c_in6, 4);

        copy_tile_init(cb_w2c_in6);
        // Tile ID 0 has my own data, so we copy to 1-4
        copy_tile(cb_w2c_in6, 0, 1);
        copy_tile(cb_w2c_in6, 1, 2);
        copy_tile(cb_w2c_in6, 2, 3);
        copy_tile(cb_w2c_in6, 3, 4);

        top8_merge_init();
        top8_merge<column_id>();

        cb_pop_front(cb_w2c_in6, 4);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_s2c_out, 1);
        pack_tile</*out_of_order_output=*/true>(0, cb_s2c_out, /*output_tile_index=*/0);
        cb_push_back(cb_s2c_out, 1);
        tile_regs_release();

        // Let DM1 know that we are done
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);
    }
}
