// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention).
//
// CT args: [has_mask, H_q, H_kv]
// (B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles are runtime args — read from RT args[0..4])
// Wait — compute kernel doesn't have RT args in the current descriptor.
// Let me pass them as CT args instead.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_mask = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_reduce = 4;
constexpr uint32_t cb_scale_factor = 5;
constexpr uint32_t cb_alpha = 8;
constexpr uint32_t cb_o = tt::CBIndex::c_16;
constexpr uint32_t cb_out = tt::CBIndex::c_17;
constexpr uint32_t cb_scores = 24;
constexpr uint32_t cb_scores_masked = 25;
constexpr uint32_t cb_max_new = 26;
constexpr uint32_t cb_max_old = 27;
constexpr uint32_t cb_exp_scores = 28;
constexpr uint32_t cb_sum_new = 29;
constexpr uint32_t cb_sum_old = 30;
constexpr uint32_t cb_o_accum = 31;

void kernel_main() {
    // CT args: [has_mask, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles, H_q, H_kv]
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t B_q_t = get_compile_time_arg_val(1);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(2);
    constexpr uint32_t D_t = get_compile_time_arg_val(3);
    constexpr uint32_t S_q_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t S_kv_tiles = get_compile_time_arg_val(5);

    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;
    constexpr uint32_t num_q_tiles = B_q_t * D_t;

    constexpr uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    constexpr uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    // --- Boot ---
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1, /*ct_dim=*/B_kv_t, /*rt_dim=*/B_q_t, /*kt_dim=*/D_t);

    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores);
    CircularBuffer scores_masked_buf(cb_scores_masked);
    CircularBuffer exp_scores_buf(cb_exp_scores);
    CircularBuffer o_buf(cb_o);
    CircularBuffer o_accum_buf(cb_o_accum);

    // Subblock sizing: keep within DEST limits (8 tiles with fp32_acc)
    // 2x2 subblock = 4 tiles per subblock
    constexpr uint32_t sb_h = (B_q_t < 2) ? B_q_t : 2;
    constexpr uint32_t sb_w = (B_kv_t < 2) ? B_kv_t : 2;
    constexpr uint32_t in0_sb = (B_q_t + sb_h - 1) / sb_h;
    constexpr uint32_t in1_sb = (B_kv_t + sb_w - 1) / sb_w;

    constexpr auto qkt_shape = MatmulBlockShape::of(in0_sb, in1_sb, sb_h, sb_w, D_t, 1);
    constexpr uint32_t pv_sb_w = (D_t < 2) ? D_t : 2;
    constexpr uint32_t pv_in1_sb = (D_t + pv_sb_w - 1) / pv_sb_w;
    constexpr auto pv_shape = MatmulBlockShape::of(in0_sb, pv_in1_sb, sb_h, pv_sb_w, B_kv_t, 1);

    for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
        // --- Phase 1: QK^T score matmul ---
        matmul_block<
            /*transpose=*/true, /*packer_l1_acc=*/false,
            /*last_block_target=*/LastBlockTarget::Out,
            /*tile_order=*/OutputCBLayout::SubblockMajor,
            /*init_mode=*/matmul_config::InitMode::Short,
            /*in0_policy=*/InputPolicy::WaitAndRetainOnLastBlock,
            /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(
            q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

        // --- Phase 2: Scale ---
        mul<cb_scores, cb_scale_factor, cb_scores,
            BroadcastDim::Scalar, InputLifecycle::Streaming, InputLifecycle::HeldBulk>(num_score_tiles);

        // --- Phase 3: Mask or passthrough ---
        if constexpr (has_mask) {
            add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
        } else {
            copy<cb_scores, cb_scores_masked>(num_score_tiles);
        }

        // --- Phase 4: Row-max ---
        reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores_masked, cb_scaler_reduce, cb_max_new,
               ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
        cb_wait_front(cb_scaler_reduce, 1);
        cb_pop_front(cb_scaler_reduce, 1);

        // --- Phase 5: alpha = exp(m_old - m_new) ---
        eltwise_chain(
            B_q_t,
            BinaryFpu<cb_max_old, cb_max_new, BinaryFpuOp::Sub, BroadcastDim::None,
                      InputLifecycle::Bulk, InputLifecycle::HeldBulk,
                      BinaryDataFormatReconfig::Input, Dst::D0,
                      OperandKind::Block, OperandKind::Block>{},
            Exp<>{},
            PackTile<cb_alpha, OutputLifecycle::Streaming>{});

        // --- Phase 6: O *= alpha (Col broadcast) ---
        mul<cb_o, cb_alpha, cb_o, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar, OperandKind::Col>(
            EltwiseShape::grid(B_q_t, D_t));

        // --- Phase 7: l *= alpha ---
        mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);

        // --- Phase 8: S -= m_new (Col broadcast) ---
        sub<cb_scores_masked, cb_max_new, cb_scores_masked, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar, OperandKind::Col>(
            EltwiseShape::grid(B_q_t, B_kv_t));

        // --- Phase 9: P = exp(S) ---
        unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);

        // --- Phase 10: rowsum(P) ---
        reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_exp_scores, cb_scaler_reduce, cb_sum_new,
               ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
        cb_wait_front(cb_scaler_reduce, 1);
        cb_pop_front(cb_scaler_reduce, 1);

        // --- Phase 11: l_i += l_blk ---
        // Use Block OperandKind with Bulk lifecycle for both inputs.
        // Block reads tile i (absolute front index), Bulk waits for all M tiles upfront
        // and pops at end. Output to cb_sum_old in-place with Bulk (reserve upfront,
        // push at end). With Block+Bulk, the chain waits for B_q_t tiles on both
        // inputs, processes all, pops B_q_t from both, reserves B_q_t output, pushes B_q_t.
        // In-place (CbA==CbOut): the reserve(B_q_t) would deadlock since B_q_t tiles
        // are at the front (not yet popped). So use DeferredPop for input A:
        // caller (Bulk) waited, chain pops at end. But in-place needs pop before reserve.
        // The chain pops inputs AFTER compute, BEFORE output reserve? No, the chain
        // order is: wait_inputs -> compute -> pop_inputs -> reserve_outputs -> pack -> push_outputs.
        // So pop happens before reserve. With DeferredPop, the chain doesn't pop (caller does).
        // Let me use DeferredPop for input A, Streaming for input B, Streaming output.
        // Actually, simplest: use DeferredPop for A (cb_sum_old, already waited by phase 7),
        // Streaming for B (cb_sum_new), Streaming output to cb_sum_old.
        // Wait, DeferredPop means PopPolicy::AtEnd with no wait. But who waited?
        // Let me just use Streaming+Block for both, Streaming output.
        // Block + Streaming: M = B_q_t, waits per tile (i+1 cumulative), pops per tile.
        // Wait — is Block + Streaming legal? Checking: Block rejects PerTile-pop!
        // "Block walks the absolute CB-front index, so it rejects PerTile-pop"
        // So Block + Streaming is ILLEGAL.
        // Use Scalar + Streaming (default) with non-in-place output.
        // Output to cb_o_accum (empty after phase 12 of PREVIOUS iteration).
        // But phase 12 hasn't run yet in THIS iteration. cb_o_accum is empty (initial state).
        // Wait, for the FIRST KV-block, cb_o_accum is empty (never written to).
        // For subsequent KV-blocks, phase 12 wrote to it and phase 12's add consumed it.
        // So cb_o_accum should be empty. Let me output there, then copy back.
        // Actually, this is getting too complex. Let me just use the simplest approach:
        // non-in-place output to a temp CB, then copy.
        // Use cb_exp_scores as temp — BUT phase 12 needs it!
        // Use cb_max_new as temp — BUT phase 13 needs it!
        // OK, let me just output to cb_sum_old directly with Streaming + Scalar (the original approach)
        // and figure out why it's hanging. The chain with Scalar + Streaming should:
        // per tile: wait_front(1), compute, pop_front(1), reserve_back(1), pack, push_back(1).
        // Pop before reserve — should work.
        add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);

        // --- Phase 12: PV = P @ V, then O += PV ---
        matmul_block<
            /*transpose=*/false, /*packer_l1_acc=*/false,
            /*last_block_target=*/LastBlockTarget::Out,
            /*tile_order=*/OutputCBLayout::SubblockMajor,
            /*init_mode=*/matmul_config::InitMode::Short,
            /*in0_policy=*/InputPolicy::WaitAndPopPerKBlock,
            /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(
            exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);
        add<cb_o, cb_o_accum, cb_o>(num_o_tiles);

        // --- Phase 13: m_i = m_new ---
        copy<cb_max_new, cb_max_old>(B_q_t);

        // Pop Q tiles for next Q-block
        if (qb < num_q_blocks - 1) {
            cb_wait_front(cb_q, num_q_tiles);
            cb_pop_front(cb_q, num_q_tiles);
        }

        // --- Phase 14: Normalize and write output ---
        // recip(l_i) in-place
        unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);
        // O *= recip(l_i) → cb_out
        mul<cb_o, cb_sum_old, cb_out, BroadcastDim::Col,
            InputLifecycle::Streaming, InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output, OperandKind::Scalar, OperandKind::Col>(
            EltwiseShape::grid(B_q_t, D_t));
    }
}
