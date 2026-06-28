// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention v2).
//
// Flash Attention v2 recurrence with online softmax. Per (B,H) work unit:
//   For each Q-block:
//     (Reader pushes m_i=-inf, l_i=0, O_i=0 initial state + Q tiles)
//     For each KV-block:
//       1. S = Q @ K^T  (matmul_block, transpose=true, Q retained)
//       2. S *= scale   (eltwise mul, scalar broadcast)
//       3a. S += mask   (eltwise add, if mask) OR
//       3b. copy S→S_masked (passthrough, if no mask)
//       4. m_blk = rowmax(S_masked)  (reduce MAX REDUCE_ROW, WaitUpfrontNoPop)
//       5. alpha = exp(m_old - m_new)  (eltwise_chain: sub + exp)
//       6. O *= alpha   (eltwise mul, Col broadcast)
//       7. l *= alpha   (eltwise mul)
//       8. S -= m_new   (eltwise sub, Col broadcast, pops S_masked)
//       9. P = exp(S)    (eltwise unary Exp)
//       10. l_blk = rowsum(P)  (reduce SUM REDUCE_ROW)
//       11. l_i += l_blk (eltwise add)
//       12. PV = P @ V → cb_o_accum; O += PV  (matmul + eltwise add)
//       13. m_i = m_new  (eltwise copy)
//     Normalize: O /= l_i  (recip + mul)
//     Copy O → cb_out (for writer to drain to DRAM)
//
// CT args: [has_mask, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
// RT args: [num_work_units]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

// CB indices (match op_design.md CB layout)
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
    constexpr uint32_t has_mask = get_compile_time_arg_val(0);
    constexpr uint32_t B_q_t = get_compile_time_arg_val(1);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(2);
    constexpr uint32_t D_t = get_compile_time_arg_val(3);
    constexpr uint32_t S_q_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t S_kv_tiles = get_compile_time_arg_val(5);

    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;
    constexpr uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    constexpr uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    // --- Boot: engine-wide init + matmul init for QK^T ---
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
    CircularBuffer out_buf(cb_out);

    // Subblock sizing: 2×2 = 4 tiles (DEST-safe with fp32 acc = 8 tile limit).
    constexpr uint32_t sb_h = (B_q_t < 2) ? B_q_t : 2;
    constexpr uint32_t sb_w = (B_kv_t < 2) ? B_kv_t : 2;
    constexpr uint32_t in0_sb = (B_q_t + sb_h - 1) / sb_h;
    constexpr uint32_t in1_sb = (B_kv_t + sb_w - 1) / sb_w;
    constexpr auto qkt_shape = MatmulBlockShape::of(in0_sb, in1_sb, sb_h, sb_w, D_t, 1);

    constexpr uint32_t pv_sb_w = (D_t < 2) ? D_t : 2;
    constexpr uint32_t pv_in1_sb = (D_t + pv_sb_w - 1) / pv_sb_w;
    constexpr auto pv_shape = MatmulBlockShape::of(in0_sb, pv_in1_sb, sb_h, pv_sb_w, B_kv_t, 1);

    uint32_t num_work_units = get_arg_val<uint32_t>(0);

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            // (Reader pushes m_i=-inf, l_i=0, O_i=0 + Q tiles before KV-block loop)

            for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {
                // (Reader pushes scalers + K/V/mask tiles for this KV-block)

                // --- Phase 1: QK^T score matmul ---
                // Q retained (WaitAndRetainOnLastBlock). K consumed.
                matmul_block<
                    /*transpose=*/true,
                    /*packer_l1_acc=*/false,
                    /*last_block_target=*/LastBlockTarget::Out,
                    /*tile_order=*/OutputCBLayout::SubblockMajor,
                    /*init_mode=*/matmul_config::InitMode::Short,
                    /*in0_policy=*/InputPolicy::WaitAndRetainOnLastBlock,
                    /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

                // --- Phase 2: Scale (scalar broadcast, in-place on cb_scores) ---
                mul<cb_scores,
                    cb_scale_factor,
                    cb_scores,
                    BroadcastDim::Scalar,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::Streaming,
                    BinaryDataFormatReconfig::Input,
                    PackTileReconfig::Output,
                    OperandKind::Scalar,
                    OperandKind::Scalar>(num_score_tiles);

                // --- Phase 3: Mask add or passthrough ---
                if constexpr (has_mask) {
                    add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
                } else {
                    copy<cb_scores, cb_scores_masked>(num_score_tiles);
                }

                // --- Phase 4: Row-max (WaitUpfrontNoPop — leaves scores for phase 8) ---
                reduce<
                    PoolType::MAX,
                    ReduceDim::REDUCE_ROW,
                    cb_scores_masked,
                    cb_scaler_reduce,
                    cb_max_new,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
                // Pop MAX scaler (reduce waits but doesn't pop)
                cb_wait_front(cb_scaler_reduce, 1);
                cb_pop_front(cb_scaler_reduce, 1);

                // --- Phase 5: alpha = exp(m_old - m_new) ---
                // cb_max_old: Bulk (wait all, pop at end). cb_max_new: HeldBulk (no pop — for phase 8, 13).
                eltwise_chain(
                    B_q_t,
                    BinaryFpu<
                        cb_max_old,
                        cb_max_new,
                        BinaryFpuOp::Sub,
                        BroadcastDim::None,
                        InputLifecycle::Bulk,
                        InputLifecycle::HeldBulk,
                        BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Block,
                        OperandKind::Block>{},
                    Exp<>{},
                    PackTile<cb_alpha, OutputLifecycle::Streaming>{});

                // --- Phase 6: O *= alpha (Col broadcast, in-place on cb_o) ---
                // Block+Bulk for cb_o (wait all, pop at end). Col+HeldBulk for cb_alpha (no pop — for phase 7).
                // Output: BulkReservePerTile (reserve per tile, push at end).
                mul<cb_o,
                    cb_alpha,
                    cb_o,
                    BroadcastDim::Col,
                    InputLifecycle::Bulk,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::BulkReservePerTile,
                    BinaryDataFormatReconfig::Input,
                    PackTileReconfig::Output,
                    OperandKind::Block,
                    OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

                // --- Phase 7: l *= alpha (in-place on cb_sum_old, both Streaming) ---
                mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);

                // --- Phase 8: S -= m_new (Col broadcast, in-place on cb_scores_masked) ---
                // Block+Bulk (wait all, pop at end — pops what WaitUpfrontNoPop left).
                // Col+HeldBulk for cb_max_new (no pop — for phase 13).
                sub<cb_scores_masked,
                    cb_max_new,
                    cb_scores_masked,
                    BroadcastDim::Col,
                    InputLifecycle::Bulk,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::BulkReservePerTile,
                    BinaryDataFormatReconfig::Input,
                    PackTileReconfig::Output,
                    OperandKind::Block,
                    OperandKind::Col>(EltwiseShape::grid(B_q_t, B_kv_t));

                // --- Phase 9: P = exp(S - m_new) ---
                unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);

                // --- Phase 10: l_blk = rowsum(P) ---
                reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_ROW,
                    cb_exp_scores,
                    cb_scaler_reduce,
                    cb_sum_new,
                    ReduceInputPolicy::WaitAndPopPerTile>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
                // Pop SUM scaler
                cb_wait_front(cb_scaler_reduce, 1);
                cb_pop_front(cb_scaler_reduce, 1);

                // --- Phase 11: l_i += l_blk (in-place, both Streaming) ---
                add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);

                // --- Phase 12: PV = P @ V → cb_o_accum, then O += PV ---
                matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/false,
                    /*last_block_target=*/LastBlockTarget::Out,
                    /*tile_order=*/OutputCBLayout::SubblockMajor,
                    /*init_mode=*/matmul_config::InitMode::Short,
                    /*in0_policy=*/InputPolicy::WaitAndPopPerKBlock,
                    /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(
                    exp_scores_buf, v_buf, o_accum_buf, o_accum_buf, pv_shape);
                add<cb_o, cb_o_accum, cb_o>(num_o_tiles);

                // --- Phase 13: m_i = m_new (Streaming: pop max_new, push max_old) ---
                copy<cb_max_new, cb_max_old>(B_q_t);

            }  // end KV-block loop

            // --- Phase 14: Normalize and write output ---
            // recip(l_i) in-place on cb_sum_old
            unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);

            // O *= recip(l_i) → cb_out (Col broadcast)
            // Block+Bulk for cb_o. Col+HeldBulk for cb_sum_old.
            // Output: cb_out, Streaming.
            mul<cb_o,
                cb_sum_old,
                cb_out,
                BroadcastDim::Col,
                InputLifecycle::Bulk,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Block,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

            // --- Pop running state and Q for next Q-block ---
            // m_i: was popped by phase 5 Bulk in the last KV-block
            // l_i: HeldBulk in phase 14 didn't pop — pop now
            cb_wait_front(cb_sum_old, B_q_t);
            cb_pop_front(cb_sum_old, B_q_t);
            // O_i: was popped by phase 14 mul (Bulk) — but output went to cb_out, not cb_o
            // Actually, phase 14 mul with InputLifecycle::Bulk pops cb_o at end.
            // So cb_o is empty. But we need to verify this.
            // Q: retained — pop now
            cb_wait_front(cb_q, num_o_tiles);
            cb_pop_front(cb_q, num_o_tiles);

        }  // end Q-block loop
    }  // end work-unit loop
}
