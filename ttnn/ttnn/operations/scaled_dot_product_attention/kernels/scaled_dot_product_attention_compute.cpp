// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention v2).
//
// Flash Attention recurrence with online softmax. Per (B,H) work unit:
//   For each Q-block:
//     Init m_i=-inf, l_i=0, O_i=0
//     For each KV-block:
//       1. S = Q @ K^T         (matmul_block, transpose=true, Q retained)
//       2. S *= scale           (eltwise mul, Scalar broadcast)
//       3. S_masked = S + mask  (eltwise add) or copy S→S_masked (no mask)
//       4. m_blk = rowmax(S_masked)  (reduce MAX REDUCE_ROW, WaitUpfrontNoPop)
//       5. alpha = exp(m_old - m_new)  (eltwise_chain: sub + exp)
//       6. O *= alpha            (eltwise mul, Col broadcast, in-place)
//       7. l *= alpha            (eltwise mul, in-place)
//       8. S_masked -= m_new     (eltwise sub, Col broadcast, pops S_masked)
//       9. P = exp(S_masked)     (eltwise unary Exp)
//       10. l_blk = rowsum(P)    (reduce SUM REDUCE_ROW, WaitUpfrontNoPop)
//       11. l += l_blk           (eltwise add, in-place)
//       12. O += P @ V           (matmul_block → cb_o_accum, then eltwise add into cb_o)
//       13. m = m_new            (eltwise copy)
//     Normalize: O /= l          (recip + mul)
//     Write O to cb_out
//
// CT args: [B_q_t, B_kv_t, D_t, has_mask, S_q_tiles, S_kv_tiles]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/device_print.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

// Debug helper: print the number of valid tiles in a CB (received - acked).
ALWI void dbg_cb_count(const char* name, uint32_t cb_id) {
    uint32_t recv = get_cb_tiles_received_ptr(cb_id)[0];
    uint32_t ack = get_cb_tiles_acked_ptr(cb_id)[0];
    DEVICE_PRINT("CB {}: id={} tiles={}-{}={}\n", name, (uint32_t)cb_id, recv, ack, recv - ack);
}

ALWI void dbg_phase(const char* phase) {
    DEVICE_PRINT("===== {} =====\n", phase);
    dbg_cb_count("cb_q", cb_q);
    dbg_cb_count("cb_k", cb_k);
    dbg_cb_count("cb_v", cb_v);
    dbg_cb_count("cb_scaler_reduce", cb_scaler_reduce);
    dbg_cb_count("cb_scale_factor", cb_scale_factor);
    dbg_cb_count("cb_alpha", cb_alpha);
    dbg_cb_count("cb_o", cb_o);
    dbg_cb_count("cb_scores", cb_scores);
    dbg_cb_count("cb_scores_masked", cb_scores_masked);
    dbg_cb_count("cb_max_new", cb_max_new);
    dbg_cb_count("cb_max_old", cb_max_old);
    dbg_cb_count("cb_exp_scores", cb_exp_scores);
    dbg_cb_count("cb_sum_new", cb_sum_new);
    dbg_cb_count("cb_sum_old", cb_sum_old);
    dbg_cb_count("cb_o_accum", cb_o_accum);
}

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
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);
    constexpr uint32_t has_mask = get_compile_time_arg_val(3);
    constexpr uint32_t S_q_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t S_kv_tiles = get_compile_time_arg_val(5);

    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;
    constexpr uint32_t num_o_tiles = B_q_t * D_t;

    constexpr uint32_t num_q_blocks = (S_q_tiles + B_q_t - 1) / B_q_t;
    constexpr uint32_t num_kv_blocks = (S_kv_tiles + B_kv_t - 1) / B_kv_t;

    // --- Boot: engine-wide init + matmul init for QK^T ---
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1,
                  /*ct_dim=*/2, /*rt_dim=*/2, /*kt_dim=*/D_t);

    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer v_buf(cb_v);
    CircularBuffer mask_buf(cb_mask);
    CircularBuffer scaler_reduce_buf(cb_scaler_reduce);
    CircularBuffer scale_factor_buf(cb_scale_factor);
    CircularBuffer alpha_buf(cb_alpha);
    CircularBuffer o_buf(cb_o);
    CircularBuffer out_buf(cb_out);
    CircularBuffer scores_buf(cb_scores);
    CircularBuffer scores_masked_buf(cb_scores_masked);
    CircularBuffer max_new_buf(cb_max_new);
    CircularBuffer max_old_buf(cb_max_old);
    CircularBuffer exp_scores_buf(cb_exp_scores);
    CircularBuffer sum_new_buf(cb_sum_new);
    CircularBuffer sum_old_buf(cb_sum_old);
    CircularBuffer o_accum_buf(cb_o_accum);

    // Matmul shapes:
    // QK^T: M=B_q_t(4), N=B_kv_t(4), K=D_t. Subblock 2x2 = 4 tiles (DEST-safe with fp32 acc).
    constexpr auto qkt_shape = MatmulBlockShape::of(2, 2, 2, 2, D_t, 1);
    // PV: M=B_q_t(4), N=D_t, K=B_kv_t(4). Subblock 2x2 (or 2x1 if D_t=1).
    constexpr uint32_t pv_sb_w = (D_t < 2) ? D_t : 2;
    constexpr uint32_t pv_in1_subblocks = (D_t + pv_sb_w - 1) / pv_sb_w;
    constexpr auto pv_shape = MatmulBlockShape::of(2, pv_in1_subblocks, 2, pv_sb_w, B_kv_t, 1);

    // Read runtime args: [num_work_units]
    uint32_t num_work_units = get_arg_val<uint32_t>(0);

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        for (uint32_t qb = 0; qb < num_q_blocks; ++qb) {
            // --- Phase 0: Wait for initial running state ---
            // Reader pushes m_i(-inf), l_i(0), O_i(0) tiles before each Q-block.
            cb_wait_front(cb_max_old, B_q_t);
            cb_wait_front(cb_sum_old, B_q_t);
            cb_wait_front(cb_o, num_o_tiles);
            cb_wait_front(cb_q, num_o_tiles);
            dbg_phase("Phase 0: after init wait");

            // --- Inner loop over KV-blocks ---
            for (uint32_t kvb = 0; kvb < num_kv_blocks; ++kvb) {

                dbg_phase("Phase 1: before QK^T");
                // --- Phase 1: QK^T score matmul ---
                // Q retained (WaitAndRetainOnLastBlock) for reuse across KV-blocks.
                matmul_block<
                    /*transpose=*/true,
                    /*packer_l1_acc=*/false,
                    /*last_block_target=*/LastBlockTarget::Out,
                    /*tile_order=*/OutputCBLayout::SubblockMajor,
                    /*init_mode=*/matmul_config::InitMode::Short,
                    /*in0_policy=*/InputPolicy::WaitAndRetainOnLastBlock,
                    /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(
                    q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

                dbg_phase("Phase 2: before scale");
                // --- Phase 2: Scale scores ---
                mul<cb_scores, cb_scale_factor, cb_scores,
                    BroadcastDim::Scalar,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk>(num_score_tiles);

                dbg_phase("Phase 3: before mask/passthrough");
                // --- Phase 3: Mask add or passthrough ---
                if constexpr (has_mask) {
                    add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles);
                } else {
                    copy<cb_scores, cb_scores_masked>(num_score_tiles);
                }

                dbg_phase("Phase 4: before row-max reduce");
                // --- Phase 4: Row-max (WaitUpfrontNoPop — leaves scores for phase 8) ---
                reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
                       cb_scores_masked, cb_scaler_reduce, cb_max_new,
                       ReduceInputPolicy::WaitUpfrontNoPop>(
                    ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
                // Reduce doesn't pop scaler; pop MAX scaler for SUM scaler access
                cb_pop_front(cb_scaler_reduce, 1);

                dbg_phase("Phase 5: before alpha=exp(m_old-m_new)");
                // --- Phase 5: alpha = exp(m_old - m_new) ---
                eltwise_chain(
                    B_q_t,
                    BinaryFpu<cb_max_old, cb_max_new, BinaryFpuOp::Sub, BroadcastDim::None,
                              InputLifecycle::Bulk, InputLifecycle::HeldBulk,
                              BinaryDataFormatReconfig::Input, Dst::D0,
                              OperandKind::Scalar, OperandKind::Scalar>{},
                    Exp<>{},
                    PackTile<cb_alpha, OutputLifecycle::Streaming>{});

                dbg_phase("Phase 6: before O*=alpha");
                // --- Phase 6: O *= alpha (Col broadcast, in-place) ---
                mul<cb_o, cb_alpha, cb_o,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::Streaming,
                    BinaryDataFormatReconfig::Input,
                    PackTileReconfig::Output,
                    OperandKind::Scalar,
                    OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

                dbg_phase("Phase 7: before l*=alpha");
                // --- Phase 7: l *= alpha (in-place, element-wise) ---
                mul<cb_sum_old, cb_alpha, cb_sum_old>(B_q_t);

                dbg_phase("Phase 8: before S-=m_new");
                // --- Phase 8: S_masked -= m_new (Col broadcast, in-place) ---
                sub<cb_scores_masked, cb_max_new, cb_scores_masked,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::Streaming,
                    BinaryDataFormatReconfig::Input,
                    PackTileReconfig::Output,
                    OperandKind::Scalar,
                    OperandKind::Col>(EltwiseShape::grid(B_q_t, B_kv_t));

                dbg_phase("Phase 9: before P=exp(S)");
                // --- Phase 9: P = exp(S_masked) ---
                unary<Exp<>, cb_scores_masked, cb_exp_scores>(num_score_tiles);

                dbg_phase("Phase 10: before rowsum(P)");
                // --- Phase 10: l_blk = rowsum(P) (WaitUpfrontNoPop for PV matmul) ---
                reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
                       cb_exp_scores, cb_scaler_reduce, cb_sum_new,
                       ReduceInputPolicy::WaitUpfrontNoPop>(
                    ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));
                cb_pop_front(cb_scaler_reduce, 1);

                dbg_phase("Phase 11: before l+=l_blk (HANG POINT)");
                // --- Phase 11: l += l_blk (in-place) ---
                add<cb_sum_old, cb_sum_new, cb_sum_old>(B_q_t);

                // --- Phase 12: PV matmul → cb_o_accum, then O += PV ---
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

                // --- Phase 13: m = m_new ---
                copy<cb_max_new, cb_max_old>(B_q_t);

            } // end KV-block loop

            // --- Phase 14: Normalize O by l_i and write to output ---
            unary<Recip<>, cb_sum_old, cb_sum_old>(B_q_t);

            mul<cb_o, cb_sum_old, cb_out,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

            // Pop consumed state CBs for this Q-block
            cb_pop_front(cb_max_old, B_q_t);
            cb_pop_front(cb_sum_old, B_q_t);
            cb_pop_front(cb_q, num_o_tiles);

        } // end Q-block loop
    } // end work-unit loop
}
