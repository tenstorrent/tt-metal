// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for Flash-Attention scaled_dot_product_attention.
//
// Per Q-block (b, h_q, q_chunk): pre-scale Q, then loop KV-blocks running the
// online-softmax recurrence (running max m / sum l / output O), normalize, emit.
//
// All compute phases use kernel_lib helpers (matmul_block / reduce / eltwise_chain).
// Persistent-CB pops are folded into the eltwise_chain operand lifecycles:
//   - m_run is popped by the correction chain (its last read),
//   - l_run by the l-update chain, o_run by the O-update chain,
//   - corr by the O-update chain (its last read),
// so commit just copies new->run into the freshly-emptied persistent CBs.
//
// Advisory deviations from op_design.md:
//   * cb_masked (idx 15) added: custom mask is applied non-in-place (cb_scores +
//     cb_mask -> cb_masked) to avoid an in-place block-alias 2x-size deadlock.
//   * m_new merge uses the binary_sfpu convenience wrapper (BinaryMax) instead of
//     a raw eltwise_chain; identical semantics.
//   * Pre-scale Q is done NON-in-place (cb_q -> cb_qs) instead of transform_in_place:
//     the in-place same-CB read+write deadlocks (reserve-before-pop on a full CB).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t start_qb = get_arg_val<uint32_t>(0);
    uint32_t num_qb = get_arg_val<uint32_t>(1);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t H_q = get_compile_time_arg_val(1);
    constexpr uint32_t H_kv = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_t = get_compile_time_arg_val(3);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(4);
    constexpr uint32_t Dt = get_compile_time_arg_val(5);
    constexpr uint32_t q_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t kv_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t q_blocks_per_bh = get_compile_time_arg_val(8);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t gqa_factor = get_compile_time_arg_val(10);
    constexpr uint32_t mask_mode = get_compile_time_arg_val(11);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(12);
    constexpr uint32_t skv_non_aligned = get_compile_time_arg_val(13);
    // mask_mode: 0 none, 1 custom (DRAM tensor), 2 causal (generated on-device).
    // Both custom and causal feed the additive mask-add -> softmax path.
    constexpr bool has_mask = (mask_mode >= 1);
    constexpr bool is_causal = (mask_mode == 2);

    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_mask = 3;
    constexpr uint32_t cb_qs = 4;  // scaled Q (matmul in0)
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;
    constexpr uint32_t cb_l_new = 10;
    constexpr uint32_t cb_pv = 11;
    constexpr uint32_t cb_o_run = 12;
    constexpr uint32_t cb_o_new = 13;
    constexpr uint32_t cb_l_inv = 14;
    constexpr uint32_t cb_masked = 15;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_scores = 24;
    constexpr uint32_t cb_probs = 25;
    constexpr uint32_t cb_m_cur = 26;
    constexpr uint32_t cb_m_run = 27;
    constexpr uint32_t cb_m_new = 28;
    constexpr uint32_t cb_corr = 29;
    constexpr uint32_t cb_l_cur = 30;
    constexpr uint32_t cb_l_run = 31;

    // Score source fed to reduce-max + softmax: masked block when a mask is present.
    constexpr uint32_t cb_sc = has_mask ? cb_masked : cb_scores;

    // CircularBuffer objects for the matmul helper.
    CircularBuffer qs_buf(cb_qs), k_buf(cb_k), v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores), probs_buf(cb_probs), pv_buf(cb_pv);

    // Boot: hw_configure (matmul source order) + one matmul-block init.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_qs, cb_k, cb_scores);
    mm_block_init(cb_qs, cb_k, cb_scores, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/Dt);

    for (uint32_t qi = 0; qi < num_qb; ++qi) {
        const uint32_t qb = start_qb + qi;
        const uint32_t qci = qb % q_blocks_per_bh;

        const uint32_t q_row0 = qci * q_chunk_t;
        uint32_t q_cnt = q_chunk_t;
        if (q_row0 + q_cnt > Sq_t) {
            q_cnt = Sq_t - q_row0;
        }

        // Phase 0: pre-scale Q (Q *= scale) into cb_qs (non-in-place; see header note).
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(q_cnt * Dt),
            ckl::CopyTile<cb_q, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
            ckl::MulUnary<>{scale_bits},
            ckl::PackTile<cb_qs, ckl::OutputLifecycle::Streaming>{});

        // Causal: process only KV blocks up to and including the diagonal block
        // for this Q-block (must match the reader's identical cap). Later blocks
        // are fully -inf and are skipped entirely.
        uint32_t kv_blocks_this_q = num_kv_blocks;
        if constexpr (is_causal) {
            kv_blocks_this_q = (q_row0 + q_cnt - 1) / kv_chunk_t + 1;
            if (kv_blocks_this_q > num_kv_blocks) {
                kv_blocks_this_q = num_kv_blocks;
            }
        }

        for (uint32_t j = 0; j < kv_blocks_this_q; ++j) {
            const uint32_t kv_row0 = j * kv_chunk_t;
            uint32_t kv_cnt = kv_chunk_t;
            if (kv_row0 + kv_cnt > Skv_t) {
                kv_cnt = Skv_t - kv_row0;
            }
            const bool first = (j == 0);

            // Phase 1: QKᵀ -> cb_scores  [q_cnt x kv_cnt]. Q retained across KV loop.
            ckl::matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                qs_buf, k_buf, scores_buf, scores_buf, ckl::MatmulBlockShape::of(q_cnt, kv_cnt, 1, 1, Dt, 1));

            // Phase 2: mask add (custom only) -> cb_masked.
            if constexpr (has_mask) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(q_cnt, kv_cnt),
                    ckl::BinaryFpu<
                        cb_scores,
                        cb_mask,
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::Bulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block>{},
                    ckl::PackTile<cb_masked, ckl::OutputLifecycle::Streaming>{});
            }

            // Phase 3: block row-max -> cb_m_cur (keep cb_sc resident for the P chain).
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sc,
                cb_scaler_max,
                cb_m_cur,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(q_cnt, kv_cnt));

            // Phase 4: m_new = max(m_run, m_cur)  (j>0 only; j==0 uses m_cur directly).
            if (!first) {
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    cb_m_run,
                    cb_m_cur,
                    cb_m_new,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::InputLifecycle::Bulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Block>(q_cnt);
            }

            // Phase 5: P = exp(scores - m).  Broadcast the per-row max (col vector) across keys.
            if (first) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(q_cnt, kv_cnt),
                    ckl::BinaryFpu<
                        cb_sc,
                        cb_m_cur,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col>{},
                    ckl::Exp<>{},
                    ckl::PackTile<cb_probs, ckl::OutputLifecycle::Streaming>{});
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(q_cnt, kv_cnt),
                    ckl::BinaryFpu<
                        cb_sc,
                        cb_m_new,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col>{},
                    ckl::Exp<>{},
                    ckl::PackTile<cb_probs, ckl::OutputLifecycle::Streaming>{});
            }

            // Phase 6: block row-sum -> cb_l_cur (keep cb_probs for the PV matmul).
            // When S_kv is non-aligned, the last kv-tile of the LAST KV block
            // carries zero-padded key columns; the partial SUM scaler zeroes them
            // so they don't leak into the softmax row-sum. Non-last blocks (and the
            // aligned case) use the full scaler at tile 0 via none().
            const bool last_kv = (j == num_kv_blocks - 1);
            const auto sum_partial = (skv_non_aligned && last_kv) ? ckl::ReducePartialScaler::last_tile_at(1)
                                                                  : ckl::ReducePartialScaler::none();
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_probs,
                cb_scaler_sum,
                cb_l_cur,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                ckl::ReduceInputBlockShape::of(q_cnt, kv_cnt),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                ckl::NoOp{},
                sum_partial);

            if (!first) {
                // Phase 7: corr = exp(m_run - m_new). Pops old m_run; keeps m_new for commit.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(q_cnt),
                    ckl::BinaryFpu<
                        cb_m_run,
                        cb_m_new,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block>{},
                    ckl::Exp<>{},
                    ckl::PackTile<cb_corr, ckl::OutputLifecycle::Streaming>{});

                // Phase 8: l_new = corr * l_run + l_cur. Pops old l_run + l_cur; keeps corr.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(q_cnt),
                    ckl::BinaryFpu<
                        cb_corr,
                        cb_l_run,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::InputLifecycle::Bulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Block>{},
                    ckl::DestReuseBinary<
                        cb_l_cur,
                        ckl::BinaryFpuOp::Add,
                        ckl::DestReuseType::DEST_TO_SRCA,
                        ckl::InputLifecycle::Bulk,
                        ckl::DestReuseReconfig::Input,
                        ckl::Dst::D0,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block>{},
                    ckl::PackTile<cb_l_new, ckl::OutputLifecycle::Streaming>{});
            }

            // Phase 9: PV = P · V -> cb_pv  [q_cnt x Dt]. Pops cb_probs, cb_v.
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                probs_buf, v_buf, pv_buf, pv_buf, ckl::MatmulBlockShape::of(q_cnt, Dt, 1, 1, kv_cnt, 1));

            if (!first) {
                // Phase 10: O_new = corr * O_run + PV. Pops old o_run + corr + pv.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(q_cnt, Dt),
                    ckl::BinaryFpu<
                        cb_o_run,
                        cb_corr,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::Bulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        ckl::OperandKind::Col>{},
                    ckl::DestReuseBinary<
                        cb_pv,
                        ckl::BinaryFpuOp::Add,
                        ckl::DestReuseType::DEST_TO_SRCA,
                        ckl::InputLifecycle::Bulk,
                        ckl::DestReuseReconfig::Input,
                        ckl::Dst::D0,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block>{},
                    ckl::PackTile<cb_o_new, ckl::OutputLifecycle::Streaming>{});

                // Phase 11: commit new -> run (persistent CBs freshly emptied above).
                ckl::copy<cb_m_new, cb_m_run>(q_cnt);
                ckl::copy<cb_l_new, cb_l_run>(q_cnt);
                ckl::copy<cb_o_new, cb_o_run>(q_cnt * Dt);
            } else {
                // Phase 11 (j==0): block results become the running state directly.
                ckl::copy<cb_m_cur, cb_m_run>(q_cnt);
                ckl::copy<cb_l_cur, cb_l_run>(q_cnt);
                ckl::copy<cb_pv, cb_o_run>(q_cnt * Dt);
            }
        }  // KV loop

        // Phase 12: 1/l. Phase 13: normalize O = O_run / l -> cb_out.
        ckl::unary<ckl::Recip<>, cb_l_run, cb_l_inv>(q_cnt);
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(q_cnt, Dt),
            ckl::BinaryFpu<
                cb_o_run,
                cb_l_inv,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Col,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::Bulk,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Block,
                ckl::OperandKind::Col>{},
            ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming>{});

        // Release the retained scaled-Q block for this Q-block.
        cb_pop_front(cb_qs, q_cnt * Dt);
        // Drain the running-max CB: normalize consumed l_run (recip) and o_run
        // (final mul), but NOT m_run — without this pop it leaks into the next
        // Q-block handled by this same core (multi-Q-block-per-core corruption).
        cb_pop_front(cb_m_run, q_cnt);
    }
}
