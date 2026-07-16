// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (FlashAttention-2).
//
// Per work unit (one q-chunk) the running online-softmax state lives in three
// resident fp32 CBs — cb_row_max (m), cb_row_sum (l), cb_out_accum (O) — and is
// folded over the KV loop. The S_q x S_kv score matrix is never materialized:
// cb_scores / cb_exp are sized to one Sq_chunk_t x Skv_chunk_t block.
//
// Every compute phase goes through a kernel_lib helper (matmul_block, reduce,
// eltwise mul/sub/add/copy, binary_sfpu<BinaryMax>, unary<Exp>/<Recip>). The
// first KV chunk initializes (m, l, O) directly (m_old = -inf => alpha = 0),
// matching the recurrence with the correction sub-steps elided.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_mask_in = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_scale = 5;
constexpr uint32_t cb_m_new = 6;
constexpr uint32_t cb_sum_chunk = 7;
constexpr uint32_t cb_kv_mask = 8;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_q_scaled = 24;
constexpr uint32_t cb_scores = 25;
constexpr uint32_t cb_exp = 26;
constexpr uint32_t cb_row_max = 27;
constexpr uint32_t cb_row_sum = 28;
constexpr uint32_t cb_pv = 29;
constexpr uint32_t cb_out_accum = 30;
constexpr uint32_t cb_corr = 31;
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t n_kv_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t has_mask_v = get_compile_time_arg_val(4);
    constexpr bool has_mask = has_mask_v != 0;
    constexpr uint32_t qk_in0_sb = get_compile_time_arg_val(5);
    constexpr uint32_t qk_in1_sb = get_compile_time_arg_val(6);
    constexpr uint32_t qk_out_sb_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_out_sb_w = get_compile_time_arg_val(8);
    constexpr uint32_t pv_in0_sb = get_compile_time_arg_val(9);
    constexpr uint32_t pv_in1_sb = get_compile_time_arg_val(10);
    constexpr uint32_t pv_out_sb_h = get_compile_time_arg_val(11);
    constexpr uint32_t pv_out_sb_w = get_compile_time_arg_val(12);
    constexpr uint32_t skv_partial = get_compile_time_arg_val(13);  // valid cols in last S_kv tile (0 => aligned)
    constexpr bool has_kv_pad = skv_partial != 0;

    const uint32_t num_wu = get_arg_val<uint32_t>(0);

    constexpr uint32_t sq_dt = Sq_chunk_t * Dt;
    constexpr uint32_t sq_skv = Sq_chunk_t * Skv_chunk_t;

    using ckl::BroadcastDim;
    using ckl::Dst;
    using ckl::EltwiseShape;
    using ckl::InputLifecycle;
    using ckl::LastBlockTarget;
    using ckl::MatmulBlockShape;
    using ckl::OperandKind;
    using ckl::OutputCBLayout;
    using ckl::OutputLifecycle;
    using ckl::ReduceInputBlockShape;
    using ckl::ReduceInputPolicy;

    // CircularBuffer wrappers for the two matmuls.
    ::CircularBuffer q_scaled_buf(cb_q_scaled), k_buf(cb_k_in), scores_buf(cb_scores);
    ::CircularBuffer exp_buf(cb_exp), v_buf(cb_v_in), pv_buf(cb_pv);

    // Boot: matmul-order hw config + one full matmul_block_init (helper's Short
    // init restores state after intervening eltwise/reduce ops per call).
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q_scaled, cb_k_in, cb_scores);
    matmul_block_init(cb_q_scaled, cb_k_in, /*transpose=*/1, qk_out_sb_w, qk_out_sb_h, Dt);

    for (uint32_t wu = 0; wu < num_wu; ++wu) {
        // Phase 1: pre-scale Q (folds attention scale) -> cb_q_scaled (resident).
        ckl::mul<
            cb_q_in,
            cb_scale,
            cb_q_scaled,
            BroadcastDim::Scalar,
            InputLifecycle::Streaming,
            InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::PackTileReconfig::Output,
            OperandKind::Scalar,
            OperandKind::Scalar>(EltwiseShape::tiles(sq_dt));

        for (uint32_t j = 0; j < n_kv_chunks; ++j) {
            const bool first = (j == 0);

            // Phase 2: scores = Q_scaled . K^T  (transpose B; cb_q_scaled retained).
            ckl::matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                q_scaled_buf,
                k_buf,
                scores_buf,
                scores_buf,
                MatmulBlockShape::of(qk_in0_sb, qk_in1_sb, qk_out_sb_h, qk_out_sb_w, Dt, 1));

            // Phase 3: additive mask (custom mode), in place, before the row-max.
            if constexpr (has_mask) {
                ckl::add<cb_scores, cb_mask_in, cb_scores>(EltwiseShape::tiles(sq_skv));
            }

            // Phase 3b: KV-padding mask (h_non_aligned), last chunk only. Additive
            // -inf on the last KV tile's padding columns so they drop out of the
            // softmax (max/exp/sum) — same additive path as the custom mask.
            if constexpr (has_kv_pad) {
                if (j == n_kv_chunks - 1) {
                    ckl::add<cb_scores, cb_kv_mask, cb_scores>(EltwiseShape::tiles(sq_skv));
                }
            }

            // Phase 4: chunk row-max -> cb_corr (cb_scores held for the exp below).
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_scores,
                cb_scaler,
                cb_corr,
                ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, Skv_chunk_t, Sq_chunk_t));

            // Phase 5: update running max m, form correction alpha.
            if (first) {
                ckl::copy<cb_corr, cb_row_max>(EltwiseShape::tiles(Sq_chunk_t));  // m = chunk_max
            } else {
                // m_new = max(chunk_max, m_old) -> cb_m_new (m_old held for alpha).
                // The held operand is walked with Block indexing (absolute base+i):
                // Scalar+HeldStream never advances the front (no pop), so it would
                // re-read tile 0 for every Sq_chunk_t>1 iteration.
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    cb_corr,
                    cb_row_max,
                    cb_m_new,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::Streaming,
                    ckl::PackTileReconfig::Output,
                    OperandKind::Scalar,
                    OperandKind::Block>(EltwiseShape::tiles(Sq_chunk_t));
                // alpha = exp(m_old - m_new) -> cb_corr (m_old popped, m_new held).
                ckl::eltwise_chain(
                    EltwiseShape::tiles(Sq_chunk_t),
                    ckl::BinaryFpu<
                        cb_row_max,
                        cb_m_new,
                        ckl::BinaryFpuOp::Sub,
                        BroadcastDim::None,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        Dst::D0,
                        OperandKind::Scalar,
                        OperandKind::Block>{},
                    ckl::Exp<>{},
                    ckl::PackTile<cb_corr, OutputLifecycle::Streaming>{});
                // m = m_new.
                ckl::copy<cb_m_new, cb_row_max>(EltwiseShape::tiles(Sq_chunk_t));
            }

            // Phase 6: P = exp(scores - m) -> cb_exp (Col bcast; scores popped, m held).
            ckl::eltwise_chain(
                EltwiseShape::grid(Sq_chunk_t, Skv_chunk_t),
                ckl::BinaryFpu<
                    cb_scores,
                    cb_row_max,
                    ckl::BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    InputLifecycle::Bulk,
                    InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Block,
                    OperandKind::Col>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_exp, OutputLifecycle::Bulk>{});

            // Phase 7: chunk row-sum + running sum l update.
            if (first) {
                // l = chunk_sum (cb_exp held for PV).
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_exp,
                    cb_scaler,
                    cb_row_sum,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, Skv_chunk_t, Sq_chunk_t));
            } else {
                // l = alpha * l (alpha held for phase 8; Block indexing walks the
                // held alpha tiles without popping — see phase-5 note).
                ckl::mul<
                    cb_row_sum,
                    cb_corr,
                    cb_row_sum,
                    BroadcastDim::None,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    OperandKind::Scalar,
                    OperandKind::Block>(EltwiseShape::tiles(Sq_chunk_t));
                // chunk_sum -> cb_sum_chunk (cb_exp held for PV).
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_exp,
                    cb_scaler,
                    cb_sum_chunk,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, Skv_chunk_t, Sq_chunk_t));
                // l += chunk_sum.
                ckl::add<cb_row_sum, cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(Sq_chunk_t));
            }

            // Phase 8: rescale running output O by alpha (Col bcast; pops alpha).
            if (!first) {
                ckl::mul<
                    cb_out_accum,
                    cb_corr,
                    cb_out_accum,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::Bulk,
                    OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    OperandKind::Scalar,
                    OperandKind::Col>(EltwiseShape::grid(Sq_chunk_t, Dt));
            }

            // Phase 9: PV = P . V -> cb_pv (cb_exp + cb_v popped).
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                exp_buf,
                v_buf,
                pv_buf,
                pv_buf,
                MatmulBlockShape::of(pv_in0_sb, pv_in1_sb, pv_out_sb_h, pv_out_sb_w, Skv_chunk_t, 1));

            // Phase 10: accumulate O += PV.
            if (first) {
                ckl::copy<cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));  // O = PV
            } else {
                ckl::add<cb_out_accum, cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));
            }
        }

        // Phase 11: normalize O = O * (1/l), pack to bf16 -> cb_out.
        ckl::unary<ckl::Recip<>, cb_row_sum, cb_row_sum>(EltwiseShape::tiles(Sq_chunk_t));
        ckl::mul<
            cb_out_accum,
            cb_row_sum,
            cb_out,
            BroadcastDim::Col,
            InputLifecycle::Streaming,
            InputLifecycle::Bulk,
            OutputLifecycle::Streaming,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::PackTileReconfig::Output,
            OperandKind::Scalar,
            OperandKind::Col>(EltwiseShape::grid(Sq_chunk_t, Dt));

        // Release the retained Q-scaled block and the running max for the next q-chunk.
        cb_pop_front(cb_q_scaled, sq_dt);
        cb_pop_front(cb_row_max, Sq_chunk_t);
    }
}
