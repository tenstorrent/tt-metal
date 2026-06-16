// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention compute kernel.
//
// Implements the FlashAttention online-softmax recurrence per work-unit
// (b, h, q-chunk). The S_q x S_kv score matrix is NEVER materialized: per
// Q-chunk (Bq_t = 1 tile-row of queries) we stream every KV-chunk
// (Bkv_t = 1 tile-row of keys/values) once and fold it into the running
// max (cb_m), running sum (cb_l) and running output (cb_o).
//
// All helper substitutions are documented inline. Raw-API usage:
//   - compute_kernel_hw_startup + mm_block_init: boot init (no helper covers
//     boot init; matmul_block's InitMode::Short handles per-call short-init).
//   - cb_pop_front(cb_q): frees the retained Q operand at unit end (the QKt
//     matmul uses WaitAndRetainOnLastBlock so it never pops cb_q).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Skv_t = get_compile_time_arg_val(0);       // KV-chunks streamed per Q-chunk
    constexpr uint32_t d_t = get_compile_time_arg_val(1);         // head-dim tiles
    constexpr uint32_t has_mask = get_compile_time_arg_val(2);    // custom-mask mode
    constexpr uint32_t scale_bits = get_compile_time_arg_val(3);  // fp32 scale, bit-cast

    const uint32_t num_units = get_arg_val<uint32_t>(0);

    // --- Circular buffers (semantic names) ---
    constexpr uint32_t cb_q_in = 0;
    constexpr uint32_t cb_k_in = 1;
    constexpr uint32_t cb_v_in = 2;
    constexpr uint32_t cb_mask_in = 3;
    constexpr uint32_t cb_scaler_max = 8;
    constexpr uint32_t cb_scaler_sum = 9;
    constexpr uint32_t cb_p = 10;
    constexpr uint32_t cb_o = 11;
    constexpr uint32_t cb_pv = 12;
    constexpr uint32_t cb_o_resc = 13;
    constexpr uint32_t cb_recip_l = 14;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_q = 24;
    constexpr uint32_t cb_scores = 25;
    constexpr uint32_t cb_m_cur = 26;
    constexpr uint32_t cb_m = 27;
    constexpr uint32_t cb_m_new = 28;
    constexpr uint32_t cb_l = 29;
    constexpr uint32_t cb_l_cur = 30;
    constexpr uint32_t cb_corr = 31;

    // Boot init (raw-API — no helper covers boot; see file header).
    compute_kernel_hw_startup(cb_q_in, cb_k_in, cb_scores);
    mm_block_init(cb_q, cb_k_in, cb_scores, /*transpose*/ 1, /*ct_dim*/ 1, /*rt_dim*/ 1, /*kt_dim*/ d_t);

    // matmul_block operand wrappers.
    CircularBuffer q_buf(cb_q), k_buf(cb_k_in), v_buf(cb_v_in);
    CircularBuffer scores_buf(cb_scores), p_buf(cb_p), pv_buf(cb_pv);

    for (uint32_t u = 0; u < num_units; ++u) {
        // 0c. scale Q: cb_q = cb_q_in * scale (folds the softmax scale into Q).
        eltwise_chain(
            EltwiseShape::tiles(d_t),
            CopyTile<cb_q_in, Dst::D0, InputLifecycle::Streaming>{},
            MulUnary<Dst::D0>(scale_bits),
            PackTile<cb_q, OutputLifecycle::Streaming>{});

        // 0b. init running stats: m = -inf, l = 0, O = 0.
        eltwise_chain(EltwiseShape::tiles(1), FillScalar<Dst::D0>(-1e30f), PackTile<cb_m>{});
        eltwise_chain(EltwiseShape::tiles(1), FillScalar<Dst::D0>(0.0f), PackTile<cb_l>{});
        eltwise_chain(EltwiseShape::tiles(d_t), FillScalar<Dst::D0>(0.0f), PackTile<cb_o>{});

        for (uint32_t j = 0; j < Skv_t; ++j) {
            // A. QKt: scores = Q . K^T (reduction over D). Q retained across kv loop.
            matmul_block<
                /*transpose*/ true,
                /*packer_l1_acc*/ false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndRetainOnLastBlock,  // never pops cb_q
                InputPolicy::WaitAndPopPerKBlock>(
                q_buf,
                k_buf,
                scores_buf,
                scores_buf,
                MatmulBlockShape::of(
                    /*in0_sb*/ 1, /*in1_sb*/ 1, /*sb_h*/ 1, /*sb_w*/ 1, /*in0_block_k*/ d_t, /*num_k_blocks*/ 1));

            // B. mask add (custom mode only): scores += mask.
            if constexpr (has_mask) {
                add<cb_scores, cb_mask_in, cb_scores, BroadcastDim::None>(EltwiseShape::tiles(1));
            }

            // C. rowmax over the KV (width) axis -> cb_m_cur [Bq_t,1]. No pop (exp reuses scores).
            reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_scores, cb_scaler_max, cb_m_cur, ReduceInputBlockShape::of(1, 1));

            // D. m_new = max(m, m_cur). m retained (corr needs old m); m_new held for E/G/K.
            binary_sfpu<
                BinaryMax<>,
                cb_m,
                cb_m_cur,
                cb_m_new,
                InputLifecycle::HeldStream,  // cb_m: wait, no pop
                InputLifecycle::Streaming>(  // cb_m_cur: wait + pop
                EltwiseShape::tiles(1));

            // E. P = exp(scores - m_new), per-row max broadcast across width (Col).
            eltwise_chain(
                EltwiseShape::tiles(1),
                BinaryFpu<
                    cb_scores,
                    cb_m_new,
                    BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,      // cb_scores: pop
                    InputLifecycle::HeldStream>{},  // cb_m_new: no pop (G/K need it)
                Exp<>{},
                PackTile<cb_p>{});

            // F. rowsum over P -> cb_l_cur [Bq_t,1]. No pop (PV reuses P).
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_p, cb_scaler_sum, cb_l_cur, ReduceInputBlockShape::of(1, 1));

            // G. corr = exp(m_old - m_new). Pops old m; m_new held (K still needs it).
            eltwise_chain(
                EltwiseShape::tiles(1),
                BinaryFpu<
                    cb_m,
                    cb_m_new,
                    BinaryFpuOp::Sub,
                    BroadcastDim::None,
                    InputLifecycle::Streaming,      // cb_m: pop (old max consumed)
                    InputLifecycle::HeldStream>{},  // cb_m_new: no pop
                Exp<>{},
                PackTile<cb_corr>{});

            // H. l = corr * l + l_cur. corr held (J1 needs it); l updated in place.
            mul<cb_corr, cb_l, cb_l, BroadcastDim::None, InputLifecycle::HeldStream, InputLifecycle::Streaming>(
                EltwiseShape::tiles(1));
            add<cb_l, cb_l_cur, cb_l, BroadcastDim::None>(EltwiseShape::tiles(1));

            // I. PV: pv = P . V (reduction over the kv-chunk).
            matmul_block<
                /*transpose*/ false,
                /*packer_l1_acc*/ false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock>(
                p_buf,
                v_buf,
                pv_buf,
                pv_buf,
                MatmulBlockShape::of(
                    /*in0_sb*/ 1, /*in1_sb*/ d_t, /*sb_h*/ 1, /*sb_w*/ 1, /*in0_block_k*/ 1, /*num_k_blocks*/ 1));

            // J1. o_resc = corr * O (per-row corr broadcast across d_t columns).
            mul<cb_o,
                cb_corr,
                cb_o_resc,
                BroadcastDim::Col,
                InputLifecycle::Bulk,  // cb_o: block, wait+pop d_t
                InputLifecycle::Bulk,  // cb_corr: col vector, wait+pop 1
                OutputLifecycle::Bulk,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Block,
                OperandKind::Col>(EltwiseShape::grid(1, d_t));

            // J2. O = o_resc + pv. Running output re-pushed.
            add<cb_o_resc,
                cb_pv,
                cb_o,
                BroadcastDim::None,
                InputLifecycle::Bulk,
                InputLifecycle::Bulk,
                OutputLifecycle::Bulk,
                BinaryDataFormatReconfig::Input,
                PackTileReconfig::Output,
                OperandKind::Block,
                OperandKind::Block>(EltwiseShape::grid(1, d_t));

            // K. advance running max: m = m_new.
            copy<cb_m_new, cb_m>(EltwiseShape::tiles(1));
        }

        // L. recip(l).
        unary<Recip<>, cb_l, cb_recip_l>(EltwiseShape::tiles(1));

        // M. O_final = O * (1/l) (per-row 1/l broadcast across d_t columns) -> cb_out.
        mul<cb_o,
            cb_recip_l,
            cb_out,
            BroadcastDim::Col,
            InputLifecycle::Bulk,
            InputLifecycle::Bulk,
            OutputLifecycle::Bulk,
            BinaryDataFormatReconfig::Input,
            PackTileReconfig::Output,
            OperandKind::Block,
            OperandKind::Col>(EltwiseShape::grid(1, d_t));

        // Z. free the retained Q (raw-API — matmul never pops it; see header).
        cb_pop_front(cb_q, d_t);
    }
}
