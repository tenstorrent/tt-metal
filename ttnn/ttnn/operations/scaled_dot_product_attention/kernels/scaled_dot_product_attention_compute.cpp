// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention compute kernel — online softmax recurrence, fully helper-composed.
//
// Per work unit (one Q chunk of cur_cq tile-rows for one (b, h)), per KV block kb:
//   0  (kb=0) init m = -1e9, l = 0, O = 0                       [eltwise_chain Fill]
//   1  S      = Q @ K^T (transpose-B)                           [matmul_block, Q retained]
//   2  S'     = scale * S (+ mask)                              [eltwise_chain]
//   3  m_prev = m                       (kb > 0)                [eltwise_chain copy]
//   4  m      = max(m, rowmax(S'))                              [accumulate_reduce_block MAX]
//   5  alpha  = exp(m_prev - m)                                 [eltwise_chain]
//   6  P      = exp(S' - m)  bcast-Col                          [eltwise_chain]
//   7  r      = rowsum(P)                                       [reduce SUM, P retained]
//   8  l      = alpha * l + r                                   [eltwise_chain]
//   9  PV     = P @ V                                           [matmul_block]
//   10 O      = alpha (.) O + PV                                [eltwise_chain, 2-block cb_o_acc]
// After the last KV block:
//   11 inv    = 1 / l                                           [eltwise_chain]
//   12 out    = O (.) inv  -> bf16                              [eltwise_chain]
// Statistics (m, l, alpha) are per-Q-row column tiles (Col0 valid); fp32 DEST
// accumulation throughout (FP32_DEST_ACC_EN, HiFi2 — DEST limit 4 tiles; all
// matmul subblocks are 1 x <=4 tiles).
//
// Only raw compute API: boot init + cb_pop_front releasing the retained Q chunk
// (the documented WaitAndRetainOnLastBlock counterpart) + the running-max
// chunk-end pop (counterpart of the next-kb reload pop the last block never gets).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_q_tiles = 0;
constexpr uint32_t cb_kt_tiles = 1;
constexpr uint32_t cb_v_tiles = 2;
constexpr uint32_t cb_mask_tiles = 3;
constexpr uint32_t cb_scaler_max = 8;
constexpr uint32_t cb_scaler_sum = 9;
constexpr uint32_t cb_cur_sum = 10;
constexpr uint32_t cb_prev_max = 11;
constexpr uint32_t cb_running_max = 12;
constexpr uint32_t cb_alpha = 13;
constexpr uint32_t cb_running_sum = 14;
constexpr uint32_t cb_inv_sum = 15;
constexpr uint32_t cb_out_tiles = 16;
constexpr uint32_t cb_scores = 24;
constexpr uint32_t cb_scores_scaled = 25;
constexpr uint32_t cb_probs = 26;
constexpr uint32_t cb_pv = 27;
constexpr uint32_t cb_o_acc = 28;
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t c_q = get_compile_time_arg_val(1);
    constexpr uint32_t c_kv = get_compile_time_arg_val(2);
    constexpr uint32_t Nq = get_compile_time_arg_val(3);
    constexpr uint32_t Nkv = get_compile_time_arg_val(4);
    constexpr uint32_t c_q_last = get_compile_time_arg_val(5);
    constexpr uint32_t c_kv_last = get_compile_time_arg_val(6);
    constexpr uint32_t sw = get_compile_time_arg_val(7);       // P@V subblock width (<= 4)
    constexpr uint32_t n_sub_w = get_compile_time_arg_val(8);  // Dt / sw
    constexpr bool HAS_MASK = get_compile_time_arg_val(9) != 0;

    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t scale_bits = get_arg_val<uint32_t>(2);

    if (num_units == 0) {
        return;
    }

    // Boot: one hw_configure-bearing init, then helpers' Short init handles all
    // subsequent matmul-state restores.
    compute_kernel_hw_startup(cb_q_tiles, cb_kt_tiles, cb_scores);
    mm_init(cb_q_tiles, cb_kt_tiles, cb_scores, /*transpose=*/1);

    CircularBuffer q_buf(cb_q_tiles);
    CircularBuffer kt_buf(cb_kt_tiles);
    CircularBuffer v_buf(cb_v_tiles);
    CircularBuffer scores_buf(cb_scores);
    CircularBuffer probs_buf(cb_probs);
    CircularBuffer pv_buf(cb_pv);

    using ckl::BinaryFpuOp;
    using ckl::BroadcastDim;
    using ckl::Dst;
    using ckl::InputLifecycle;
    using ckl::OperandKind;
    using ckl::OutputLifecycle;

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t qc = unit % Nq;
        const uint32_t cur_cq = (qc == Nq - 1) ? c_q_last : c_q;

        // ---- Phase 0: init m = -1e9 (not -inf: avoids inf-inf NaN), l = 0, O = 0 ----
        ckl::eltwise_chain(cur_cq, ckl::FillScalar<Dst::D0>{-1e9f}, ckl::PackTile<cb_prev_max>{});
        ckl::eltwise_chain(cur_cq, ckl::FillScalar<Dst::D0>{0.0f}, ckl::PackTile<cb_running_sum>{});
        ckl::eltwise_chain(cur_cq * Dt, ckl::FillScalar<Dst::D0>{0.0f}, ckl::PackTile<cb_o_acc>{});

        for (uint32_t kb = 0; kb < Nkv; ++kb) {
            const uint32_t cur_ckv = (kb == Nkv - 1) ? c_kv_last : c_kv;

            // ---- Phase 1: S = Q @ K^T (tile-order transpose by reader, intra-tile here) ----
            ckl::matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,  // Q reused across the KV loop
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                q_buf,
                kt_buf,
                scores_buf,
                q_buf,  // interm unused (num_k_blocks == 1)
                ckl::MatmulBlockShape::of(cur_cq, 1, 1, cur_ckv, Dt, 1));

            // ---- Phase 2: S' = scale*S (+ mask), block-wise mask before the max ----
            if constexpr (HAS_MASK) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(cur_cq, cur_ckv),
                    ckl::CopyTile<cb_scores, Dst::D0, InputLifecycle::Streaming>{},
                    ckl::MulUnary<Dst::D0>{scale_bits},
                    ckl::DestReuseBinary<
                        cb_mask_tiles,
                        BinaryFpuOp::Add,
                        ckl::DestReuseType::DEST_TO_SRCB,
                        InputLifecycle::Streaming>{},
                    ckl::PackTile<cb_scores_scaled>{});
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(cur_cq, cur_ckv),
                    ckl::CopyTile<cb_scores, Dst::D0, InputLifecycle::Streaming>{},
                    ckl::MulUnary<Dst::D0>{scale_bits},
                    ckl::PackTile<cb_scores_scaled>{});
            }

            // ---- Phase 3 (kb > 0): m_prev = m (running max held; reload pops it) ----
            if (kb > 0) {
                ckl::eltwise_chain(
                    cur_cq,
                    ckl::CopyTile<
                        cb_running_max,
                        Dst::D0,
                        InputLifecycle::HeldBulk,
                        ckl::CopyTileReconfig::Input,
                        OperandKind::Block>{},
                    ckl::PackTile<cb_prev_max>{});
            }

            // ---- Phase 4: m = max(m, rowmax(S')); scores stay fronted for phase 6 ----
            ckl::accumulate_reduce_block<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_scores_scaled,
                cb_scaler_max,
                cb_running_max,
                ckl::ReduceInputBlockShape::of(cur_cq, cur_ckv),
                kb,
                Nkv);

            // ---- Phase 5: alpha = exp(m_prev - m); m stays fronted ----
            // Both operands are CBs, so use BinaryFpu (A - B = m_prev - m) instead of
            // dest-reuse: DEST_TO_SRCA Sub zeroed rows > 0 (probe_004 DPRINT), and
            // DEST_TO_SRCB computes m - m_prev (probe_002).
            ckl::eltwise_chain(
                cur_cq,
                ckl::BinaryFpu<
                    cb_prev_max,
                    cb_running_max,
                    BinaryFpuOp::Sub,
                    BroadcastDim::None,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Scalar,
                    OperandKind::Block>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_alpha>{});

            // ---- Phase 6: P = exp(S' - m), Col broadcast; pops S', m survives ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(cur_cq, cur_ckv),
                ckl::BinaryFpu<
                    cb_scores_scaled,
                    cb_running_max,
                    BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    InputLifecycle::DeferredPop,
                    InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Block,
                    OperandKind::Col>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_probs, OutputLifecycle::Bulk>{});

            // ---- Phase 7: r = rowsum(P); P retained for P@V (col0 scaler, matmul path) ----
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_probs, cb_scaler_sum, cb_cur_sum, ckl::ReduceInputBlockShape::of(cur_cq, cur_ckv));

            // ---- Phase 8: l = alpha*l + r; alpha held for phase 10 ----
            ckl::eltwise_chain(
                cur_cq,
                ckl::CopyTile<cb_running_sum, Dst::D0, InputLifecycle::Streaming>{},
                ckl::DestReuseBinary<
                    cb_alpha,
                    BinaryFpuOp::Mul,
                    ckl::DestReuseType::DEST_TO_SRCB,
                    InputLifecycle::HeldBulk,
                    ckl::DestReuseReconfig::Input,
                    Dst::D0,
                    Dst::D0,
                    OperandKind::Block>{},
                ckl::DestReuseBinary<
                    cb_cur_sum,
                    BinaryFpuOp::Add,
                    ckl::DestReuseType::DEST_TO_SRCB,
                    InputLifecycle::Streaming>{},
                ckl::PackTile<cb_running_sum>{});

            // ---- Phase 9: PV = P @ V; pops P and V ----
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                probs_buf,
                v_buf,
                pv_buf,
                probs_buf,  // interm unused (num_k_blocks == 1)
                ckl::MatmulBlockShape::of(cur_cq, n_sub_w, 1, sw, cur_ckv, 1));

            // ---- Phase 10: O = alpha (.) O + PV; same-CB read/write needs 2-block cb_o_acc ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(cur_cq, Dt),
                ckl::BinaryFpu<
                    cb_o_acc,
                    cb_alpha,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Col,
                    InputLifecycle::Bulk,
                    InputLifecycle::Bulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    Dst::D0,
                    OperandKind::Block,
                    OperandKind::Col>{},
                ckl::DestReuseBinary<
                    cb_pv,
                    BinaryFpuOp::Add,
                    ckl::DestReuseType::DEST_TO_SRCB,
                    InputLifecycle::Streaming>{},
                ckl::PackTile<cb_o_acc, OutputLifecycle::Bulk>{});
        }

        // ---- Phase 11: inv = 1 / l ----
        ckl::eltwise_chain(
            cur_cq,
            ckl::CopyTile<cb_running_sum, Dst::D0, InputLifecycle::Streaming>{},
            ckl::Recip<>{},
            ckl::PackTile<cb_inv_sum>{});

        // ---- Phase 12: out = O (.) inv -> bf16 ----
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(cur_cq, Dt),
            ckl::BinaryFpu<
                cb_o_acc,
                cb_inv_sum,
                BinaryFpuOp::Mul,
                BroadcastDim::Col,
                InputLifecycle::Bulk,
                InputLifecycle::Bulk,
                ckl::BinaryDataFormatReconfig::Input,
                Dst::D0,
                OperandKind::Block,
                OperandKind::Col>{},
            ckl::PackTile<cb_out_tiles>{});

        // Release the retained Q chunk (WaitAndRetainOnLastBlock counterpart) and the
        // last running-max block (the pop the next kb's reload would have issued).
        cb_pop_front(cb_q_tiles, cur_cq * Dt);
        cb_pop_front(cb_running_max, cur_cq);
    }
}
