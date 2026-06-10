// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention compute kernel — online softmax recurrence, fully helper-composed.
//
// Per work unit (one Q chunk of cur_cq tile-rows for one (b, h)), per KV block kb:
//   0  (kb=0) init m_prev = -1e9 (full tile), l = 0, O = 0      [eltwise_chain Fill]
//   1  S      = Q @ K^T (transpose-B)                           [matmul_block, Q retained]
//   2  S'     = scale * S (+ mask)                              [eltwise_chain, SFPU scale]
//   4  cur    = rowmax(S') (col tile)                           [reduce MAX]
//   4b cur_f  = bcast-Col(cur)  (unpack-to-dest, exact)         [eltwise_chain UnaryBcast]
//   4c m      = max(m_prev, cur_f)  (full tiles, SFPU)          [eltwise_chain BinaryMax]
//   5  alpha  = exp(m_prev - m)     (full tiles, SFPU)          [eltwise_chain]
//   5b m_prev = m              (kb < last)                      [eltwise_chain copy]
//   6  P      = exp(S' - m)         (full tiles, SFPU)          [eltwise_chain]
//   7  r      = rowsum(P)                                       [reduce SUM, P retained]
//   8  l      = alpha * l + r       (SFPU)                      [eltwise_chain]
//   9  PV     = P @ V                                           [matmul_block]
//   10 O      = alpha * O + PV      (SFPU, 2-block cb_o_acc)    [eltwise_chain]
// After the last KV block:
//   11 inv    = bcast-Col(l) -> recip (full fp32 tile)          [eltwise_chain]
//   12 out    = O * inv  -> bf16   (SFPU mul, fp32-exact)       [eltwise_chain]
// The whole stat path (max/alpha/P/l/O-rescale/normalize) deliberately avoids the
// FPU (Refinement 5): FPU operands truncate fp32 to ~tf32 (~0.25 bf16-ulp per
// operand), which flipped ~1-2% of outputs by one bf16 ulp on near-uniform-
// attention inputs. m_prev/m/alpha are FULL fp32 tiles (cb_prev_max/cb_m_full/
// cb_alpha); UnaryBcast/CopyTile of Float32 CBs use unpack-to-dest (fp32-exact,
// UnpackToDestEn); Sub/Max/Mul/Add run on SFPU (fp32 RNE). FPU remains only for
// the matmuls (HiFi3) and the two reduces. Residual: exp/recip SFPU ulp + pack RNE.
// Statistics (m, l, alpha) are per-Q-row column tiles (Col0 valid); fp32 DEST
// accumulation throughout (FP32_DEST_ACC_EN, HiFi2 — DEST limit 4 tiles; all
// matmul subblocks are 1 x <=4 tiles).
//
// Only raw compute API: boot init + cb_pop_front releasing the retained Q chunk
// (the documented WaitAndRetainOnLastBlock counterpart) + the m_prev pops
// (old front after the 5b rotation push; final block's at unit end).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"

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
constexpr uint32_t cb_cur_max_full = 29;  // block rowmax bcast to full tile (fp32)
constexpr uint32_t cb_m_full = 30;        // running max as full tile (fp32)
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
    // Output dtype is bfloat16: RNE-round in DEST (SFPU typecast) before pack —
    // the packer truncates fp32->bf16, which drops a full bf16 ulp on values
    // sitting just below the grid (probe_015/016, Refinement 5 flip-rate fix).
    constexpr bool OUT_IS_BF16 = get_compile_time_arg_val(10) != 0;

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

            // ---- Phase 4: cur_max = rowmax(S') (col tile); scores stay fronted ----
            // Plain (non-accumulating) MAX reduce; the running max lives as a FULL
            // fp32 tile in cb_m_full so all stat math runs SFPU fp32-exact.
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_scores_scaled, cb_scaler_max, cb_running_max, ckl::ReduceInputBlockShape::of(cur_cq, cur_ckv));

            // ---- Phase 4b: cur_max_full = bcast-Col(cur_max) (unpack-to-dest, exact) ----
            ckl::eltwise_chain(
                cur_cq,
                ckl::UnaryBcast<BroadcastDim::Col, cb_running_max, InputLifecycle::Streaming>{},
                ckl::PackTile<cb_cur_max_full>{});

            // ---- Phase 4c: m = max(m_prev, cur_max) on SFPU (full tiles, fp32-exact) ----
            ckl::eltwise_chain(
                cur_cq,
                ckl::CopyTile<
                    cb_prev_max,
                    Dst::D0,
                    InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::CopyTile<cb_cur_max_full, Dst::D1, InputLifecycle::Streaming>{},
                ckl::BinaryMax<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::PackTile<cb_m_full>{});

            // ---- Phase 5: alpha = exp(m_prev - m) on SFPU (full tiles) ----
            ckl::eltwise_chain(
                cur_cq,
                ckl::CopyTile<
                    cb_prev_max,
                    Dst::D0,
                    InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::CopyTile<
                    cb_m_full,
                    Dst::D1,
                    InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_alpha>{});

            // ---- Phase 5b (kb < last): m_prev <- m; old m_prev popped after push ----
            if (kb + 1 < Nkv) {
                ckl::eltwise_chain(
                    cur_cq,
                    ckl::CopyTile<
                        cb_m_full,
                        Dst::D0,
                        InputLifecycle::HeldBulk,
                        ckl::CopyTileReconfig::Input,
                        OperandKind::Block>{},
                    ckl::PackTile<cb_prev_max>{});
                cb_pop_front(cb_prev_max, cur_cq);
            }

            // ---- Phase 6: P = exp(S' - m) on SFPU; pops S' and m_full ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(cur_cq, cur_ckv),
                ckl::CopyTile<
                    cb_scores_scaled,
                    Dst::D0,
                    InputLifecycle::DeferredPop,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::CopyTile<
                    cb_m_full,
                    Dst::D1,
                    InputLifecycle::DeferredPop,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Col>{},
                ckl::SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_probs, OutputLifecycle::Bulk>{});

            // ---- Phase 7: r = rowsum(P); P retained for P@V (col0 scaler, matmul path) ----
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_probs, cb_scaler_sum, cb_cur_sum, ckl::ReduceInputBlockShape::of(cur_cq, cur_ckv));

            // ---- Phase 8: l = alpha*l + r on SFPU (fp32-exact); alpha held for phase 10 ----
            ckl::eltwise_chain(
                cur_cq,
                ckl::CopyTile<cb_running_sum, Dst::D0, InputLifecycle::Streaming>{},
                ckl::CopyTile<
                    cb_alpha,
                    Dst::D1,
                    InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::CopyTile<cb_cur_sum, Dst::D2, InputLifecycle::Streaming>{},
                ckl::AddBinary<Dst::D0, Dst::D2, Dst::D0>{},
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

            // ---- Phase 10: O = alpha*O + PV on SFPU (fp32-exact); pops alpha.
            //      Same-CB read/write needs 2-block cb_o_acc. ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(cur_cq, Dt),
                ckl::CopyTile<
                    cb_o_acc,
                    Dst::D0,
                    InputLifecycle::Bulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::
                    CopyTile<cb_alpha, Dst::D1, InputLifecycle::Bulk, ckl::CopyTileReconfig::Input, OperandKind::Col>{},
                ckl::MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::CopyTile<cb_pv, Dst::D2, InputLifecycle::Streaming>{},
                ckl::AddBinary<Dst::D0, Dst::D2, Dst::D0>{},
                ckl::PackTile<cb_o_acc, OutputLifecycle::Bulk>{});
        }

        // ---- Phase 11: inv = recip(bcast-Col(l)) as a FULL fp32 tile ----
        // UnaryBcast of a Float32 CB unpacks to DEST (fp32-exact, no FPU
        // truncation); recip runs on SFPU over the full tile. Result: per-Q-row
        // 1/l replicated across all 32 columns, packed fp32 to cb_inv_sum.
        ckl::eltwise_chain(
            cur_cq,
            ckl::UnaryBcast<BroadcastDim::Col, cb_running_sum, InputLifecycle::Streaming>{},
            ckl::Recip<>{},
            ckl::PackTile<cb_inv_sum>{});

        // ---- Phase 12: out = O * inv on SFPU (fp32-exact), pack -> output dtype ----
        // copy_tile of Float32 CBs is unpack-to-dest (exact); MulBinary is SFPU
        // fp32 RNE — no FPU tf32 truncation on either operand. For bf16 output,
        // RNE-round in DEST (SFPU typecast) first: the packer truncates.
        if constexpr (OUT_IS_BF16) {
            // Software RNE on the fp32 bits (t += 0x7FFF + ((t>>16)&1)); the
            // truncating packer then drops exactly the rounded-off bits.
            // Sign-magnitude fp32 makes the same bit trick correct for negatives.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(cur_cq, Dt),
                ckl::CopyTile<
                    cb_o_acc,
                    Dst::D0,
                    InputLifecycle::Bulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::CopyTile<
                    cb_inv_sum,
                    Dst::D1,
                    InputLifecycle::Bulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Col>{},
                ckl::MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::PackTile<cb_out_tiles>{});
        } else {
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(cur_cq, Dt),
                ckl::CopyTile<
                    cb_o_acc,
                    Dst::D0,
                    InputLifecycle::Bulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Block>{},
                ckl::CopyTile<
                    cb_inv_sum,
                    Dst::D1,
                    InputLifecycle::Bulk,
                    ckl::CopyTileReconfig::Input,
                    OperandKind::Col>{},
                ckl::MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                ckl::PackTile<cb_out_tiles>{});
        }

        // Release the retained Q chunk (WaitAndRetainOnLastBlock counterpart) and the
        // last m_prev block (the pop Phase 5b would have issued on a next kb).
        cb_pop_front(cb_q_tiles, cur_cq * Dt);
        cb_pop_front(cb_prev_max, cur_cq);
    }
}
