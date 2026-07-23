// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-attention compute kernel (online-softmax recurrence).
//
// Per Q-block, stream every KV-block once maintaining running (m, l, O):
//   m_cur = max(m_prev, rowmax(S))                          [S = Q·Kᵀ (+mask)]
//   corr  = exp((m_prev - m_cur)·scale)
//   P     = exp((S - m_cur)·scale)
//   l_cur = corr·l_prev + rowsum(P)
//   O_cur = corr·O_prev + P·V
// then O /= l, per row. The S_q×S_kv score matrix is never materialized —
// only one (Sq_chunk_t × Sk_chunk_t) block (cb_qk_scores) lives in L1.
//
// scale (1/sqrt(D) or explicit) is folded into every exp (MulUnary before Exp),
// never into QKᵀ — matches the numerically-exact online form.
//
// Helper usage (all kernel_lib): matmul_block (QKᵀ transpose + PV),
// reduce<MAX/SUM,REDUCE_ROW> (row-max / row-sum), binary_sfpu<BinaryMax> (running
// max), eltwise_chain (fused sub·scale·exp) + mul/add/unary<Recip> (rescale +
// normalize). Ping-pong (m,l,O) is realized by a templated per-KV-step function
// instantiated for both parities (CB ids are compile-time), selected per
// iteration at runtime.
//
// matmul_block: TileRowMajor output so the downstream reduce/eltwise see a
// row-major (rows × cols) tile grid; num_k_blocks=1 (whole contraction in DEST),
// interm placeholder = out_buf; QKᵀ retains in0 (Q) via WaitAndRetainOnLastBlock.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_mask_in = 3;
constexpr uint32_t cb_scaler_max = 4;
constexpr uint32_t cb_scaler_sum = 5;
constexpr uint32_t cb_out = 16;

constexpr uint32_t cb_qk_scores = 24;
constexpr uint32_t cb_max_A = 25;
constexpr uint32_t cb_max_B = 26;
constexpr uint32_t cb_max_new = 27;
constexpr uint32_t cb_sum_A = 28;
constexpr uint32_t cb_sum_B = 29;
constexpr uint32_t cb_sum_new = 30;
constexpr uint32_t cb_exp_max_diff = 31;
constexpr uint32_t cb_out_A = 6;
constexpr uint32_t cb_out_B = 7;
constexpr uint32_t cb_out_new = 8;
constexpr uint32_t cb_sum_scaled = 9;
constexpr uint32_t cb_out_scaled = 10;

struct MMParams {
    uint32_t sq, sk, dht, scale_bits;
    uint32_t qk_in0_sb, qk_in1_sb, qk_sb_h, qk_sb_w;
    uint32_t pv_in0_sb, pv_in1_sb, pv_sb_h, pv_sb_w;
};
}  // namespace

// One online-softmax KV-step. CB ids for the running (m,l,O) prev/cur buffers
// are template params; instantiated for both parities.
template <
    uint32_t CB_PREV_MAX,
    uint32_t CB_CUR_MAX,
    uint32_t CB_PREV_SUM,
    uint32_t CB_CUR_SUM,
    uint32_t CB_PREV_OUT,
    uint32_t CB_CUR_OUT,
    bool HAS_MASK>
void kv_step(bool first, const MMParams& p) {
    const uint32_t sq = p.sq, sk = p.sk, dht = p.dht;

    // ---- 1. S = Q·Kᵀ  (Q retained across the KV loop; K popped) ----
    {
        CircularBuffer q_buf(cb_q_in), k_buf(cb_k_in), qk_buf(cb_qk_scores);
        ckl::matmul_block<
            /*transpose=*/true,
            /*packer_l1_acc=*/false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock>(
            q_buf,
            k_buf,
            qk_buf,
            qk_buf,
            ckl::MatmulBlockShape::of(p.qk_in0_sb, p.qk_in1_sb, p.qk_sb_h, p.qk_sb_w, dht, 1));
    }

    // ---- 2. + mask (custom) ----
    if constexpr (HAS_MASK) {
        ckl::add<ckl::input(cb_qk_scores), ckl::input(cb_mask_in), ckl::output(cb_qk_scores)>(
            ckl::EltwiseShape::grid(sq, sk));
    }

    // ---- 3. row-max + running-max update ----
    if (first) {
        ckl::reduce<
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_qk_scores,
            cb_scaler_max,
            CB_CUR_MAX,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(sq, sk));
    } else {
        ckl::reduce<
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_qk_scores,
            cb_scaler_max,
            cb_max_new,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(sq, sk));
        // m_cur = max(m_prev, m_new); keep m_prev (needed by correction, popped there).
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            ckl::input(CB_PREV_MAX, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
            ckl::input(cb_max_new),
            ckl::output(CB_CUR_MAX)>(ckl::EltwiseShape::tiles(sq));
        // corr = exp((m_prev - m_cur)·scale); pops m_prev, keeps m_cur.
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(sq),
            ckl::BinaryFpu<
                ckl::input(CB_PREV_MAX, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(CB_CUR_MAX, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None>{},
            ckl::MulUnary<>{p.scale_bits},
            ckl::Exp<>{},
            ckl::PackTile<ckl::output(cb_exp_max_diff)>{});
    }

    // ---- 4. P = exp((S - m_cur)·scale)  (in place, m_cur col-broadcast) ----
    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(sq, sk),
        ckl::BinaryFpu<
            ckl::input(cb_qk_scores),
            ckl::input(CB_CUR_MAX, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Col),
            ckl::BinaryFpuOp::Sub,
            ckl::BroadcastDim::Col>{},
        ckl::MulUnary<>{p.scale_bits},
        ckl::Exp<>{},
        ckl::PackTile<ckl::output(cb_qk_scores)>{});

    // ---- row-sum of P (keep P resident for PV matmul) ----
    const uint32_t sum_target = first ? CB_CUR_SUM : cb_sum_new;
    if (first) {
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_qk_scores,
            cb_scaler_sum,
            CB_CUR_SUM,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(sq, sk));
    } else {
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_qk_scores,
            cb_scaler_sum,
            cb_sum_new,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(sq, sk));
    }
    (void)sum_target;

    // ---- 5. O_new = P·V  (P popped, V popped) ----
    const uint32_t out_target = first ? CB_CUR_OUT : cb_out_new;
    if (first) {
        CircularBuffer p_buf(cb_qk_scores), v_buf(cb_v_in), o_buf(CB_CUR_OUT);
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor>(
            p_buf,
            v_buf,
            o_buf,
            o_buf,
            ckl::MatmulBlockShape::of(p.pv_in0_sb, p.pv_in1_sb, p.pv_sb_h, p.pv_sb_w, sk, 1));
    } else {
        CircularBuffer p_buf(cb_qk_scores), v_buf(cb_v_in), o_buf(cb_out_new);
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor>(
            p_buf,
            v_buf,
            o_buf,
            o_buf,
            ckl::MatmulBlockShape::of(p.pv_in0_sb, p.pv_in1_sb, p.pv_sb_h, p.pv_sb_w, sk, 1));
    }
    (void)out_target;

    // ---- 6. rescale prior l, O by corr (only when not the first KV-block) ----
    if (!first) {
        // l_cur = corr·l_prev + rowsum(P)
        ckl::mul<
            ckl::input(CB_PREV_SUM),
            ckl::input(cb_exp_max_diff, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
            ckl::output(cb_sum_scaled)>(ckl::EltwiseShape::tiles(sq));
        ckl::add<ckl::input(cb_sum_new), ckl::input(cb_sum_scaled), ckl::output(CB_CUR_SUM)>(
            ckl::EltwiseShape::tiles(sq));
        // O_cur = corr·O_prev + O_new   (corr col-broadcast across DHT)
        ckl::mul<
            ckl::input(CB_PREV_OUT),
            ckl::input(cb_exp_max_diff, ckl::InputLifecycle::Bulk, ckl::OperandKind::Col),
            ckl::output(cb_out_scaled),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(sq, dht));
        ckl::add<ckl::input(cb_out_new), ckl::input(cb_out_scaled), ckl::output(CB_CUR_OUT)>(
            ckl::EltwiseShape::grid(sq, dht));
    }
}

// Normalize O /= l and drain to cb_out; release the leftover running max.
template <uint32_t CB_SUM_F, uint32_t CB_OUT_F, uint32_t CB_MAX_F>
void finalize(uint32_t sq, uint32_t dht) {
    // 1/l (in place on the final running sum)
    ckl::unary<ckl::Recip<>, ckl::input(CB_SUM_F), ckl::output(CB_SUM_F)>(ckl::EltwiseShape::tiles(sq));
    // O · (1/l)  → cb_out  (per-row reciprocal col-broadcast across DHT)
    ckl::mul<
        ckl::input(CB_OUT_F),
        ckl::input(CB_SUM_F, ckl::InputLifecycle::Bulk, ckl::OperandKind::Col),
        ckl::output(cb_out),
        ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(sq, dht));
    // the last KV-block's running max was never consumed — release it.
    cb_pop_front(CB_MAX_F, sq);
}

void kernel_main() {
    const uint32_t sq = get_compile_time_arg_val(0);
    const uint32_t sk = get_compile_time_arg_val(1);
    const uint32_t dht = get_compile_time_arg_val(2);
    constexpr uint32_t K_NUM_CHUNKS = get_compile_time_arg_val(3);
    constexpr uint32_t HAS_MASK = get_compile_time_arg_val(4);
    const uint32_t scale_bits = get_compile_time_arg_val(5);
    const uint32_t qk_in0_sb = get_compile_time_arg_val(6);
    const uint32_t qk_in1_sb = get_compile_time_arg_val(7);
    const uint32_t qk_sb_h = get_compile_time_arg_val(8);
    const uint32_t qk_sb_w = get_compile_time_arg_val(9);
    const uint32_t pv_in0_sb = get_compile_time_arg_val(10);
    const uint32_t pv_in1_sb = get_compile_time_arg_val(11);
    const uint32_t pv_sb_h = get_compile_time_arg_val(12);
    const uint32_t pv_sb_w = get_compile_time_arg_val(13);

    const uint32_t q_count = get_arg_val<uint32_t>(0);

    const MMParams p{
        sq, sk, dht, scale_bits, qk_in0_sb, qk_in1_sb, qk_sb_h, qk_sb_w, pv_in0_sb, pv_in1_sb, pv_sb_h, pv_sb_w};
    const uint32_t q_block_tiles = sq * dht;

    compute_kernel_hw_startup(cb_q_in, cb_k_in, cb_out);
    mm_block_init(cb_q_in, cb_k_in, cb_qk_scores, 0, 1, 1, dht);

    for (uint32_t qb = 0; qb < q_count; ++qb) {
        for (uint32_t k = 0; k < K_NUM_CHUNKS; ++k) {
            const bool first = (k == 0);
            if ((k & 1u) == 0u) {
                kv_step<cb_max_A, cb_max_B, cb_sum_A, cb_sum_B, cb_out_A, cb_out_B, HAS_MASK != 0>(first, p);
            } else {
                kv_step<cb_max_B, cb_max_A, cb_sum_B, cb_sum_A, cb_out_B, cb_out_A, HAS_MASK != 0>(first, p);
            }
        }

        // Final state lives in the "cur" buffers of the last KV-block.
        if constexpr (((K_NUM_CHUNKS - 1) & 1u) == 0u) {
            finalize<cb_sum_B, cb_out_B, cb_max_B>(sq, dht);
        } else {
            finalize<cb_sum_A, cb_out_A, cb_max_A>(sq, dht);
        }

        // Q was retained across the KV loop — release it for the next Q-block.
        cb_pop_front(cb_q_in, q_block_tiles);
    }
}
