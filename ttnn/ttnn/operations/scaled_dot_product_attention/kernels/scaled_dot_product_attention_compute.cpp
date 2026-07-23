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
// The running (m, l, O) are held in FIXED CBs updated IN PLACE each KV-block
// (read the old value via a Held policy, pop it, push the new value in the same
// step). No A/B ping-pong and no compile-time parity split — one straight-line
// body, so the kernel binary stays within the kernel-config buffer even for
// long sequences. k_num_chunks is a runtime arg so the KV loop stays rolled.
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
constexpr uint32_t cb_m = 25;        // running max (in place)
constexpr uint32_t cb_max_cur = 26;  // this block's m_cur (scratch)
constexpr uint32_t cb_max_new = 27;  // this block's rowmax (scratch)
constexpr uint32_t cb_l = 28;        // running sum (in place)
constexpr uint32_t cb_sum_new = 30;  // this block's rowsum (scratch)
constexpr uint32_t cb_exp_max_diff = 31;
constexpr uint32_t cb_o = 6;            // running output accumulator (in place)
constexpr uint32_t cb_out_new = 8;      // this block's P·V (scratch)
constexpr uint32_t cb_sum_scaled = 9;   // scratch
constexpr uint32_t cb_out_scaled = 10;  // scratch

struct MMParams {
    uint32_t sq, sk, dht, scale_bits;
    uint32_t qk_in0_sb, qk_in1_sb, qk_sb_h, qk_sb_w;
    uint32_t pv_in0_sb, pv_in1_sb, pv_sb_h, pv_sb_w;
    uint32_t has_mask;
    uint32_t ablate;  // /perf-measure ablation gate: 0=normal, 1=matmul-stub, 2=+reduce-stub, 3=+exp/rescale-stub
};

// One online-softmax KV-step, updating the running (m,l,O) CBs in place.
//
// ExpMode (compile-time): the SFPU exp datapath. Approx::Exact is the numerically
// exact exp_tile (byte-identical default). Approx::Fast selects the hardware's fast
// approximate exp_tile — a large SFPU-floor reduction (Refinement 3d measured 1.44×
// on the flagged shape: 10.25 → 7.12 ms, since the phase-4 exp over the whole score
// block is the single dominant SFPU cost — 21%+ of the wall). Fast exp trades a small
// amount of accuracy (flagged shape PCC 0.9997→0.9967), so it is gated on the compute
// config's `math_approx_mode` (the user's explicit opt-in to approximate SFPU math);
// the exact default is unchanged.
template <ckl::Approx ExpMode>
void kv_step(bool first, const MMParams& p) {
    const uint32_t sq = p.sq, sk = p.sk, dht = p.dht;

    // ---- 1. S = Q·Kᵀ  (Q retained across the KV loop; K popped) ----
    if (p.ablate >= 1) {
        // matmul-stub: keep CB scaffolding, no FPU work (measures SFPU/softmax floor).
        cb_wait_front(cb_q_in, sq * dht);  // Q present (retained; final pop frees it)
        cb_wait_front(cb_k_in, sk * dht);
        cb_pop_front(cb_k_in, sk * dht);
        cb_reserve_back(cb_qk_scores, sq * sk);
        cb_push_back(cb_qk_scores, sq * sk);
    } else {
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
    if (p.has_mask) {
        ckl::add<ckl::input(cb_qk_scores), ckl::input(cb_mask_in), ckl::output(cb_qk_scores)>(
            ckl::EltwiseShape::grid(sq, sk));
    }

    // ---- 3. rowmax → m_cur ----
    if (p.ablate >= 2) {
        // reduce-stub: keep CB scaffolding, no reduce (measures the two-reduce cost).
        cb_reserve_back(cb_max_new, sq);
        cb_push_back(cb_max_new, sq);
    } else {
        ckl::reduce<
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_qk_scores,
            cb_scaler_max,
            cb_max_new,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(sq, sk));
    }

    if (first) {
        // m_cur = rowmax → keep it directly in cb_max_cur (then copied into cb_m).
        ckl::copy<ckl::input(cb_max_new), ckl::output(cb_max_cur)>(ckl::EltwiseShape::tiles(sq));
    } else {
        // m_cur = max(m_prev, m_new); keep m_prev for the correction.
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            ckl::input(cb_m, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
            ckl::input(cb_max_new),
            ckl::output(cb_max_cur)>(ckl::EltwiseShape::tiles(sq));
        // corr = exp((m_prev - m_cur)·scale); pops m_prev (old cb_m), keeps m_cur.
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(sq),
            ckl::BinaryFpu<
                ckl::input(cb_m, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(cb_max_cur, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None>{},
            ckl::MulUnary<>{p.scale_bits},
            ckl::Exp<ExpMode>{},
            ckl::PackTile<ckl::output(cb_exp_max_diff)>{});
    }

    // ---- 4. P = exp((S - m_cur)·scale)  (in place, m_cur col-broadcast) ----
    // ablate>=3: skip the exp chain (the dominant per-KV-block SFPU op over the whole
    // sq×sk score block) to isolate its cost. cb_qk_scores is left as the matmul-stub
    // pushed it (sq·sk tiles present) and popped by the PV matmul-stub → CB balance holds.
    if (p.ablate < 3) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(sq, sk),
            ckl::BinaryFpu<
                ckl::input(cb_qk_scores),
                ckl::input(cb_max_cur, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Col),
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::Col>{},
            ckl::MulUnary<>{p.scale_bits},
            ckl::Exp<ExpMode>{},
            ckl::PackTile<ckl::output(cb_qk_scores)>{});
    }

    // m_cur -> running max cb_m (cb_m was popped above for !first; freshly filled for first).
    ckl::copy<ckl::input(cb_max_cur), ckl::output(cb_m)>(ckl::EltwiseShape::tiles(sq));

    // ---- 5. rowsum(P) → cb_sum_new  (keep P resident for PV matmul) ----
    if (p.ablate >= 2) {
        // reduce-stub: keep CB scaffolding, no reduce.
        cb_reserve_back(cb_sum_new, sq);
        cb_push_back(cb_sum_new, sq);
    } else {
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_qk_scores,
            cb_scaler_sum,
            cb_sum_new,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(sq, sk));
    }

    // ---- 6. O_new = P·V  (P popped, V popped) ----
    if (p.ablate >= 1) {
        // matmul-stub: keep CB scaffolding, no FPU work.
        cb_wait_front(cb_qk_scores, sq * sk);
        cb_pop_front(cb_qk_scores, sq * sk);
        cb_wait_front(cb_v_in, sk * dht);
        cb_pop_front(cb_v_in, sk * dht);
        cb_reserve_back(cb_out_new, sq * dht);
        cb_push_back(cb_out_new, sq * dht);
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

    // ---- 7. update running l, O ----
    if (first) {
        // l = rowsum(P); O = O_new
        ckl::copy<ckl::input(cb_sum_new), ckl::output(cb_l)>(ckl::EltwiseShape::tiles(sq));
        ckl::copy<ckl::input(cb_out_new), ckl::output(cb_o)>(ckl::EltwiseShape::tiles(sq * dht));
    } else {
        // l = corr·l_prev + rowsum(P)   (cb_l updated in place)
        ckl::mul<
            ckl::input(cb_l),
            ckl::input(cb_exp_max_diff, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
            ckl::output(cb_sum_scaled)>(ckl::EltwiseShape::tiles(sq));
        ckl::add<ckl::input(cb_sum_new), ckl::input(cb_sum_scaled), ckl::output(cb_l)>(ckl::EltwiseShape::tiles(sq));
        // O = corr·O_prev + O_new   (corr col-broadcast across DHT; cb_o in place)
        ckl::mul<
            ckl::input(cb_o),
            ckl::input(cb_exp_max_diff, ckl::InputLifecycle::Bulk, ckl::OperandKind::Col),
            ckl::output(cb_out_scaled),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(sq, dht));
        ckl::add<ckl::input(cb_out_new), ckl::input(cb_out_scaled), ckl::output(cb_o)>(
            ckl::EltwiseShape::grid(sq, dht));
    }
}

}  // namespace

void kernel_main() {
    const uint32_t sq = get_compile_time_arg_val(0);
    const uint32_t sk = get_compile_time_arg_val(1);
    const uint32_t dht = get_compile_time_arg_val(2);
    const uint32_t has_mask = get_compile_time_arg_val(3);
    const uint32_t scale_bits = get_compile_time_arg_val(4);
    const uint32_t qk_in0_sb = get_compile_time_arg_val(5);
    const uint32_t qk_in1_sb = get_compile_time_arg_val(6);
    const uint32_t qk_sb_h = get_compile_time_arg_val(7);
    const uint32_t qk_sb_w = get_compile_time_arg_val(8);
    const uint32_t pv_in0_sb = get_compile_time_arg_val(9);
    const uint32_t pv_in1_sb = get_compile_time_arg_val(10);
    const uint32_t pv_sb_h = get_compile_time_arg_val(11);
    const uint32_t pv_sb_w = get_compile_time_arg_val(12);
    const uint32_t ablate = get_compile_time_arg_val(13);
    // Refinement 3d — SFPU-floor lever: fast approximate exp_tile. Gated on the
    // compute config's math_approx_mode (CT arg 14); default 0 = exact = byte-identical.
    constexpr uint32_t exp_approx = get_compile_time_arg_val(14);

    const uint32_t q_count = get_arg_val<uint32_t>(0);
    const uint32_t k_num_chunks = get_arg_val<uint32_t>(1);

    const MMParams p{
        sq,
        sk,
        dht,
        scale_bits,
        qk_in0_sb,
        qk_in1_sb,
        qk_sb_h,
        qk_sb_w,
        pv_in0_sb,
        pv_in1_sb,
        pv_sb_h,
        pv_sb_w,
        has_mask,
        ablate};
    const uint32_t q_block_tiles = sq * dht;

    compute_kernel_hw_startup(cb_q_in, cb_k_in, cb_out);
    mm_block_init(cb_q_in, cb_k_in, cb_qk_scores, 0, 1, 1, dht);

    for (uint32_t qb = 0; qb < q_count; ++qb) {
        for (uint32_t k = 0; k < k_num_chunks; ++k) {
            // exp_approx is a compile-time constant → the unused kv_step instantiation
            // is dead-code-eliminated (no binary bloat).
            if constexpr (exp_approx != 0) {
                kv_step<ckl::Approx::Fast>(k == 0, p);
            } else {
                kv_step<ckl::Approx::Exact>(k == 0, p);
            }
        }

        // O /= l, per row (recip in place on cb_l, then O·(1/l) → cb_out).
        ckl::unary<ckl::Recip<>, ckl::input(cb_l), ckl::output(cb_l)>(ckl::EltwiseShape::tiles(sq));
        ckl::mul<
            ckl::input(cb_o),
            ckl::input(cb_l, ckl::InputLifecycle::Bulk, ckl::OperandKind::Col),
            ckl::output(cb_out),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(sq, dht));

        // Release the leftover running max (never consumed) and the retained Q.
        cb_pop_front(cb_m, sq);
        cb_pop_front(cb_q_in, q_block_tiles);
    }
}
