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
#include "api/compute/pack.h"  // pack_reconfig_l1_acc (fused O-accumulate reset)
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"
#include "scaled_dot_product_attention_causal.hpp"

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
    uint32_t ablate;  // /perf-measure ablation gate: 0=normal, 1=matmul-stub, 2=+reduce-stub, 3=+exp/rescale-stub
};

// One online-softmax KV-step, updating the running (m,l,O) CBs in place.
//
// CorrExpMode / PExpMode (compile-time): the SFPU exp datapath for the two distinct
// exp sites, selected INDEPENDENTLY (Refinement 3d-a / 5a — hybrid precision recovery).
//   * PExpMode drives the phase-4 P = exp((S - m)·scale) over the WHOLE sq×sk score
//     block — the single dominant SFPU cost (Refinement 3d ablation: 21%+ of the wall).
//   * CorrExpMode drives the phase-3 corr = exp((m_prev - m_cur)·scale) over just sq
//     tiles — tiny, but it rescales the ENTIRE accumulated running (l,O) every time the
//     running max updates, so its error compounds multiplicatively across the ~74-block
//     KV loop.
// Approx::Exact is the numerically exact exp_tile; Approx::Fast selects the hardware's
// fast approximate exp_tile. Refinement 3d measured all-Fast at 1.44× (10.25→7.12 ms)
// on the flagged shape but PCC 0.9997→0.9967 (misses the 0.997 anchor by 0.0003).
// Refinement 3d-a/5a's hybrid — CorrExpMode=Exact + PExpMode=Fast — keeps the
// accuracy-critical (compounding) corr-exp exact while taking the fast path on the bulk
// P-exp, to recover PCC at (most of) the speed. The 3-value exp_mode CT arg (14) selects:
//   0 = exact  (Corr=Exact, P=Exact)  — byte-identical default, zero regression.
//   1 = fast   (Corr=Fast,  P=Fast)   — Refinement 3d's all-fast lever.
//   2 = hybrid (Corr=Exact, P=Fast)   — Refinement 3d-a/5a precision-recovery lever.
// apply_mask (runtime): add the mask block sitting in cb_mask_in to the QKᵀ scores
// before the row-max. For mask_mode=custom it is always true (streamed additive
// mask). For mask_mode=causal it is true ONLY on the diagonal-straddling KV-blocks
// (the reader generates + pushes cb_mask_in exactly for those); fully-past blocks
// carry no mask and skip the add — the reader pushes nothing for them, so the CB
// balance holds. For mask_mode=none it is always false.
template <ckl::Approx CorrExpMode, ckl::Approx PExpMode, bool Fuse>
void kv_step(bool first, bool apply_mask, const MMParams& p) {
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

    // ---- 2. + mask (custom: always; causal: straddling KV-blocks only) ----
    if (apply_mask) {
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
            ckl::Exp<CorrExpMode>{},
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
            ckl::Exp<PExpMode>{},
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

    if constexpr (Fuse) {
        // ---- 6+7 (FUSED). l update, then O = corr·O_prev + P·V in one shot ----
        // The PV matmul L1-accumulates P·V directly onto the running cb_o via the packer
        // (caller-owned Interm target + packer_l1_acc + accumulate_output), so the separate
        // cb_out_new PV target and the O += cb_out_new add pass both disappear. For !first,
        // cb_o is rescaled in place by corr BEFORE the matmul so it is the accumulator the
        // matmul adds onto. Requires full q-chunks (cb_o a single-block full ring), gated
        // host-side by fuse_oaccum = (sqt % sq_chunk_t == 0).

        // -- running l update (unchanged from the non-fused path; corr held for O rescale) --
        if (first) {
            // l = rowsum(P)
            ckl::copy<ckl::input(cb_sum_new), ckl::output(cb_l)>(ckl::EltwiseShape::tiles(sq));
        } else {
            // l = corr·l_prev + rowsum(P)   (cb_l updated in place; corr HELD)
            ckl::mul<
                ckl::input(cb_l),
                ckl::input(cb_exp_max_diff, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
                ckl::output(cb_sum_scaled)>(ckl::EltwiseShape::tiles(sq));
            ckl::add<ckl::input(cb_sum_new), ckl::input(cb_sum_scaled), ckl::output(cb_l)>(
                ckl::EltwiseShape::tiles(sq));
            // O_prev *= corr in place (corr col-broadcast across DHT; consumes corr).
            ckl::mul<
                ckl::input(cb_o),
                ckl::input(cb_exp_max_diff, ckl::InputLifecycle::Bulk, ckl::OperandKind::Col),
                ckl::output(cb_o),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::grid(sq, dht));
        }

        // -- O += P·V — PV matmul packs directly onto cb_o (caller-owned, in place) --
        if (p.ablate >= 1) {
            // matmul-stub: mirror the real path's CB ops exactly, no FPU work.
            cb_wait_front(cb_qk_scores, sq * sk);
            cb_pop_front(cb_qk_scores, sq * sk);
            cb_wait_front(cb_v_in, sk * dht);
            cb_pop_front(cb_v_in, sk * dht);
            if (first) {
                cb_reserve_back(cb_o, sq * dht);
                cb_push_back(cb_o, sq * dht);
            } else {
                cb_wait_front(cb_o, sq * dht);
                cb_pop_front(cb_o, sq * dht);
                cb_reserve_back(cb_o, sq * dht);
                cb_push_back(cb_o, sq * dht);
            }
        } else {
            CircularBuffer p_buf(cb_qk_scores), v_buf(cb_v_in), o_buf(cb_o);
            const auto pv_shape = ckl::MatmulBlockShape::of(p.pv_in0_sb, p.pv_in1_sb, p.pv_sb_h, p.pv_sb_w, sk, 1);
            if (first) {
                // Seed: block 0 overwrites (accumulate_output=false). cb_o empty → reserve/push.
                cb_reserve_back(cb_o, sq * dht);
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    ckl::LastBlockTarget::Interm,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::NoPostCompute,
                    ckl::NoPreKBlock,
                    ckl::NoPostKBlock,
                    /*untilize_block_ct_dim=*/0,
                    ckl::NoKBlockInnerDimFn,
                    ckl::NoIn0Source,
                    ckl::NoIn1BaseOffset,
                    /*caller_owns_pack_target=*/true,
                    /*accumulate_output=*/false>(
                    p_buf, v_buf, o_buf, o_buf, pv_shape, {}, {}, /*in1_per_core_w=*/0, /*out_row_width=*/dht);
                cb_push_back(cb_o, sq * dht);
            } else {
                // Accumulate: block 0 adds onto the resident corr·O (accumulate_output=true).
                // cb_o is caller-owned: pop the corr·O we just wrote (rd ptr advances; tiles
                // stay in L1), reserve (full ring → wr ptr wraps back onto them), let the
                // matmul L1-accumulate P·V in place, then publish the updated O.
                cb_wait_front(cb_o, sq * dht);
                cb_pop_front(cb_o, sq * dht);
                cb_reserve_back(cb_o, sq * dht);
                ckl::matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/true,
                    ckl::LastBlockTarget::Interm,
                    ckl::OutputCBLayout::TileRowMajor,
                    ckl::matmul_config::InitMode::Short,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::InputPolicy::WaitAndPopPerKBlock,
                    ckl::NoPostCompute,
                    ckl::NoPreKBlock,
                    ckl::NoPostKBlock,
                    /*untilize_block_ct_dim=*/0,
                    ckl::NoKBlockInnerDimFn,
                    ckl::NoIn0Source,
                    ckl::NoIn1BaseOffset,
                    /*caller_owns_pack_target=*/true,
                    /*accumulate_output=*/true>(
                    p_buf, v_buf, o_buf, o_buf, pv_shape, {}, {}, /*in1_per_core_w=*/0, /*out_row_width=*/dht);
                cb_push_back(cb_o, sq * dht);
            }
            ckernel::pack_reconfig_l1_acc(0);  // restore overwrite mode for next QKᵀ pack / normalize.
        }
    } else {
        // ---- 6. O_new = P·V  (P popped, V popped) ---- [non-fused fallback, byte-identical]
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
            ckl::add<ckl::input(cb_sum_new), ckl::input(cb_sum_scaled), ckl::output(cb_l)>(
                ckl::EltwiseShape::tiles(sq));
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
}

}  // namespace

void kernel_main() {
    const uint32_t sq = get_compile_time_arg_val(0);
    const uint32_t sk = get_compile_time_arg_val(1);
    const uint32_t dht = get_compile_time_arg_val(2);
    // mask_regime: 0=none, 1=custom (add streamed mask every KV-block),
    // 2=causal (on-device triangular mask, KV-loop truncated + mask only on the
    // diagonal-straddling blocks).
    const uint32_t mask_regime = get_compile_time_arg_val(3);
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
    // Refinement 3d / 3d-a / 5a — SFPU-floor exp lever. CT arg 14 selects the exp
    // datapath for the two exp sites (corr-exp, P-exp) independently:
    //   0 = exact  (Corr=Exact, P=Exact)  — byte-identical default, zero regression.
    //   1 = fast   (Corr=Fast,  P=Fast)   — 3d's all-fast lever (math_approx_mode=True).
    //   2 = hybrid (Corr=Exact, P=Fast)   — 3d-a/5a precision-recovery lever: keep the
    //       compounding corr-exp exact, take the fast path on the bulk P-exp.
    // Compile-time constant → the two unused kv_step instantiations are dead-code-
    // eliminated (only the selected branch is compiled; no binary bloat).
    constexpr uint32_t exp_mode = get_compile_time_arg_val(14);
    // Refinement 4 — causal masking: Q_NUM_CHUNKS lets compute recover the global
    // query-chunk index per Q-block (qc = (q_start + qb) % q_num_chunks) so it can
    // truncate the KV loop + gate the mask add identically to the reader.
    const uint32_t q_num_chunks = get_compile_time_arg_val(15);
    // Fusion #1 — fused O-accumulate. When set, the PV matmul L1-accumulates P·V directly
    // onto the running output cb_o (dropping the cb_out_new PV target + the O += PV add pass).
    // Host gates this on full q-chunks (sqt % sq_chunk_t == 0), so cb_o is a single-block
    // full ring the packer can accumulate onto in place.
    constexpr uint32_t fuse_oaccum = get_compile_time_arg_val(16);

    const uint32_t q_count = get_arg_val<uint32_t>(0);
    const uint32_t k_num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t q_start = get_arg_val<uint32_t>(2);

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
        ablate};
    const uint32_t q_block_tiles = sq * dht;

    compute_kernel_hw_startup(cb_q_in, cb_k_in, cb_out);
    mm_block_init(cb_q_in, cb_k_in, cb_qk_scores, 0, 1, 1, dht);

    for (uint32_t qb = 0; qb < q_count; ++qb) {
        // Causal (mask_regime==2): truncate the KV loop to the blocks at/before the
        // diagonal and stamp the mask only on the straddling blocks — reader agrees
        // tile-for-tile via the shared sdpa_causal predicates. custom/none run the
        // full KV loop (kv_end = k_num_chunks); apply_mask = (regime==custom).
        const uint32_t qc = (mask_regime == 2u) ? (q_start + qb) % q_num_chunks : 0u;
        const uint32_t kv_end = (mask_regime == 2u) ? sdpa_causal::kc_count(qc, sq, sk, k_num_chunks) : k_num_chunks;

        for (uint32_t k = 0; k < kv_end; ++k) {
            const bool apply_mask =
                (mask_regime == 1u) || (mask_regime == 2u && sdpa_causal::needs_mask(qc, k, sq, sk));
            // exp_mode is a compile-time constant → the two unused kv_step
            // instantiations are dead-code-eliminated (no binary bloat).
            if constexpr (exp_mode == 1u) {
                kv_step<ckl::Approx::Fast, ckl::Approx::Fast, fuse_oaccum != 0u>(k == 0, apply_mask, p);
            } else if constexpr (exp_mode == 2u) {
                kv_step<ckl::Approx::Exact, ckl::Approx::Fast, fuse_oaccum != 0u>(k == 0, apply_mask, p);
            } else {
                kv_step<ckl::Approx::Exact, ckl::Approx::Exact, fuse_oaccum != 0u>(k == 0, apply_mask, p);
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
