// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// KV-OUTER compute kernel for scaled_dot_product_attention (FlashAttention-2),
// the NoC-multicast (KV read-once + broadcast) variant.
//
// WHY A SEPARATE KERNEL FROM THE PRODUCTION Q-OUTER COMPUTE:
// Multicast requires all cores in a row-group to consume the SAME KV chunk at
// the SAME time (one DRAM read of K/V, broadcast to the whole row), which forces
// the loop KV-OUTER: for each KV chunk j, every core folds j into EACH of the
// several q-chunks it owns. So this kernel holds up to MAX_SUBCHUNK q-chunks'
// (m,l,O,q_scaled) state RESIDENT — the "restructure inflates resident state"
// cost the roofline flagged (perf_findings.md § NoC-multicast).
//
// SINGLE CODE PATH (not unrolled per sub-chunk): a 4x unroll blew the L1 kernel-
// config buffer (Program size too large). Instead the online-softmax state for
// the sub-chunks a core owns lives in ROTATING resident CBs (cb_row_max/row_sum/
// out_accum each hold q_cnt blocks). Processing sub-chunks 0..q_cnt-1 in FIFO
// order each KV chunk is one full rotation: pop the front (= sub-chunk s's state),
// fold this KV chunk in, push the updated state to the back. The FIFO order is
// preserved across KV chunks, so sub-chunk s's state is always the s-th popped;
// the physical slot drift is invisible. q_scaled is read-only so it can't ride a
// pop/push rotation — it stays in MAX_SUBCHUNK separate CBs and the QKᵀ matmul
// reads q_scaled[s] via In0SourceFn (a runtime CB index), one matmul instantiation.
//
// The per-phase online-softmax math mirrors the shipped kernel's non-fused
// branches (helpers, broadcast dims, first-chunk init m=-inf => alpha=0), with
// fast SFPU exp (the biggest compute win). l = alpha*l + rowsum and O = alpha*O +
// PV are each done as mul(->temp) + add(->resident) so the resident CB sees
// exactly ONE pop+push (one rotation step) per sub-chunk.
//
// Guard (host): bf16, self-attn, MHA, mask none, tile-aligned,
// fp32_dest_acc_en=False, divisible chunks, one (b,h) per row-group, subchunk<=4.

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
// ---- CB indices (must match the descriptor + reader/writer) ----
constexpr uint32_t cb_q_stage = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_scale = 4;
constexpr uint32_t cb_scores = 5;
constexpr uint32_t cb_exp = 6;
constexpr uint32_t cb_corr = 7;
constexpr uint32_t cb_m_new = 8;
constexpr uint32_t cb_sum_chunk = 9;
constexpr uint32_t CB_Q_SCALED_0 = 10;  // 10..13 (read-only, one per sub-chunk)
constexpr uint32_t cb_pv = 14;
constexpr uint32_t cb_l_tmp = 15;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_row_max = 17;    // ROTATING: q_cnt blocks (running m)
constexpr uint32_t cb_row_sum = 18;    // ROTATING: q_cnt blocks (running l)
constexpr uint32_t cb_out_accum = 19;  // ROTATING: q_cnt blocks (running O)
constexpr uint32_t cb_o_tmp = 20;
constexpr uint32_t cb_m_old = 21;  // depth-1 scratch: old running max popped off the rotating ring
constexpr uint32_t MAX_SUBCHUNK = 4;

// ROTATION-vs-ABSOLUTE-ADDRESSING NOTE: OperandKind::Block indexes a held operand
// ABSOLUTELY (base + i from the CB's slot 0), NOT front-relative. That is fine for
// a depth-1 CB (front == base) but WRONG for a rotating multi-block ring: once the
// ring rotates the current block is no longer at slot 0. So the running max is
// POPPED off cb_row_max (front-relative) into the depth-1 scratch cb_m_old before
// any held/Block read, and normalize reciprocals into a depth-1 scratch too — never
// a held/Block read or in-place op directly on a rotating ring.

// Per-K-block in0 source functor: returns a runtime-selected q_scaled CB so ONE
// QKᵀ matmul instantiation reads sub-chunk s's scaled Q (all q_scaled CBs share
// the bf16 format the unpacker is configured for at boot).
struct QScaledSource {
    uint32_t cb;
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t /*bound_cb*/) const { return cb; }
};

// Matmul N-subblock decomposition: largest divisor of n that is <= dest_limit.
FORCE_INLINE void decomp_n(uint32_t n, uint32_t dest_limit, uint32_t& in1_num_subblocks, uint32_t& out_subblock_w) {
    uint32_t w = (n < dest_limit) ? n : dest_limit;
    while (w > 1 && (n % w) != 0) {
        --w;
    }
    out_subblock_w = w;
    in1_num_subblocks = n / w;
}

// Fold one KV chunk into sub-chunk `s`'s running (m, l, O). `first` (j==0) seeds
// the state (m=chunk_max, l=rowsum, O=PV; no rescale) and FILLS the rotating rings;
// otherwise applies the alpha recurrence and ROTATES each ring by one block.
FORCE_INLINE void fold_kv(uint32_t s, bool first) {
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

    constexpr uint32_t Dt_c = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t dest_limit = get_compile_time_arg_val(5);
    constexpr bool use_fast_exp = get_compile_time_arg_val(6) != 0;
    constexpr ckl::Approx exp_approx = use_fast_exp ? ckl::Approx::Fast : ckl::Approx::Exact;
    constexpr auto RCFG = ckl::BinaryDataFormatReconfig::None;  // uniform bf16 chain
    constexpr auto PRCFG = ckl::PackTileReconfig::None;
    constexpr uint32_t sq_dt = Sq_chunk_t * Dt_c;

    uint32_t qk_in1_sb = 0, qk_out_sb_w = 0;
    decomp_n(Skv_chunk_t, dest_limit, qk_in1_sb, qk_out_sb_w);
    uint32_t pv_in1_sb = 0, pv_out_sb_w = 0;
    decomp_n(Dt_c, dest_limit, pv_in1_sb, pv_out_sb_w);

    ::CircularBuffer qs_buf(CB_Q_SCALED_0), k_buf(cb_k_in), scores_buf(cb_scores);
    ::CircularBuffer exp_buf(cb_exp), v_buf(cb_v_in), pv_buf(cb_pv);

    // Phase 2: scores = Q_scaled[s] . K^T. q_scaled[s] via In0SourceFn (retained);
    // K retained across all sub-chunks of this KV chunk (popped by the caller after).
    ckl::matmul_block<
        /*transpose=*/true,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndRetainOnLastBlock,
        ckl::InputPolicy::WaitAndRetainOnLastBlock,
        ckl::NoPostCompute,
        ckl::NoPreKBlock,
        ckl::NoPostKBlock,
        /*untilize_block_ct_dim=*/0,
        ckl::NoKBlockInnerDimFn,
        QScaledSource,
        ckl::NoIn1BaseOffset>(
        qs_buf,
        k_buf,
        scores_buf,
        scores_buf,
        MatmulBlockShape::of(Sq_chunk_t, qk_in1_sb, 1, qk_out_sb_w, Dt_c, 1),
        {},
        {},
        /*in1_per_core_w=*/0,
        /*out_row_width=*/0,
        {},
        {},
        QScaledSource{CB_Q_SCALED_0 + s});

    // Phase 4: chunk row-max -> cb_corr (cb_scores retained for the exp below).
    ckl::reduce<
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        cb_scores,
        cb_scaler,
        cb_corr,
        ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, Skv_chunk_t, Sq_chunk_t));

    // Phase 5: current running max -> cb_m_new; alpha -> cb_corr (non-first). The
    // running max cb_row_max is popped here (non-first) and re-pushed after the exp.
    if (first) {
        ckl::copy<cb_corr, cb_m_new>(EltwiseShape::tiles(Sq_chunk_t));  // m = chunk_max
    } else {
        // Pop the old running max off the rotating ring FRONT into a depth-1 scratch
        // so the held/Block reads below are front-correct (see the ROTATION note).
        ckl::copy<cb_row_max, cb_m_old>(EltwiseShape::tiles(Sq_chunk_t));
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            cb_corr,
            cb_m_old,
            cb_m_new,
            InputLifecycle::Streaming,
            InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming,
            PRCFG,
            OperandKind::Scalar,
            OperandKind::Block>(EltwiseShape::tiles(Sq_chunk_t));  // m_new = max(chunk_max, m_old); m_old held
        ckl::eltwise_chain(
            EltwiseShape::tiles(Sq_chunk_t),
            ckl::BinaryFpu<
                cb_m_old,
                cb_m_new,
                ckl::BinaryFpuOp::Sub,
                BroadcastDim::None,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                RCFG,
                Dst::D0,
                OperandKind::Scalar,
                OperandKind::Block>{},
            ckl::Exp<>{},
            ckl::PackTile<cb_corr, OutputLifecycle::Streaming>{});  // alpha = exp(m_old - m_new); m_old popped
    }

    // Phase 6: P = exp(scores - m_new) -> cb_exp (Col bcast; scores popped, m_new held).
    ckl::eltwise_chain(
        EltwiseShape::grid(Sq_chunk_t, Skv_chunk_t),
        ckl::BinaryFpu<
            cb_scores,
            cb_m_new,
            ckl::BinaryFpuOp::Sub,
            BroadcastDim::Col,
            InputLifecycle::Bulk,
            InputLifecycle::HeldBulk,
            RCFG,
            Dst::D0,
            OperandKind::Block,
            OperandKind::Col>{},
        ckl::Exp<exp_approx>{},
        ckl::PackTile<cb_exp, OutputLifecycle::Bulk>{});

    // Push the new running max -> cb_row_max (FILLS the ring on the first KV chunk,
    // COMPLETES the rotation on later ones since phase 5 popped the old max).
    ckl::copy<cb_m_new, cb_row_max>(EltwiseShape::tiles(Sq_chunk_t));

    // Phase 7: chunk row-sum + running l. First seeds l = rowsum(P). Otherwise
    // l = alpha*l + rowsum, done as mul(->cb_l_tmp) + add(->cb_row_sum) so cb_row_sum
    // rotates by exactly one (pop old l via the mul, push new l via the add).
    if (first) {
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_exp,
            cb_scaler,
            cb_row_sum,
            ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, Skv_chunk_t, Sq_chunk_t));
    } else {
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_exp,
            cb_scaler,
            cb_sum_chunk,
            ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, Skv_chunk_t, Sq_chunk_t));
        ckl::mul<
            cb_row_sum,
            cb_corr,
            cb_l_tmp,
            BroadcastDim::None,
            InputLifecycle::Streaming,
            InputLifecycle::HeldBulk,
            OutputLifecycle::Streaming,
            RCFG,
            PRCFG,
            OperandKind::Scalar,
            OperandKind::Block>(EltwiseShape::tiles(Sq_chunk_t));  // l_tmp = alpha*l_old; corr held for O
        ckl::add<cb_l_tmp, cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(Sq_chunk_t));  // l = l_tmp + rowsum
    }

    // Phase 9: PV = P . V (cb_exp popped; V retained across sub-chunks).
    ckl::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::SubblockMajor,
        ckl::matmul_config::InitMode::Short,
        ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::InputPolicy::WaitAndRetainOnLastBlock>(
        exp_buf, v_buf, pv_buf, pv_buf, MatmulBlockShape::of(Sq_chunk_t, pv_in1_sb, 1, pv_out_sb_w, Skv_chunk_t, 1));

    // Phase 8+10: O update. First seeds O = PV. Otherwise O = alpha*O + PV, done as
    // mul(->cb_o_tmp) + add(->cb_out_accum) so cb_out_accum rotates by exactly one.
    if (first) {
        ckl::copy<cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));  // O = PV (fills ring)
    } else {
        ckl::mul<
            cb_out_accum,
            cb_corr,
            cb_o_tmp,
            BroadcastDim::Col,
            InputLifecycle::Streaming,
            InputLifecycle::Bulk,
            OutputLifecycle::Streaming,
            RCFG,
            PRCFG,
            OperandKind::Scalar,
            OperandKind::Col>(EltwiseShape::grid(Sq_chunk_t, Dt_c));  // o_tmp = alpha*O_old; corr popped (last use)
        ckl::add<cb_o_tmp, cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));  // O = o_tmp + PV
    }
}

// Phase 1: pre-scale sub-chunk S's raw Q (cb_q_stage) by the attention scale ->
// resident cb_q_scaled[S]. Unrolled over MAX_SUBCHUNK (each is one tiny mul).
template <uint32_t S>
FORCE_INLINE void prescale_subchunk() {
    using ckl::BroadcastDim;
    using ckl::EltwiseShape;
    using ckl::InputLifecycle;
    using ckl::OperandKind;
    using ckl::OutputLifecycle;
    constexpr uint32_t Dt_c = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    ckl::mul<
        cb_q_stage,
        cb_scale,
        CB_Q_SCALED_0 + S,
        BroadcastDim::Scalar,
        InputLifecycle::Streaming,
        InputLifecycle::HeldBulk,
        OutputLifecycle::Streaming,
        ckl::BinaryDataFormatReconfig::None,
        ckl::PackTileReconfig::None,
        OperandKind::Scalar,
        OperandKind::Scalar>(EltwiseShape::tiles(Sq_chunk_t * Dt_c));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt_c = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t n_kv_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t max_subchunk = get_compile_time_arg_val(4);
    constexpr uint32_t dest_limit = get_compile_time_arg_val(5);
    static_assert(max_subchunk <= MAX_SUBCHUNK, "host subchunk cap exceeds the kernel's resident-CB layout");

    const uint32_t q_cnt = get_arg_val<uint32_t>(0);  // sub-chunks this core owns (>=1, <= MAX_SUBCHUNK)

    constexpr uint32_t q_tiles = Sq_chunk_t * Dt_c;
    constexpr uint32_t kv_tiles = Skv_chunk_t * Dt_c;

    // Boot: matmul-order hw config + one full matmul_block_init.
    uint32_t qk_in1_sb = 0, qk_out_sb_w = 0;
    decomp_n(Skv_chunk_t, dest_limit, qk_in1_sb, qk_out_sb_w);
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(CB_Q_SCALED_0, cb_k_in, cb_scores);
    matmul_block_init(CB_Q_SCALED_0, cb_k_in, /*transpose=*/1, qk_out_sb_w, /*out_subblock_h=*/1, Dt_c);

    // Phase 1: pre-scale each owned sub-chunk's Q -> resident cb_q_scaled[s].
    if (0 < q_cnt) {
        prescale_subchunk<0>();
    }
    if (1 < q_cnt) {
        prescale_subchunk<1>();
    }
    if (2 < q_cnt) {
        prescale_subchunk<2>();
    }
    if (3 < q_cnt) {
        prescale_subchunk<3>();
    }

    // KV-OUTER loop: fold each KV chunk into every owned sub-chunk (one full
    // rotation of the resident rings), then release the retained K/V once.
    for (uint32_t j = 0; j < n_kv_chunks; ++j) {
        const bool first = (j == 0);
        for (uint32_t s = 0; s < q_cnt; ++s) {
            fold_kv(s, first);
        }
        cb_pop_front(cb_k_in, kv_tiles);
        cb_pop_front(cb_v_in, kv_tiles);
    }

    // Phase 11: normalize + emit each owned sub-chunk's output (rings drain in
    // FIFO order 0..q_cnt-1, matching the writer's drain order). Reciprocal into the
    // depth-1 scratch cb_l_tmp (never in-place on the rotating ring — that would
    // rotate cb_row_sum out from under the following mul; see the ROTATION note).
    for (uint32_t s = 0; s < q_cnt; ++s) {
        ckl::unary<ckl::Recip<>, cb_row_sum, cb_l_tmp>(ckl::EltwiseShape::tiles(Sq_chunk_t));
        ckl::mul<
            cb_out_accum,
            cb_l_tmp,
            cb_out,
            ckl::BroadcastDim::Col,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Bulk,
            ckl::OutputLifecycle::Streaming,
            ckl::BinaryDataFormatReconfig::None,
            ckl::PackTileReconfig::None,
            ckl::OperandKind::Scalar,
            ckl::OperandKind::Col>(ckl::EltwiseShape::grid(Sq_chunk_t, Dt_c));
        cb_pop_front(CB_Q_SCALED_0 + s, q_tiles);  // release the retained scaled-Q block
    }
}
