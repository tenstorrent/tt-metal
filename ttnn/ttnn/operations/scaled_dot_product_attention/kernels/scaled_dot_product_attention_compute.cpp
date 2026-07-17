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
//
// ---------------------------------------------------------------------------
// RAW-LLK HELPER SUBSTITUTION (R3e) — the fused exp + row-sum dual-pack.
// ---------------------------------------------------------------------------
// Phase 6 (P = exp(scores - m)) has TWO variants selected at compile time by
// `fuse_rowsum` (host-gated to fp32_dest_acc_en=False, the throughput regime):
//   * fuse_rowsum == false: the helper eltwise_chain (sub<Col> -> Exp -> Pack),
//     byte-identical to R3c/R3d. A separate reduce<SUM,REDUCE_ROW> computes the
//     per-chunk row-sum (phase 7). This is the max-precision path, untouched.
//   * fuse_rowsum == true: `fused_exp_dual_pack` (raw LLK) packs the exp result
//     to cb_exp AND, in the SAME DEST window, L1-accumulates the element-wise
//     partial row-sum across the skv column tiles into cb_sum_chunk[i] via
//     `pack_tile<true>` + `pack_reconfig_l1_acc`. The dedicated per-chunk
//     reduce<SUM> is then eliminated; the running sum is kept in partial
//     (rows x 1, 32-col, un-reduced) form across the KV loop and collapsed to a
//     scalar ONCE after the loop (phase 11) with a single reduce<SUM,REDUCE_ROW>
//     (the FPU matmul-with-ones). This is production SDPA's approach
//     (compute_common.hpp sub_exp_block_bcast_cols_inplace do_reduce +
//     matmul_reduce); it is exact because rowsum is linear, so it commutes with
//     the per-chunk alpha rescale and the running accumulate.
//
// WHY RAW LLK (helper limitation, not a shortcut): the kernel_lib eltwise chain
// is SINGLE-TERMINAL — a chain that uses L1 accumulation (OutputLifecycle::
// L1Accumulation) must have EVERY PackTile be L1-accumulating and target ONE CB
// (eltwise_chain.inl static_asserts `all_writers_l1_accumulation` and
// `pack_dfbs_consistent`). So a chain CANNOT co-pack a normal cb_exp AND an
// L1-accumulating cb_sum_chunk from one DEST window. The fused dual-pack is
// therefore not expressible with helpers and is hand-written here. cb_exp and
// cb_sum_chunk share `interm_format` (bf16 in this regime), so the two pack
// targets need no pack_reconfig_data_format between them.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/bcast.h"                 // sub_tiles_bcast_cols, sub_bcast_cols_init_short
#include "api/compute/eltwise_unary/exp.h"     // exp_tile, exp_tile_init
#include "api/compute/pack.h"                  // pack_tile, pack_reconfig_l1_acc
#include "api/compute/reconfig_data_format.h"  // reconfig_data_format, pack_reconfig_data_format
#include "api/compute/reg_api.h"               // tile_regs_acquire/commit/wait/release

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace ckl = compute_kernel_lib;

namespace {
// Derive the matmul N-subblock decomposition for an N tile-count (R1b). Replaces the
// phase-0 host `_matmul_subblocks` (single source of truth now on-device — the
// partial chunk's subblock count re-derives without a second host formula).
// out_subblock_h is always 1 (=> SubblockMajor output is tile-row-major, which the
// reduce and Col-broadcast steps downstream require), so in0_num_subblocks = M is
// passed directly and only the N split (in1_num_subblocks, out_subblock_w) is derived
// here. Largest divisor of n that is <= dest_limit. n >= 1 always, so out_subblock_w
// lands in [1, min(n, dest_limit)].
FORCE_INLINE void decomp_n(uint32_t n, uint32_t dest_limit, uint32_t& in1_num_subblocks, uint32_t& out_subblock_w) {
    uint32_t w = (n < dest_limit) ? n : dest_limit;
    while (w > 1 && (n % w) != 0) {
        --w;
    }
    out_subblock_w = w;
    in1_num_subblocks = n / w;
}

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

// R3e: fused exp + partial-row-sum dual-pack (raw LLK — see file-head comment).
// P = exp(scores - m) is packed to cb_exp (row-major, for the PV matmul) AND, in
// the SAME DEST window, the element-wise partial row-sum across this chunk's
// skv_valid column tiles is L1-accumulated into cb_sum_chunk[i] (rows x 1, 32-col,
// un-reduced). No dedicated per-chunk reduce<SUM> read of cb_exp; the intra-tile
// column collapse is deferred to a single reduce after the KV loop (phase 11).
//
// Reached only when fp32_dest_acc_en=False, so DEST is 16 bf16 tiles (8 per
// half-sync section) and skv_valid <= Skv_chunk_t <= 8 fits one acquire section.
// cb_scores/cb_exp/cb_sum_chunk are all interm_format (bf16) here, so the two
// pack targets share a data format (no pack_reconfig_data_format between them).
FORCE_INLINE void fused_exp_dual_pack(uint32_t sq_valid, uint32_t skv_valid) {
    const uint32_t sq_skv = sq_valid * skv_valid;
    cb_wait_front(cb_scores, sq_skv);
    cb_wait_front(cb_row_max, sq_valid);  // running max (held; NOT popped by this phase)
    cb_reserve_back(cb_exp, sq_skv);
    cb_reserve_back(cb_sum_chunk, sq_valid);

    // LIGHTWEIGHT reconfig (mirrors the eltwise_chain's own lowering — see file-head
    // note), NOT a full init_bcast: reconfig the unpack/math src formats to
    // (scores, row_max), switch the math/unpack op-type to sub-bcast-col, init the
    // fast SFPU exp (matches ckl::Exp<Approx::Fast> == exp_tile<true>, default
    // ClampToNegative), and reconfig ONLY the pack DATA FORMAT to cb_exp. A full
    // init_bcast (llk_pack_hw_configure + llk_pack_init) would re-init the packer and
    // clobber the boot-time matmul_block_init packer state that the per-KV-chunk
    // matmul's InitMode::Short relies on (Short does not fully re-issue packer cfg),
    // drifting across chunks x work-units. L1-acc is forced off so the cb_exp packs
    // below never accumulate.
    ckernel::reconfig_data_format(cb_scores, cb_row_max);
    ckernel::sub_bcast_cols_init_short(cb_scores, cb_row_max);
    ckernel::exp_tile_init<true>();
    ckernel::pack_reconfig_data_format(cb_exp);
    ckernel::pack_reconfig_l1_acc(0);

    for (uint32_t i = 0; i < sq_valid; ++i) {
        ckernel::tile_regs_acquire();
        for (uint32_t j = 0; j < skv_valid; ++j) {
            // DEST[j] = scores[i,j] - m[i]  (per-row max broadcast across columns)
            ckernel::sub_tiles_bcast_cols(cb_scores, cb_row_max, i * skv_valid + j, i, j);
            ckernel::exp_tile<true>(j);
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        // (a) normal pack: exp -> cb_exp[i*skv_valid + j] (l1_acc still 0).
        for (uint32_t j = 0; j < skv_valid; ++j) {
            ckernel::pack_tile<true>(j, cb_exp, i * skv_valid + j);
        }
        // (b) L1-accumulate the row's skv_valid column tiles into cb_sum_chunk[i]:
        // seed with the first column (l1_acc=0 => overwrite), accumulate the rest
        // (l1_acc=1). DEST is unchanged by the (a) packs, so it is re-read here.
        ckernel::pack_tile<true>(0, cb_sum_chunk, i);
        ckernel::pack_reconfig_l1_acc(1);
        for (uint32_t j = 1; j < skv_valid; ++j) {
            ckernel::pack_tile<true>(j, cb_sum_chunk, i);
        }
        ckernel::pack_reconfig_l1_acc(0);  // reset before next row / downstream ops
        ckernel::tile_regs_release();
    }
    cb_push_back(cb_exp, sq_skv);
    cb_push_back(cb_sum_chunk, sq_valid);
    cb_pop_front(cb_scores, sq_skv);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t n_kv_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t has_mask_v = get_compile_time_arg_val(4);
    constexpr bool has_mask = has_mask_v != 0;
    constexpr uint32_t skv_partial = get_compile_time_arg_val(5);  // valid cols in last S_kv tile (0 => aligned)
    constexpr bool has_kv_pad = skv_partial != 0;
    constexpr uint32_t Sq_t = get_compile_time_arg_val(6);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(8);
    constexpr uint32_t dest_limit = get_compile_time_arg_val(9);
    // R3c (perf): fast/approximate SFPU exp for the dominant P=exp phase, gated by
    // the host to the fp32_dest_acc_en=False throughput regime only (loose tolerance
    // absorbs the approximation; the max-precision regime stays exact -> no
    // regression). The exp's Approx template is a compile-time value.
    constexpr bool use_fast_exp = get_compile_time_arg_val(10) != 0;
    constexpr ckl::Approx exp_approx = use_fast_exp ? ckl::Approx::Fast : ckl::Approx::Exact;
    // R3e (perf): fuse the per-chunk row-sum reduce into the exp pack via packer L1
    // accumulation (raw-LLK dual-pack — see file-head comment). Host-gated to the
    // fp32_dest_acc_en=False throughput regime (== use_fast_exp), so the softmax
    // intermediates are bf16 and the max-precision path stays byte-identical.
    constexpr bool fuse_rowsum = get_compile_time_arg_val(11) != 0;

    const uint32_t num_wu = get_arg_val<uint32_t>(0);
    const uint32_t start_wu = get_arg_val<uint32_t>(1);

    // PV pack width (N = Dt, constant across the loop) — use the optimal decomposition.
    uint32_t pv_in1_sb = 0, pv_out_sb_w = 0;
    decomp_n(Dt, dest_limit, pv_in1_sb, pv_out_sb_w);
    // QKᵀ pack width (out_subblock_w) must stay CONSTANT across the KV loop:
    // mm_block_init_short (the per-call InitMode::Short) does not fully re-issue the
    // packer config, so changing out_subblock_w mid-loop to a non-power-of-2 partial
    // width (e.g. 4 -> 3) wedges the packer (are_packers_configured_correctly). When
    // Skv_t divides Skv_chunk_t there is no partial chunk (incl. the perf-flagged
    // shape, Skv_t%4==0) -> keep the optimal full-width decomposition; otherwise force
    // out_subblock_w = 1 (safe constant; only the small generality shapes with a
    // partial KV chunk pay the narrower subblock — the perf path is untouched). Only
    // the subblock COUNT (in1_num_subblocks) then varies per chunk, which is a loop
    // bound, not a packer reconfig.
    constexpr bool has_partial_kv = (Skv_t % Skv_chunk_t) != 0;
    uint32_t qk_full_in1_sb = 0, qk_full_out_sb_w = 0;
    decomp_n(Skv_chunk_t, dest_limit, qk_full_in1_sb, qk_full_out_sb_w);
    const uint32_t qk_out_sb_w = has_partial_kv ? 1u : qk_full_out_sb_w;

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
    matmul_block_init(cb_q_scaled, cb_k_in, /*transpose=*/1, qk_out_sb_w, /*out_subblock_h=*/1, Dt);

    for (uint32_t wu = 0; wu < num_wu; ++wu) {
        // R1b: partial q-chunk. Decode qc for this work unit (qc = w % n_q_chunks,
        // since n_q_chunks divides H*n_q_chunks) and take only the valid tile-rows
        // of the last q-chunk. sq_valid is the M extent for every phase below.
        const uint32_t w = start_wu + wu;
        const uint32_t qc = w % n_q_chunks;
        const uint32_t sq_off = qc * Sq_chunk_t;
        const uint32_t sq_valid = (Sq_chunk_t < Sq_t - sq_off) ? Sq_chunk_t : (Sq_t - sq_off);
        const uint32_t sq_dt = sq_valid * Dt;

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
            // R1b: partial KV chunk. skv_valid is the QKᵀ N extent, PV K extent, and
            // score-block width for this chunk (only the last chunk is < Skv_chunk_t).
            const uint32_t skv_off = j * Skv_chunk_t;
            const uint32_t skv_valid = (Skv_chunk_t < Skv_t - skv_off) ? Skv_chunk_t : (Skv_t - skv_off);
            const uint32_t sq_skv = sq_valid * skv_valid;
            // QKᵀ N subblock COUNT for this chunk (out_subblock_w is the loop-constant
            // qk_out_sb_w). qk_out_sb_w divides skv_valid: it is 1 when a partial chunk
            // exists, else the full-width divisor of Skv_chunk_t (all chunks full).
            const uint32_t qk_in1_sb = skv_valid / qk_out_sb_w;

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
                MatmulBlockShape::of(sq_valid, qk_in1_sb, 1, qk_out_sb_w, Dt, 1));

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
                ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, skv_valid, sq_valid));

            // Phase 5: update running max m, form correction alpha.
            if (first) {
                ckl::copy<cb_corr, cb_row_max>(EltwiseShape::tiles(sq_valid));  // m = chunk_max
            } else {
                // m_new = max(chunk_max, m_old) -> cb_m_new (m_old held for alpha).
                // The held operand is walked with Block indexing (absolute base+i):
                // Scalar+HeldStream never advances the front (no pop), so it would
                // re-read tile 0 for every sq_valid>1 iteration.
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
                    OperandKind::Block>(EltwiseShape::tiles(sq_valid));
                // alpha = exp(m_old - m_new) -> cb_corr (m_old popped, m_new held).
                ckl::eltwise_chain(
                    EltwiseShape::tiles(sq_valid),
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
                ckl::copy<cb_m_new, cb_row_max>(EltwiseShape::tiles(sq_valid));
            }

            // Phase 6: P = exp(scores - m) -> cb_exp (Col bcast; scores popped, m held).
            // R3c: fast SFPU exp for the dominant softmax P=exp phase (measured ~54%
            // of per-chunk compute as the exact exp; the fast exp is ~75% cheaper ->
            // flagged shape 9.01->5.80 ms = 1.55x). `exp_approx` is Fast only in the
            // fp32_dest_acc_en=False throughput regime (host-gated); the max-precision
            // regime stays Exact (byte-identical, no regression). The alpha-correction
            // exp (phase 5) stays exact always to protect the running (m, l, O).
            if constexpr (fuse_rowsum) {
                // R3e: raw-LLK dual-pack — cb_exp AND the partial row-sum (cb_sum_chunk)
                // from ONE exp DEST window (fast exp; scores popped, m held). Replaces
                // the phase-7 per-chunk reduce<SUM> below.
                fused_exp_dual_pack(sq_valid, skv_valid);
            } else {
                ckl::eltwise_chain(
                    EltwiseShape::grid(sq_valid, skv_valid),
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
                    ckl::Exp<exp_approx>{},
                    ckl::PackTile<cb_exp, OutputLifecycle::Bulk>{});
            }

            // Phase 7: chunk row-sum + running sum l update.
            if constexpr (fuse_rowsum) {
                // R3e: the partial (rows x 1, 32-col, un-reduced) row-sum for this chunk
                // is already in cb_sum_chunk (from the dual-pack above). Keep the running
                // sum l in the SAME partial form across the KV loop: alpha-rescale +
                // accumulate here, collapse to a scalar ONCE after the loop (phase 11).
                if (first) {
                    // l_partial = chunk_partial_sum (bf16 -> fp32).
                    ckl::copy<cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(sq_valid));
                } else {
                    // l_partial = alpha * l_partial (alpha col-vector bcast across the 32
                    // cols; alpha held for phase 8's O rescale).
                    ckl::mul<
                        cb_row_sum,
                        cb_corr,
                        cb_row_sum,
                        BroadcastDim::Col,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        OperandKind::Scalar,
                        OperandKind::Col>(EltwiseShape::grid(sq_valid, 1));
                    // l_partial += chunk_partial_sum.
                    ckl::add<cb_row_sum, cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(sq_valid));
                }
            } else if (first) {
                // l = chunk_sum (cb_exp held for PV).
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_exp,
                    cb_scaler,
                    cb_row_sum,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, skv_valid, sq_valid));
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
                    OperandKind::Block>(EltwiseShape::tiles(sq_valid));
                // chunk_sum -> cb_sum_chunk (cb_exp held for PV).
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_exp,
                    cb_scaler,
                    cb_sum_chunk,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, skv_valid, sq_valid));
                // l += chunk_sum.
                ckl::add<cb_row_sum, cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(sq_valid));
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
                    OperandKind::Col>(EltwiseShape::grid(sq_valid, Dt));
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
                MatmulBlockShape::of(sq_valid, pv_in1_sb, 1, pv_out_sb_w, skv_valid, 1));

            // Phase 10: accumulate O += PV.
            if (first) {
                ckl::copy<cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));  // O = PV
            } else {
                ckl::add<cb_out_accum, cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));
            }
        }

        // Phase 11: normalize O = O * (1/l), pack to bf16 -> cb_out.
        if constexpr (fuse_rowsum) {
            // R3e: collapse the partial (32-col) running sum to the scalar denominator
            // l ONCE per q-chunk — a single reduce<SUM,REDUCE_ROW> (the FPU
            // matmul-with-ones), amortized over the whole KV loop instead of one
            // reduce per chunk. -> cb_corr (free after the loop; pops cb_row_sum).
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_row_sum,
                cb_scaler,
                cb_corr,
                ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, 1, sq_valid));
            ckl::unary<ckl::Recip<>, cb_corr, cb_corr>(EltwiseShape::tiles(sq_valid));
            ckl::mul<
                cb_out_accum,
                cb_corr,
                cb_out,
                BroadcastDim::Col,
                InputLifecycle::Streaming,
                InputLifecycle::Bulk,
                OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                OperandKind::Scalar,
                OperandKind::Col>(EltwiseShape::grid(sq_valid, Dt));
        } else {
            ckl::unary<ckl::Recip<>, cb_row_sum, cb_row_sum>(EltwiseShape::tiles(sq_valid));
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
                OperandKind::Col>(EltwiseShape::grid(sq_valid, Dt));
        }

        // Release the retained Q-scaled block and the running max for the next q-chunk.
        cb_pop_front(cb_q_scaled, sq_dt);
        cb_pop_front(cb_row_max, sq_valid);
    }
}
