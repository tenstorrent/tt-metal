// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash-Attention compute kernel (online softmax, O(S) memory).
//
// Per work unit (b, h, q_block) — with q_chunk_t == k_chunk_t == 1 so every
// per-row statistic (m_i, l_i, corr) is a single tile and all column
// broadcasts use the default Scalar tile index (one row-tile, so index 0 is
// always correct). The score / prob blocks are q_chunk_t x k_chunk_t = 1 tile;
// the full S_q x S_kv matrix is never materialized.
//
// Recurrence over KV blocks j:
//   S      = Q . Kᵀ           (matmul_block, transpose)
//   S     *= scale            (mul, Scalar bcast)
//   S     += M_ij             (add, custom mask; or on-device triangular bias
//                              on the diagonal block when is_causal)
//   m_blk  = rowmax(S)        (reduce MAX REDUCE_ROW)
//   m_new  = max(m_prev,m_blk)(binary_sfpu BinaryMax)   [j>0]
//   corr   = exp(m_prev-m_new)(sub+exp chain)           [j>0]
//   P      = exp(S - m_new)   (sub<Col>+exp chain)
//   l_blk  = rowsum(P)        (reduce SUM REDUCE_ROW)
//   l_i    = corr*l_i + l_blk (mul + add)               [j>0; j==0: l_i=l_blk]
//   PV     = P . V_j          (matmul_block)
//   O_i    = corr*O_i + PV    (mul<Col> + add)          [j>0; j==0: O_i=PV]
// After loop:
//   O_i   /= l_i              (recip + mul<Col>)
//
// ── Advisory deviations from op_design.md (helper choices; algorithm/topology
//    unchanged) ───────────────────────────────────────────────────────────
//  * Running max via BinaryMax SFPU (eltwise_binary_sfpu.hpp) instead of the
//    reduce-helper Accumulate path. The design rejected a binary-max helper on
//    the premise that none exists; BinaryMax DOES exist in the helper library,
//    and the explicit path gives deterministic, easy-to-audit CB push/pop
//    balance for the persistent m_i accumulator. The online-softmax algorithm
//    (running max + rescale) is identical.
//  * Persistent accumulator CBs (cb_q_in, cb_max, cb_l) and the per-iteration
//    correction CB (cb_corr) are drained with raw cb_pop_front at the
//    work-unit / iteration boundary. The design sanctioned this only for
//    cb_q_in; held broadcast operands and running accumulators need the same
//    treatment (identical idiom to toy_variance's `cb_pop_front(cb_mean, Ht)`
//    after a WaitUpfrontNoPop hold).

#include <cstdint>

#include "api/compute/matmul.h"  // mm_block_init (boot)
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace ckl = compute_kernel_lib;

namespace {
// Inputs / scalers
constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_mask_in = 3;
constexpr uint32_t cb_causal_mask = 7;  // on-device triangular bias (causal only)
constexpr uint32_t cb_scale = 8;
constexpr uint32_t cb_scaler_max = 9;
constexpr uint32_t cb_scaler_sum = 15;
// Running stats / scratch (all q_chunk_t == 1 tile)
constexpr uint32_t cb_max = 10;       // running max m_i (persists across KV loop)
constexpr uint32_t cb_max_prev = 11;  // snapshot of m_prev
constexpr uint32_t cb_corr = 12;      // exp(m_prev - m_new)
constexpr uint32_t cb_l = 13;         // running sum l_i (persists)
constexpr uint32_t cb_l_block = 14;   // rowsum(P) for this block
constexpr uint32_t cb_m_blk = 23;     // rowmax(S) for this block
// Score / prob / output blocks
constexpr uint32_t cb_qk = 24;
constexpr uint32_t cb_p = 25;
constexpr uint32_t cb_o_acc = 26;  // running output O_i (persists)
constexpr uint32_t cb_pv = 27;
constexpr uint32_t cb_o_tmp = 28;  // corr*O_i scratch (avoid in-place block bcast)
constexpr uint32_t cb_out = 16;

constexpr uint32_t Q_CHUNK_T = 1;
constexpr uint32_t K_CHUNK_T = 1;
}  // namespace

void kernel_main() {
    constexpr uint32_t D_t = get_compile_time_arg_val(0);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(1);
    constexpr bool has_mask = get_compile_time_arg_val(2) != 0;
    constexpr bool is_causal = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t S_q_t = get_compile_time_arg_val(4);

    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);

    // Boot: single hw_configure-bearing matmul init. matmul_block's Short init
    // (default) handles all subsequent matmul re-inits; reduce/eltwise helpers
    // own their per-op inits and rely on the pack-dest / math-pack-sync init
    // that mm_block_init issues here.
    mm_block_init(cb_q_in, cb_k_in, cb_qk, /*transpose=*/1, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/D_t);

    CircularBuffer cb_q_buf(cb_q_in);
    CircularBuffer cb_k_buf(cb_k_in);
    CircularBuffer cb_v_buf(cb_v_in);
    CircularBuffer cb_qk_buf(cb_qk);
    CircularBuffer cb_p_buf(cb_p);
    CircularBuffer cb_pv_buf(cb_pv);

    using ckl::BinaryFpu;
    using ckl::BinaryFpuOp;
    using ckl::BroadcastDim;
    using ckl::EltwiseShape;
    using ckl::InputLifecycle;
    using ckl::PackTile;

    constexpr auto qk_shape = ckl::MatmulBlockShape::of(
        /*in0_sb=*/Q_CHUNK_T,
        /*in1_sb=*/K_CHUNK_T,
        /*sb_h=*/1,
        /*sb_w=*/1,
        /*in0_block_k=*/D_t,
        /*num_k_blocks=*/1);
    constexpr auto pv_shape = ckl::MatmulBlockShape::of(
        /*in0_sb=*/Q_CHUNK_T,
        /*in1_sb=*/D_t,
        /*sb_h=*/1,
        /*sb_w=*/1,
        /*in0_block_k=*/K_CHUNK_T,
        /*num_k_blocks=*/1);

    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(Q_CHUNK_T, K_CHUNK_T);

    for (uint32_t u = 0; u < num_units; ++u) {
        // Causal: this work unit owns Q tile-row qi (= global unit index mod
        // S_q_t). Only KV blocks j in [0, qi] are processed — j > qi is the
        // whole-future region (skipped, not read by the reader either) and
        // j == qi is the diagonal block that gets the on-device triangular bias.
        const uint32_t qi = is_causal ? ((start_unit + u) % S_q_t) : 0;
        const uint32_t kv_blocks = is_causal ? (qi + 1) : num_kv_blocks;

        for (uint32_t j = 0; j < kv_blocks; ++j) {
            const bool first = (j == 0);

            // ---- C: S = Q . Kᵀ  (Q retained across the KV loop, K popped) ----
            ckl::matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                cb_q_buf, cb_k_buf, cb_qk_buf, cb_q_buf /*interm placeholder (num_k_blocks==1)*/, qk_shape);

            // ---- D: S *= scale  (scalar broadcast, in place) ----
            ckl::mul<
                cb_qk,
                cb_scale,
                cb_qk,
                BroadcastDim::Scalar,
                InputLifecycle::Streaming,
                InputLifecycle::HeldStream>(K_CHUNK_T);

            // ---- E: S += M_ij  (custom mask, elementwise, in place) ----
            if constexpr (has_mask) {
                ckl::add<
                    cb_qk,
                    cb_mask_in,
                    cb_qk,
                    BroadcastDim::None,
                    InputLifecycle::Streaming,
                    InputLifecycle::Streaming>(K_CHUNK_T);
            }

            // ---- E (causal): S += triangular bias on the diagonal block ----
            // Past blocks (j < qi) are fully attended (no mask). Future blocks
            // (j > qi) are never reached (skipped by kv_blocks). Only the
            // diagonal block j == qi gets the constant upper-triangular -inf
            // bias from cb_causal_mask (HeldStream: filled once, never popped).
            if constexpr (is_causal) {
                if (j == qi) {
                    ckl::add<
                        cb_qk,
                        cb_causal_mask,
                        cb_qk,
                        BroadcastDim::None,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldStream>(K_CHUNK_T);
                }
            }

            // ---- G: m_blk = rowmax(S)  (cb_qk retained for the P subtract) ----
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(cb_qk, cb_scaler_max, cb_m_blk, reduce_shape);

            // ---- F/H: running max + correction ----
            if (first) {
                // m_i = m_blk
                ckl::copy<cb_m_blk, cb_max>(Q_CHUNK_T);
            } else {
                // snapshot m_prev (no pop — reused for corr, cb_max reused below)
                ckl::copy<cb_max, cb_max_prev, InputLifecycle::HeldStream>(Q_CHUNK_T);
                // m_new = max(m_prev, m_blk)  (in place into cb_max)
                ckl::binary_sfpu<ckl::BinaryMax<>, cb_max, cb_m_blk, cb_max>(Q_CHUNK_T);
                // corr = exp(m_prev - m_new)  (keep cb_max = m_new for the P subtract)
                ckl::eltwise_chain(
                    EltwiseShape(Q_CHUNK_T),
                    BinaryFpu<
                        cb_max_prev,
                        cb_max,
                        BinaryFpuOp::Sub,
                        BroadcastDim::None,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldStream>{},
                    ckl::Exp<>{},
                    PackTile<cb_corr>{});
            }

            // ---- I: P = exp(S - m_new)  (subtract m_new down columns, then exp) ----
            // cb_qk: popped here (was retained by the MAX reduce).
            // cb_max: HeldStream — kept as m_prev for the next KV iteration.
            ckl::eltwise_chain(
                EltwiseShape::grid(Q_CHUNK_T, K_CHUNK_T),
                BinaryFpu<
                    cb_qk,
                    cb_max,
                    BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldStream>{},
                ckl::Exp<>{},
                PackTile<cb_p>{});

            // ---- J: l_blk = rowsum(P)  (cb_p retained for the P.V matmul) ----
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(cb_p, cb_scaler_sum, cb_l_block, reduce_shape);

            if (first) {
                ckl::copy<cb_l_block, cb_l>(Q_CHUNK_T);
            } else {
                // l_i = corr*l_i + l_blk
                ckl::
                    mul<cb_l, cb_corr, cb_l, BroadcastDim::None, InputLifecycle::Streaming, InputLifecycle::HeldStream>(
                        Q_CHUNK_T);
                ckl::add<cb_l, cb_l_block, cb_l>(Q_CHUNK_T);
            }

            // ---- K: PV = P . V_j ; O_i = corr*O_i + PV ----
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::SubblockMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                cb_p_buf, cb_v_buf, cb_pv_buf, cb_p_buf /*interm placeholder*/, pv_shape);

            if (first) {
                ckl::copy<cb_pv, cb_o_acc>(D_t);
            } else {
                // O_i = corr*O_i (corr broadcast down columns across D) + PV
                ckl::mul<
                    cb_o_acc,
                    cb_corr,
                    cb_o_tmp,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::HeldStream>(EltwiseShape::grid(Q_CHUNK_T, D_t));
                ckl::add<cb_o_tmp, cb_pv, cb_o_acc>(D_t);
                // corr was held by both the l and O updates — drain it now.
                cb_pop_front(cb_corr, Q_CHUNK_T);
            }
        }

        // ---- L: 1/l_i ----
        ckl::unary<ckl::Recip<>, cb_l, cb_l>(Q_CHUNK_T);

        // ---- M: O_i /= l_i  (1/l_i broadcast down columns across D) -> cb_out ----
        ckl::mul<cb_o_acc, cb_l, cb_out, BroadcastDim::Col, InputLifecycle::Streaming, InputLifecycle::HeldStream>(
            EltwiseShape::grid(Q_CHUNK_T, D_t));

        // ---- N: release per-work-unit persistent / held CBs ----
        cb_pop_front(cb_q_in, D_t);       // Q retained by every QKᵀ matmul
        cb_pop_front(cb_max, Q_CHUNK_T);  // final m_i
        cb_pop_front(cb_l, Q_CHUNK_T);    // 1/l_i held by the normalize mul
    }
}
