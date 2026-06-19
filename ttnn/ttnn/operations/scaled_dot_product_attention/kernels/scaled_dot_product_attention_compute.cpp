// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for Flash-Attention SDPA (online softmax).
//
// Per work item (b, h_q, qb): stream KV blocks updating a running max (m),
// running exp-sum (l), and running weighted output (O); finally normalize
// O = O / l. All score-bearing CBs are sized to one B_q x B_kv block
// (Flash constraint) — the full S_q x S_kv matrix is never materialized.
//
// Scale is folded into the SCORES (per KV block, in place on cb_qk) rather than
// into Q. Folding into Q would require an in-place SFPU transform on the
// reader-fed cb_q_in, which leaves cb_q_in with zero tiles visible to the QK
// matmul's unpacker (in-place pop+push on a remote-producer CB nets to zero
// available for the downstream same-thread cb_wait_front -> UNPACK hang).
// (Q*scale) . K^T == scale * (Q . K^T), so the score-scale is mathematically
// identical and the in-place transform on cb_qk (a locally matmul-produced CB)
// is the legal pattern.
//
// Online-softmax recurrence per KV block j:
//   S_j   = scale * (Q . K_j^T)            (QK matmul transpose=true, then scale)
//   S_j  += mask_ij                        (if mask present)
//   m_j   = max(m_{j-1}, rowmax(S_j))
//   alpha = exp(m_{j-1} - m_j)             (j>0; correction factor)
//   l_j   = alpha*l_{j-1} + rowsum(exp(S_j - m_j))
//   O_j   = alpha*O_{j-1} + exp(S_j - m_j) . V_j
// After last block: O = O / l.
//
// Helper notes / advisory deviations from op_design.md:
//   * Running max uses a separate rowmax (reduce<MAX>) + binary_sfpu<BinaryMax>
//     against cb_m_prev, instead of reduce<MAX>+Accumulate. The reduce helper's
//     Accumulate path POPS its accumulator CB during reload, which would destroy
//     m_{j-1} before alpha (phase 4) can read it. op_design.md documents this
//     exact separate-op alternative in its gotchas (the Quasar fallback note).
//   * Per-row vector ops use Scalar+Streaming (front-relative, in-place 1x) and
//     broadcast operands use Col + held lifecycles, mirroring the proven
//     softmax sub<COL> pattern in toy_variance's compute kernel.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_mask_in = 3;
constexpr uint32_t cb_pad_mask = 4;
constexpr uint32_t cb_max_scaler = 8;
constexpr uint32_t cb_sum_scaler = 9;
constexpr uint32_t cb_alpha = 10;
constexpr uint32_t cb_l_recip = 11;
constexpr uint32_t cb_m_blk = 12;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_qk = 24;
constexpr uint32_t cb_p = 25;
constexpr uint32_t cb_o_blk = 26;
constexpr uint32_t cb_o_run = 27;
constexpr uint32_t cb_m_prev = 28;
constexpr uint32_t cb_m_run = 29;
constexpr uint32_t cb_l_run = 30;
constexpr uint32_t cb_l_blk = 31;
}  // namespace

void kernel_main() {
    constexpr uint32_t B_q = get_compile_time_arg_val(0);
    constexpr uint32_t B_kv = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t n_kv = get_compile_time_arg_val(4);
    constexpr bool has_mask = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t scale_bits = get_compile_time_arg_val(6);
    constexpr uint32_t kv_partial = get_compile_time_arg_val(7);  // S_kv % 32 (0 => aligned)

    const uint32_t num_work = get_arg_val<uint32_t>(0);

    constexpr uint32_t q_tiles = B_q * DHt;
    constexpr uint32_t qk_tiles = B_q * B_kv;
    constexpr uint32_t o_tiles = B_q * vDHt;

    // Boot: a single hw_configure-bearing init (the only one in the kernel).
    // mm_block_init is a superset of compute_kernel_hw_startup's setup (it issues
    // the same llk_*_hw_configure + pack_sync/dest/init) plus matmul init, so it
    // covers the eltwise/reduce helpers too. Calling BOTH double-issues
    // pack_sync_init and skews the math/pack/unpack semaphores.
    mm_block_init(cb_q_in, cb_k_in, cb_qk, /*transpose*/ 0, /*ct*/ 1, /*rt*/ 1, /*kt*/ DHt);

    CircularBuffer q_buf(cb_q_in);
    CircularBuffer k_buf(cb_k_in);
    CircularBuffer v_buf(cb_v_in);
    CircularBuffer qk_buf(cb_qk);
    CircularBuffer p_buf(cb_p);
    CircularBuffer oblk_buf(cb_o_blk);

    constexpr auto qk_grid = ckl::EltwiseShape::grid(B_q, B_kv);
    constexpr auto o_grid = ckl::EltwiseShape::grid(B_q, vDHt);
    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(B_q, B_kv, 1);

    for (uint32_t w = 0; w < num_work; ++w) {
        for (uint32_t j = 0; j < n_kv; ++j) {
            const bool first = (j == 0);

            // ---------- Phase 1: S = Q . K^T ----------
            // transpose=true (within-tile) + reader grid-transpose -> K^T.
            // Q retained across the whole KV loop (WaitAndRetainOnLastBlock).
            //
            // NOTE: scale is NOT folded into Q. An in-place SFPU pre-scale on
            // cb_q_in (the reader-fed input CB) leaves cb_q_in with zero tiles
            // visible to the matmul's unpacker — the in-place pop+push on a
            // remote-producer (reader) CB nets to zero available for the
            // downstream same-thread cb_wait_front, hanging UNPACK. Instead the
            // scale is folded into the scores below (Phase 1b), in place on
            // cb_qk (a locally matmul-produced CB), which is the legal in-place
            // transform pattern. Mathematically identical:
            //   (Q*scale) . K^T == scale * (Q . K^T).
            ckl::matmul_block<
                /*transpose*/ true,
                /*packer_l1_acc*/ false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndRetainOnLastBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                q_buf, k_buf, qk_buf, q_buf, ckl::MatmulBlockShape::of(B_q, B_kv, 1, 1, DHt, 1));

            // ---------- Phase 1b: S *= scale (fold scale into scores) ----------
            // In-place on cb_qk (matmul-produced, locally framed) — legal.
            ckl::transform_in_place<cb_qk>(qk_tiles, ckl::MulUnary<>{scale_bits});

            // ---------- Phase 2: S += mask ----------
            if constexpr (has_mask) {
                ckl::add<
                    cb_qk,
                    cb_mask_in,
                    cb_qk,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::Streaming,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(qk_tiles);
            }

            // ---------- Phase 2c: S += pad_mask (KV column padding -> -inf) ----------
            // Only on the LAST kv block, only when S_kv is non-tile-aligned. K's
            // padded rows are zero, so the padded KV columns of the last score
            // tile-column score 0; without this the row-max would pick up 0 (if
            // all valid scores are negative) and the row-sum would gain
            // exp(0 - m) per padded column, both corrupting the softmax. The
            // additive -inf mask is generated once by the reader (persistent,
            // never popped) — add it held (Block-indexed), in place on cb_qk.
            // Mirrors Phase 5's operand structure (A = in-place Streaming/Scalar,
            // B = persistent HeldBulk/Block).
            if constexpr (kv_partial != 0) {
                if (j == n_kv - 1) {
                    ckl::add<
                        cb_qk,
                        cb_pad_mask,
                        cb_qk,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::Streaming,  // scores: in-place pop+repush
                        ckl::InputLifecycle::HeldBulk,   // pad mask: persistent, never popped
                        ckl::OutputLifecycle::Streaming,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Scalar,            // scores index = front
                        ckl::OperandKind::Block>(qk_tiles);  // mask index = tile i
                }
            }

            // ---------- Phase 3a: m_blk = rowmax(S) ----------
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_qk,
                cb_max_scaler,
                cb_m_blk,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(reduce_shape);

            // ---------- Phase 3b: m_run = max(m_prev, m_blk) ----------
            if (first) {
                ckl::copy<cb_m_blk, cb_m_run>(B_q);
            } else {
                // m_prev is HELD (not popped — reused by phase 4). A held operand
                // MUST use OperandKind::Block (per-row tile index i) + HeldBulk; a
                // Scalar index on a non-popped operand reads tile 0 every iter,
                // which is wrong for B_q > 1. m_blk is popped → Scalar+Streaming
                // (front advances per pop) reads the correct per-row tile.
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    cb_m_prev,
                    cb_m_blk,
                    cb_m_run,
                    ckl::InputLifecycle::HeldBulk,   // m_prev: held, Block-indexed
                    ckl::InputLifecycle::Streaming,  // m_blk: popped
                    ckl::OutputLifecycle::Streaming,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Block,         // A (m_prev) index = i
                    ckl::OperandKind::Scalar>(B_q);  // B (m_blk) index = front
            }

            if (!first) {
                // ---------- Phase 4: alpha = exp(m_prev - m_run) ----------
                // m_prev is popped (last use) → Scalar+Streaming reads per-row tile i.
                // m_run is HELD (reused by phase 7) → Block+HeldBulk reads per-row
                // tile i; a Scalar index on the held operand would read tile 0 for
                // every row (wrong for B_q > 1).
                ckl::eltwise_chain(
                    ckl::EltwiseShape(B_q),
                    ckl::BinaryFpu<
                        cb_m_prev,
                        cb_m_run,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::Streaming,  // m_prev: pop (last use)
                        ckl::InputLifecycle::HeldBulk,   // m_run: held, Block-indexed
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Scalar,
                        ckl::OperandKind::Block>{},
                    ckl::Exp<>{},
                    ckl::PackTile<
                        cb_alpha,
                        ckl::OutputLifecycle::Streaming,
                        ckl::PackTileReconfig::Output,
                        ckl::Dst::D0>{});

                // ---------- Phase 5: l_run = alpha * l_run ----------
                // l_run is in-place (Scalar+Streaming, 1x, front advances per pop).
                // alpha is HELD (reused by phase 6) → Block+HeldBulk reads per-row
                // tile i (Scalar on the held operand would read tile 0 for all rows).
                ckl::mul<
                    cb_l_run,
                    cb_alpha,
                    cb_l_run,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Block>(B_q);

                // ---------- Phase 6: O_run = alpha * O_run (Col bcast) ----------
                ckl::mul<
                    cb_o_run,
                    cb_alpha,
                    cb_o_run,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::Bulk,  // alpha: Col, pop at end
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Col>(o_grid);
            }

            // ---------- Phase 7: P = exp(S - m_run) (Col bcast) ----------
            ckl::eltwise_chain(
                qk_grid,
                ckl::BinaryFpu<
                    cb_qk,
                    cb_m_run,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,  // scores: pop (only popper)
                    ckl::InputLifecycle::HeldBulk,   // m_run: keep for commit
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Col>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_p, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output, ckl::Dst::D0>{});

            // ---------- Phase 8: commit m_prev = m_run ----------
            ckl::copy<cb_m_run, cb_m_prev>(B_q);

            // ---------- Phase 9: l_blk = rowsum(P) ----------
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_p,
                cb_sum_scaler,
                cb_l_blk,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(reduce_shape);

            // ---------- Phase 10: l_run += l_blk ----------
            if (first) {
                ckl::copy<cb_l_blk, cb_l_run>(B_q);
            } else {
                ckl::add<
                    cb_l_run,
                    cb_l_blk,
                    cb_l_run,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::Streaming,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(B_q);
            }

            // ---------- Phase 11: O_blk = P . V ----------
            ckl::matmul_block<
                /*transpose*/ false,
                /*packer_l1_acc*/ false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock>(
                p_buf, v_buf, oblk_buf, p_buf, ckl::MatmulBlockShape::of(B_q, vDHt, 1, 1, B_kv, 1));

            // ---------- Phase 12: O_run += O_blk ----------
            if (first) {
                ckl::copy<cb_o_blk, cb_o_run>(o_tiles);
            } else {
                ckl::add<
                    cb_o_run,
                    cb_o_blk,
                    cb_o_run,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::Streaming,
                    ckl::OutputLifecycle::Streaming,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::PackTileReconfig::Output,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Scalar>(o_tiles);
            }
        }

        // ---------- Phase 10a: recip(l_final) ----------
        ckl::unary<ckl::Recip<>, cb_l_run, cb_l_recip>(B_q);

        // ---------- Phase 10b: O = O_run * recip (Col bcast) ----------
        ckl::mul<
            cb_o_run,
            cb_l_recip,
            cb_out,
            ckl::BroadcastDim::Col,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Bulk,  // recip: Col, pop at end
            ckl::OutputLifecycle::Streaming,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::PackTileReconfig::Output,
            ckl::OperandKind::Scalar,
            ckl::OperandKind::Col>(o_grid);

        // ---------- Cleanup ----------
        cb_pop_front(cb_q_in, q_tiles);  // Q retained across KV loop (not popped by matmul)
        cb_wait_front(cb_m_prev, B_q);   // leftover m_final from last commit
        cb_pop_front(cb_m_prev, B_q);
    }
}
