// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Flash Attention — Compute Kernel (TRISC)
// Implements online softmax Flash Attention algorithm.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/fill.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace ckl = compute_kernel_lib;

// CB indices
constexpr uint32_t cb_q = 0;
constexpr uint32_t cb_k = 1;
constexpr uint32_t cb_v = 2;
constexpr uint32_t cb_attn_mask = 3;
constexpr uint32_t cb_scale = 4;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_scores = 24;
constexpr uint32_t cb_m = 25;
constexpr uint32_t cb_l = 26;
constexpr uint32_t cb_o = 27;
constexpr uint32_t cb_m_new = 28;
constexpr uint32_t cb_psum = 29;
constexpr uint32_t cb_pv = 30;
constexpr uint32_t cb_scaler = 31;

constexpr uint32_t NEG_INF_BITS = 0xFF800000;

// Helper: fill a CB with num_tiles of a constant value (raw LLK init)
// Uses the correct tile_regs protocol: acquire → fill → commit → wait → pack → release → push
void init_cb_constant_f(uint32_t cb_id, uint32_t num_tiles, float value) {
    fill_tile_init();
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        tile_regs_acquire();
        fill_tile(0, value);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_id);
        tile_regs_release();
        cb_push_back(cb_id, 1);
    }
}

void init_cb_constant_bits(uint32_t cb_id, uint32_t num_tiles, uint32_t fill_bits) {
    fill_tile_init();
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        tile_regs_acquire();
        fill_tile_bitcast(0, fill_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_id);
        tile_regs_release();
        cb_push_back(cb_id, 1);
    }
}

void kernel_main() {
    constexpr uint32_t B_q = get_compile_time_arg_val(0);
    constexpr uint32_t B_kv = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);
    const uint32_t S_q_t = get_compile_time_arg_val(3);
    const uint32_t S_kv_t = get_compile_time_arg_val(4);
    const uint32_t num_q_blocks = get_compile_time_arg_val(5);
    const uint32_t num_kv_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t has_mask = get_compile_time_arg_val(7);
    const uint32_t scale_bits = get_compile_time_arg_val(8);

    // CircularBuffer objects for matmul_block (requires Buf&)
    CircularBuffer cb_q_buf(cb_q);
    CircularBuffer cb_k_buf(cb_k);
    CircularBuffer cb_v_buf(cb_v);
    CircularBuffer cb_scores_buf(cb_scores);
    CircularBuffer cb_pv_buf(cb_pv);

    // Boot init — compute_kernel_hw_startup (hardware configure) MUST be called
    // before any compute API. Use Regular src order matching the eltwise/reduce
    // chains. The matmul helpers' mm_block_init_short handles the matmul-specific
    // state_configure (srca=in1, srcb=in0).
    ckernel::compute_kernel_hw_startup(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1, /*ct_dim=*/B_kv, /*rt_dim=*/B_q, /*kt_dim=*/D_t);

    // Phase 0: Init persistent state (m_i=-inf, l_i=0, O_i=0)
    init_cb_constant_bits(cb_m, B_q, NEG_INF_BITS);  // m_i = -inf
    init_cb_constant_f(cb_l, B_q, 0.0f);             // l_i = 0
    init_cb_constant_f(cb_o, B_q * D_t, 0.0f);       // O_i = 0

    // Outer loop: Q blocks
    for (uint32_t qb = 0; qb < num_q_blocks; qb++) {
        // Inner loop: KV blocks
        for (uint32_t kvb = 0; kvb < num_kv_blocks; kvb++) {
            cb_wait_front(cb_k, B_kv * D_t);
            cb_wait_front(cb_v, B_kv * D_t);
            if constexpr (has_mask) {
                cb_wait_front(cb_attn_mask, B_q * B_kv);
            }
            cb_wait_front(cb_scaler, 2);

            // Phase 1: QK^T — S = Q @ K^T
            // Q tiles are re-pushed by reader per KV block, so WaitAndPopPerKBlock is correct.
            // DataFormatReconfig::INPUT_AND_OUTPUT — reconfig back to matmul mode after the
            // eltwise/reduce chains that ran in the previous KV block iteration.
            ckl::matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                ckl::NoPreKBlock,
                ckl::NoPostKBlock,
                0,  // untilize_block_ct_dim
                ckl::NoKBlockInnerDimFn,
                ckl::NoIn0Source,
                ckl::NoIn1BaseOffset,
                false,  // caller_owns_pack_target
                ckl::NoneActivation,
                ckl::matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_q_buf,
                cb_k_buf,
                cb_scores_buf,
                cb_scores_buf,
                ckl::MatmulBlockShape::of(B_q, B_kv, B_q, B_kv, D_t, 1));

            // Phase 2: Scale scores — S *= scale
            ckl::transform_in_place<cb_scores>(ckl::EltwiseShape::grid(B_q, B_kv), ckl::MulUnary<>{scale_bits});

            // Phase 2b: Apply mask (optional)
            if constexpr (has_mask) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(B_q, B_kv),
                    ckl::BinaryFpu<cb_scores, cb_attn_mask, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None>{},
                    ckl::PackTile<cb_scores, ckl::OutputLifecycle::Streaming>{});
            }

            // Phase 3: RowMax — m_block = rowmax(S)
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_scores,
                cb_scaler,
                cb_m_new,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(B_q, B_kv));
            // NOTE: scaler popped after reduce — MAX scaler (tile 0) consumed
            cb_pop_front(cb_scaler, 1);

            // Phase 4: OnlineMax — m_new = max(m_i, m_block)
            // Fix F4: use OperandKind::Block for 1D tiles(B_q) chains
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(B_q),
                ckl::CopyTile<
                    cb_m,
                    ckl::Dst::D0,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    ckl::OperandKind::Block>{},
                ckl::CopyTile<cb_m_new, ckl::Dst::D1, ckl::InputLifecycle::Streaming>{},
                ckl::BinaryMax<ckl::Dst::D1, ckl::Dst::D0, ckl::Dst::D0>{},
                ckl::PackTile<cb_m_new, ckl::OutputLifecycle::Streaming>{});

            // Phase 5: ExpScores — P = exp(S - m_new)
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(B_q, B_kv),
                ckl::BinaryFpu<
                    cb_scores,
                    cb_m_new,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Col>{},
                ckl::Exp<ckl::Approx::Fast, ckl::Approx::Fast, ckl::Dst::D0>{},
                ckl::PackTile<cb_scores, ckl::OutputLifecycle::Streaming>{});

            // Phase 6: Copy P from cb_scores to cb_pv
            ckl::copy<cb_scores, cb_pv>(ckl::EltwiseShape::tiles(B_q * B_kv));

            // Phase 7: Rescale l_i — l_i *= exp(m_i - m_new)
            // Fix F4: use OperandKind::Block for HeldBulk cb_m, cb_m_new
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(B_q),
                ckl::CopyTile<cb_l, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                ckl::CopyTile<
                    cb_m,
                    ckl::Dst::D1,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    ckl::OperandKind::Block>{},
                ckl::CopyTile<
                    cb_m_new,
                    ckl::Dst::D2,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    ckl::OperandKind::Block>{},
                ckl::SubBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
                ckl::Exp<ckl::Approx::Fast, ckl::Approx::Fast, ckl::Dst::D1>{},
                ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                ckl::PackTile<cb_l, ckl::OutputLifecycle::Streaming>{});

            // Phase 8: Rescale O_i — O_i *= exp(m_i - m_new)
            // Compute factor = exp(m_i - m_new) in cb_psum first
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(B_q),
                ckl::CopyTile<
                    cb_m,
                    ckl::Dst::D0,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    ckl::OperandKind::Block>{},
                ckl::CopyTile<
                    cb_m_new,
                    ckl::Dst::D1,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::CopyTileReconfig::Input,
                    ckl::OperandKind::Block>{},
                ckl::SubBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                ckl::Exp<ckl::Approx::Fast, ckl::Approx::Fast, ckl::Dst::D0>{},
                ckl::PackTile<cb_psum, ckl::OutputLifecycle::Streaming>{});

            // O_i *= factor (Col broadcast of factor across D_t)
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(B_q, D_t),
                ckl::BinaryFpu<
                    cb_o,
                    cb_psum,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Scalar,
                    ckl::OperandKind::Col>{},
                ckl::PackTile<cb_o, ckl::OutputLifecycle::Streaming>{});

            cb_pop_front(cb_psum, B_q);

            // Phase 9: Update m_i — pop old m_i, copy m_new → m_i
            // Fix F3: don't pop cb_m_new before copy — copy will pop it (Streaming)
            cb_pop_front(cb_m, B_q);
            ckl::copy<cb_m_new, cb_m>(ckl::EltwiseShape::tiles(B_q));

            // Phase 10: RowSum — psum = rowsum(P)
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_pv,
                cb_scaler,
                cb_psum,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(B_q, B_kv));
            cb_pop_front(cb_scaler, 1);

            // Phase 11: l_i += psum
            ckl::add<cb_l, cb_psum, cb_l>(ckl::EltwiseShape::tiles(B_q));

            // Phase 12: PV = P @ V
            // Shape: M=B_q, N=D_t, K=B_kv. Single subblock (1,1) of size (B_q, D_t).
            // DEST constraint: B_q * D_t <= DEST_AUTO_LIMIT (4 with fp32_dest_acc_en).
            // DataFormatReconfig::INPUT_AND_OUTPUT — reconfig back to matmul mode after
            // the eltwise/reduce chains in Phases 2-11.
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                ckl::LastBlockTarget::Out,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                ckl::NoPreKBlock,
                ckl::NoPostKBlock,
                0,  // untilize_block_ct_dim
                ckl::NoKBlockInnerDimFn,
                ckl::NoIn0Source,
                ckl::NoIn1BaseOffset,
                false,
                ckl::NoneActivation,
                ckl::matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_pv_buf, cb_v_buf, cb_scores_buf, cb_scores_buf, ckl::MatmulBlockShape::of(1, 1, B_q, D_t, B_kv, 1));

            // Phase 13: O_i += PV
            ckl::add<cb_o, cb_scores, cb_o>(ckl::EltwiseShape::grid(B_q, D_t));

            // Cleanup: K and V tiles are already popped by the matmul helpers
            // (WaitAndPopPerKBlock in both Phase 1 QK^T and Phase 12 PV matmuls).
            // Only pop mask if present.
            if constexpr (has_mask) {
                cb_pop_front(cb_attn_mask, B_q * B_kv);
            }
        }

        // Scaler tiles are already consumed per KV block (MAX scaler popped
        // after rowmax at line ~153, SUM scaler popped after rowsum at line ~262).
        // No extra scaler pop here — the reader pushes exactly 2 per KV block.

        // Phase 14: Normalize — O = O_i * recip(l_i)
        // l_i from REDUCE_ROW has values only in col 0. DivBinary (SFPU) doesn't broadcast.
        // Step 1: compute recip(l_i) in cb_psum (col-0 only, but FPU broadcast will fix it)
        // Step 2: O_i * recip(l_i) with BinaryFpu<Mul, Col> which broadcasts B across columns
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(B_q),
            ckl::CopyTile<
                cb_l,
                ckl::Dst::D0,
                ckl::InputLifecycle::HeldBulk,
                ckl::CopyTileReconfig::Input,
                ckl::OperandKind::Block>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_psum, ckl::OutputLifecycle::Streaming>{});

        // O_i *= recip(l_i) with Col broadcast
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(B_q, D_t),
            ckl::BinaryFpu<
                cb_o,
                cb_psum,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Col,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldBulk,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Col>{},
            ckl::PackTile<cb_output, ckl::OutputLifecycle::Streaming>{});

        cb_pop_front(cb_psum, B_q);

        // Pop persistent state
        cb_pop_front(cb_o, B_q * D_t);
        cb_pop_front(cb_l, B_q);
        cb_pop_front(cb_m, B_q);

        // Re-init persistent state for next Q block.
        init_cb_constant_bits(cb_m, B_q, NEG_INF_BITS);
        init_cb_constant_f(cb_l, B_q, 0.0f);
        init_cb_constant_f(cb_o, B_q * D_t, 0.0f);
    }
}
