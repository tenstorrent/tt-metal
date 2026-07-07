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
constexpr uint32_t cb_pv_out = 23;  // PV matmul output (separate from cb_scores)
constexpr uint32_t cb_qk_partials = 22;  // QK^T K-block spill/reload intermediates
constexpr uint32_t cb_scaler = 31;

// Use a large negative finite value instead of -inf for m_i initialization.
// The SFPU exp() approximation may not handle -inf correctly (producing NaN
// instead of 0). A large negative value like -1e38 ensures exp(-1e38) ≈ 0
// while remaining a valid finite float.
constexpr float NEG_INF_F = -1e38f;

// Helper: fill a CB with num_tiles of a constant value (raw LLK init)
// Uses the correct tile_regs protocol: acquire → fill → commit → wait → pack → release → push
// Must call pack_reconfig_data_format before pack_tile — the packer may be
// configured for a different CB's format from the previous operation.
void init_cb_constant_f(uint32_t cb_id, uint32_t num_tiles, float value) {
    fill_tile_init();
    PACK((pack_reconfig_data_format(cb_id)));
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
    constexpr uint32_t is_causal = get_compile_time_arg_val(9);
    constexpr uint32_t has_kv_padding = get_compile_time_arg_val(10);
    constexpr uint32_t D_CHUNK = get_compile_time_arg_val(11);
    const uint32_t num_k_blocks_qk = get_compile_time_arg_val(12);
    const uint32_t num_d_chunks = get_compile_time_arg_val(13);

    // RT arg: number of work units (B,H pairs) this core processes
    const uint32_t num_work_units = get_arg_val<uint32_t>(0);

    // CircularBuffer objects for matmul_block (requires Buf&)
    CircularBuffer cb_q_buf(cb_q);
    CircularBuffer cb_k_buf(cb_k);
    CircularBuffer cb_v_buf(cb_v);
    CircularBuffer cb_scores_buf(cb_scores);
    CircularBuffer cb_pv_buf(cb_pv);
    CircularBuffer cb_pv_out_buf(cb_pv_out);
    CircularBuffer cb_qk_partials_buf(cb_qk_partials);

    // Boot init — compute_kernel_hw_startup (hardware configure) MUST be called
    // before any compute API. Use Regular src order matching the eltwise/reduce
    // chains. The matmul helpers' mm_block_init_short handles the matmul-specific
    // state_configure (srca=in1, srcb=in0).
    // kt_dim = D_CHUNK (per-K-block tile count for QK^T K-blocking)
    ckernel::compute_kernel_hw_startup(cb_q, cb_k, cb_scores);
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1, /*ct_dim=*/B_kv, /*rt_dim=*/B_q, /*kt_dim=*/D_CHUNK);

    // PV matmul subblocking: DEST limit is 4 tiles with fp32_dest_acc_en.
    // out_subblock_h * out_subblock_w must fit. With B_q=1, out_subblock_h=1,
    // out_subblock_w = min(D_CHUNK, 4/B_q). If D_CHUNK > 4/B_q, split N.
    constexpr uint32_t PV_MAX_SUBBLOCK_W = (B_q <= 4) ? (4 / B_q) : 1;
    constexpr uint32_t PV_SUBBLOCK_W = (D_CHUNK < PV_MAX_SUBBLOCK_W) ? D_CHUNK : PV_MAX_SUBBLOCK_W;
    constexpr uint32_t PV_NUM_SUBBLOCKS_N = (D_CHUNK + PV_SUBBLOCK_W - 1) / PV_SUBBLOCK_W;

    // Loop over work units (multiple (B,H) pairs per core when B*H > num_cores)
    for (uint32_t wu = 0; wu < num_work_units; wu++) {
        // Outer loop: Q blocks
        for (uint32_t qb = 0; qb < num_q_blocks; qb++) {
            // D_CHUNK outer loop: process D_t in N-chunks for PV/O_i/output.
            // For each D_CHUNK, replay all KV blocks. QK^T is K-blocked over
            // the full D_t (cb_q/cb_k are D_CHUNK-bounded). PV uses only the
            // current D_CHUNK of V (cb_v is D_CHUNK-bounded). O_i is
            // D_CHUNK-sized. When D_CHUNK == D_t (small head_dim), this loop
            // runs once — identical to the original non-chunked behavior.
            for (uint32_t dc = 0; dc < num_d_chunks; dc++) {
                // Phase 0: Init persistent state (m_i=neg_inf, l_i=0, O_i=0)
                // O_i is now D_CHUNK-sized (not D_t).
                init_cb_constant_f(cb_m, B_q, NEG_INF_F);       // m_i = neg_inf
                init_cb_constant_f(cb_l, B_q, 0.0f);            // l_i = 0
                init_cb_constant_f(cb_o, B_q * D_CHUNK, 0.0f);  // O_i = 0
                // Inner loop: KV blocks
                for (uint32_t kvb = 0; kvb < num_kv_blocks; kvb++) {
                    // Q and K are pushed in K-blocks of D_CHUNK tiles.
                    // matmul_block with WaitAndPopPerKBlock waits for
                    // B_q * D_CHUNK tiles per K-block.
                    // V is pushed as B_kv * D_CHUNK tiles (current N-chunk only).
                    // NOTE: mask and scaler waits are moved to AFTER the QK^T matmul.
                    // The reader pushes Q/K in K-blocks, then V, then mask, then scalers.
                    // If cb_q fills up while the reader is pushing K-blocks, the reader
                    // blocks on cb_reserve_back. If compute waits for mask/scaler before
                    // QK^T, it deadlocks: reader can't push more Q/K (cb_q full), compute
                    // can't start QK^T (waiting for mask). Moving mask/scaler waits to
                    // after QK^T (which consumes Q/K and frees cb_q) breaks the cycle.

                    // Phase 1: QK^T — S = Q @ K^T (K-blocked over D_t)
                    // num_k_blocks = D_t / D_CHUNK. The helper handles spill/reload
                    // of partials via cb_qk_partials (software path, packer_l1_acc=false).
                    // interm_buf = cb_qk_partials (separate CB for TileRowMajor spill).
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
                        cb_qk_partials_buf,
                        ckl::MatmulBlockShape::of(B_q, B_kv, B_q, B_kv, D_CHUNK, num_k_blocks_qk));

                    // Phase 2: Scale scores — S *= scale
                    ckl::transform_in_place<cb_scores>(ckl::EltwiseShape::grid(B_q, B_kv), ckl::MulUnary<>{scale_bits});

                    // Phase 2b: Apply mask (custom, causal, or padding)
                    // Wait for mask tiles here (after QK^T consumed Q/K tiles).
                    if constexpr (has_mask || is_causal || has_kv_padding) {
                        cb_wait_front(cb_attn_mask, B_q * B_kv);
                    }
                    if constexpr (has_mask || is_causal || has_kv_padding) {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::grid(B_q, B_kv),
                            ckl::BinaryFpu<cb_scores, cb_attn_mask, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None>{},
                            ckl::PackTile<cb_scores, ckl::OutputLifecycle::Streaming>{});
                    }

                    // Phase 3: RowMax — m_block = rowmax(S)
                    // Scalers are pushed by the reader after Q/K/V. Wait here
                    // (after QK^T matmul consumed Q/K) to avoid deadlock.
                    cb_wait_front(cb_scaler, 2);
                    ckl::reduce<
                        ckernel::PoolType::MAX,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_scores,
                        cb_scaler,
                        cb_m_new,
                        ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::of(B_q, B_kv));
                    cb_pop_front(cb_scaler, 1);

                    // Phase 4: OnlineMax — m_new = max(m_i, m_block)
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
                        ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                        ckl::PackTile<cb_scores, ckl::OutputLifecycle::Streaming>{});

                    // Phase 6: Copy P from cb_scores to cb_pv
                    ckl::copy<cb_scores, cb_pv>(ckl::EltwiseShape::tiles(B_q * B_kv));

                    // Phase 7: Rescale l_i — l_i *= exp(m_i - m_new)
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
                        ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D1>{},
                        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                        ckl::PackTile<cb_l, ckl::OutputLifecycle::Streaming>{});

                    // Phase 8: Rescale O_i — O_i *= exp(m_i - m_new)
                    // O_i is now D_CHUNK tiles (not D_t).
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
                        ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                        ckl::PackTile<cb_psum, ckl::OutputLifecycle::Streaming>{});

                    // O_i *= factor (Col broadcast across D_CHUNK columns)
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(B_q, D_CHUNK),
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

                    // Phase 12: PV = P @ V (N-chunked: N=D_CHUNK)
                    // M=B_q, N=D_CHUNK, K=B_kv.
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
                        cb_pv_buf,
                        cb_v_buf,
                        cb_pv_out_buf,
                        cb_pv_out_buf,
                        ckl::MatmulBlockShape::of(1, PV_NUM_SUBBLOCKS_N, B_q, PV_SUBBLOCK_W, B_kv, 1));

                    // Phase 13: O_i += PV (D_CHUNK tiles)
                    ckl::add<cb_o, cb_pv_out, cb_o>(ckl::EltwiseShape::grid(B_q, D_CHUNK));

                    // Cleanup: K and V tiles are already popped by the matmul helpers.
                    // Mask tiles are already popped by BinaryFpu<Add> chain.
                }

                // Phase 14: Normalize — O = O_i * recip(l_i)
                // O_i is D_CHUNK tiles. l_i is constant (B_q tiles).
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

                // O_i *= recip(l_i) with Col broadcast across D_CHUNK
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(B_q, D_CHUNK),
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

                // Pop persistent state. cb_o was already popped by the Phase 14 chain
                // (BinaryFpu<cb_o, ..., Streaming> pops B_q*D_CHUNK tiles). cb_l is
                // HeldBulk in the recip chain (not popped), so pop manually.
                cb_pop_front(cb_l, B_q);
                cb_pop_front(cb_m, B_q);
            }
        }
    }  // end work-unit loop
}
