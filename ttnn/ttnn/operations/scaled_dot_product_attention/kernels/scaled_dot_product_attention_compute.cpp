// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention).
//
// Stage 0 (init): Boot the Tensix compute pipeline. The reader initializes
// running-state CBs (cb_max_old, cb_sum_old, cb_o) via raw constant fills.
//
// Stage 1 (qkt_matmul): QK^T score matmul via matmul_block (transpose=true).
//   S = Q @ K^T → cb_scores (16 tiles)
//   - in0 = cb_q (B_q_t * D_t tiles, retained via WaitAndRetainOnLastBlock)
//   - in1 = cb_k (B_kv_t * D_t tiles, consumed via WaitAndPopPerKBlock)
//   - out = cb_scores (B_q_t * B_kv_t = 16 tiles)
//   - MatmulBlockShape::of(2, 2, 2, 2, D_t, 1)
//   - tile_order=TileRowMajor: output tiles in row-major order (tile(r,c) at
//     index r*B_kv_t + c), required by the reduce helper which expects
//     row-major tile layout in the CB.
//
// Stage 2 (scale): Scale scores by 1/sqrt(D) via eltwise mul (scalar broadcast).
//   cb_scores *= cb_scale_factor (1 tile, HeldBulk, BroadcastDim::Scalar)
//   - In-place: CbA == CbOut == cb_scores
//   - The mul helper with init_mode=Short reconfigures from matmul to eltwise
//
// Stage 3 (mask): Passthrough copy cb_scores → cb_scores_masked (no mask for
//   the pinned shape). The copy helper (CopyTile → PackTile) streams 16 tiles
//   from cb_scores (Streaming — wait+pop per tile) into cb_scores_masked.
//   When an attn_mask is present, this phase becomes an eltwise add instead.
//
// Stage 4 (rowmax): Row-max reduce via reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>.
//   m_blk = max(cb_scores_masked, dim=-1) → cb_max_new (4 tiles)
//   - Input: cb_scores_masked (16 tiles, WaitUpfrontNoPop — NOT popped, left
//     for reuse in phase 8 subtract_max)
//   - Scaler: cb_scaler_reduce (1 tile, prepared by reader)
//   - Output: cb_max_new (B_q_t = 4 tiles, per-row max)
//   - ReduceInputBlockShape::of(B_q_t, B_kv_t, 1) = of(4, 4, 1)
//   - WaitUpfrontNoPop: waits for all 16 input tiles, pops 0, pushes 4 output
//
// Stage 5 (alpha): Compute alpha = exp(m_old - m_new) via eltwise_chain.
//   alpha = exp(cb_max_old - cb_max_new) → cb_alpha (4 tiles)
//   - Input A: cb_max_old (4 tiles, Bulk — consumed)
//   - Input B: cb_max_new (4 tiles, HeldBulk — retained for phase 8 + 13)
//   - Output: cb_alpha (4 tiles, Streaming)
//   - Chain: BinaryFpu<Sub> → Exp → PackTile<cb_alpha>
//
// Stage 6 (rescale_o): Rescale O by alpha via eltwise mul (Col broadcast).
//   O_i = alpha * O_i → cb_o (in-place, 4×D_t = 16 tiles)
//   - CbA  = cb_o (Streaming — per-tile wait+pop, OperandKind::Scalar)
//   - CbB  = cb_alpha (HeldBulk — retained for phase 7, OperandKind::Col)
//   - CbOut = cb_o (in-place: CbA == CbOut)
//   - BroadcastDim::Col: alpha tile[ht] broadcasts across all D_t column tiles
//   - Shape: EltwiseShape::grid(B_q_t, D_t) — 2D for Col broadcast
//
// Stage 7 (rescale_l): Rescale l by alpha via eltwise mul (no broadcast).
//   l_i = alpha * l_i → cb_sum_old (in-place, B_q_t = 4 tiles)
//   - CbA  = cb_sum_old (4 tiles, Streaming — per-tile wait+pop)
//   - CbB  = cb_alpha   (4 tiles, Streaming — consumed; phase 6 used HeldBulk
//     without popping, so alpha tiles remain for this phase)
//   - CbOut = cb_sum_old (in-place: CbA == CbOut)
//   - BroadcastDim::None (both operands are Col0 [B_q_t, 1] — same shape)
//   - Shape: B_q_t tiles (1D column vector)
//
// DPRINT: cb_sum_old tile 0, first 4×4 elements — l rescaled by alpha.
// Printed from the unpacker (TRISC0), front of CB (between cb_wait_front and
// cb_pop_front). On the first KV-block, l_old=0 and alpha=0, so l stays 0.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    // CB indices (match op_design.md CB layout).
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_reduce = 4;
    constexpr uint32_t cb_scale_factor = 5;
    constexpr uint32_t cb_alpha = 8;
    constexpr uint32_t cb_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_scores = 24;
    constexpr uint32_t cb_scores_masked = 25;
    constexpr uint32_t cb_max_new = 26;
    constexpr uint32_t cb_max_old = 27;
    constexpr uint32_t cb_exp_scores = 28;
    constexpr uint32_t cb_sum_new = 29;
    constexpr uint32_t cb_sum_old = 30;

    // Compile-time args: [B_q_t, B_kv_t, D_t]
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);   // 4
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(1);  // 4
    constexpr uint32_t D_t = get_compile_time_arg_val(2);     // D/32

    // --- Boot: engine-wide init + matmul init ---
    // compute_kernel_hw_startup: engine-wide init for eltwise (called once
    // at the very beginning, before any operation-specific init).
    // SrcOrder::Reverse: matmul maps in0→SrcB, in1→SrcA.
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q, cb_k, cb_scores);

    // mm_block_init: matmul boot (hw_configure-bearing, top of kernel_main).
    // transpose=1 (K^T), ct_dim=2 (out_subblock_w), rt_dim=2 (out_subblock_h),
    // kt_dim=D_t (in0_block_k).
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1, /*ct_dim=*/2, /*rt_dim=*/2, /*kt_dim=*/D_t);

    // --- Stage 1: QK^T score matmul ---
    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer scores_buf(cb_scores);

    // MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
    //                       out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks)
    // M=B_q_t=4, N=B_kv_t=4, K=D_t.
    constexpr auto qkt_shape = MatmulBlockShape::of(2, 2, 2, 2, D_t, 1);

    // transpose=true (K^T), packer_l1_acc=false, init_mode=Short (default),
    // tile_order=TileRowMajor (row-major CB order for reduce compatibility),
    // in0_policy=WaitAndRetainOnLastBlock (retain Q),
    // in1_policy=WaitAndPopPerKBlock (consume K per KV-block).
    matmul_block<
        /*transpose=*/true,
        /*packer_l1_acc=*/false,
        /*last_block_target=*/LastBlockTarget::Out,
        /*tile_order=*/OutputCBLayout::TileRowMajor,
        /*init_mode=*/matmul_config::InitMode::Short,
        /*in0_policy=*/InputPolicy::WaitAndRetainOnLastBlock,
        /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

    // --- Stage 2: Scale scores by 1/sqrt(D) ---
    // cb_scores *= cb_scale_factor (scalar broadcast, in-place).
    //   CbA = cb_scores (Streaming — per-tile wait+pop)
    //   CbB = cb_scale_factor (HeldBulk — wait upfront, never popped)
    //   CbOut = cb_scores (in-place)
    //   BroadcastDim::Scalar (single tile broadcast across all score tiles)
    // The mul helper with default BinaryDataFormatReconfig::Input reconfigures
    // the unpacker from matmul to eltwise format.
    constexpr uint32_t num_score_tiles = B_q_t * B_kv_t;  // 16
    mul<
        /*CbA=*/cb_scores,
        /*CbB=*/cb_scale_factor,
        /*CbOut=*/cb_scores,
        /*Bcast=*/BroadcastDim::Scalar,
        /*ALife=*/InputLifecycle::Streaming,
        /*BLife=*/InputLifecycle::HeldBulk>(num_score_tiles);

    // --- Stage 3: Mask add (if attn_mask) or passthrough copy (no mask) ---
    // For the pinned shape (no attn_mask), this is a passthrough: copy all 16
    // score tiles from cb_scores to cb_scores_masked.
    //   CbIn  = cb_scores (Streaming — per-tile wait+pop)
    //   CbOut = cb_scores_masked (Streaming — per-tile reserve+push)
    // The copy helper (CopyTile → PackTile) handles the full 16-tile block.
    // When attn_mask is present, this becomes: add<cb_scores, cb_mask, cb_scores_masked>(num_score_tiles).
    copy<cb_scores, cb_scores_masked>(num_score_tiles);

    // --- Stage 4: Row-max reduce ---
    // m_blk = max(cb_scores_masked, dim=-1) → cb_max_new (B_q_t tiles)
    //   Input: cb_scores_masked (16 tiles, WaitUpfrontNoPop — NOT popped)
    //   Scaler: cb_scaler_reduce (1 tile, prepared by reader)
    //   Output: cb_max_new (B_q_t = 4 tiles, per-row max)
    //   ReduceInputBlockShape::of(B_q_t, B_kv_t, 1) = of(4, 4, 1)
    //
    // WaitUpfrontNoPop: waits for all 16 input tiles upfront, processes them
    // with indexed access, pops 0, and pushes all 4 output tiles at the end.
    // This leaves cb_scores_masked intact for phase 8 (subtract m_new).
    //
    // reconfig_mode=INPUT_AND_OUTPUT (default): reconfigures unpacker from
    // eltwise copy format to reduce format, and packer from cb_scores_masked
    // to cb_max_new format.
    reduce<
        /*reduce_type=*/PoolType::MAX,
        /*reduce_dim=*/ReduceDim::REDUCE_ROW,
        /*input_dfb_id=*/cb_scores_masked,
        /*scaler_dfb_id=*/cb_scaler_reduce,
        /*output_dfb_id=*/cb_max_new,
        /*input_policy=*/ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));

    // --- Stage 4 DPRINT: cb_max_new tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0). After the reduce completes, all 4
    // tiles are in cb_max_new. We wait_front tile 0, print it, then pop_front
    // 1 tile to keep the CB balanced for later stages (phase 5 alpha will
    // consume cb_max_new tiles).
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_max_new, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_4 rowmax:\n{}\n", TileSlice(cb_max_new, 0, sr, true, true));
    }
#endif

    // Pop the MAX scaler tile. The reduce<MAX> with WaitUpfrontNoPop waited for
    // the scaler but did NOT pop it. cb_scaler_reduce holds 2 pages:
    //   [0] = MAX scaler (row-0 fill, consumed by Stage 4)
    //   [1] = SUM scaler (col-0 fill, consumed by Stage 10)
    // Popping 1 tile here exposes the SUM scaler at the front for Stage 10.
    cb_wait_front(cb_scaler_reduce, 1);
    cb_pop_front(cb_scaler_reduce, 1);

    // --- Stage 5: Compute alpha = exp(m_old - m_new) ---
    // alpha = exp(m_i_old - m_blk) per Q-block row → cb_alpha (B_q_t tiles)
    //   Input A: cb_max_old (4 tiles, Bulk — wait upfront + pop at end; consumed)
    //   Input B: cb_max_new (4 tiles, HeldBulk — wait upfront, no pop; retained
    //     for phase 8 subtract_max and phase 13 update_m)
    //   Output:  cb_alpha (4 tiles, Streaming — per-tile reserve+push)
    //
    // Fused chain: BinaryFpu<Sub> writes m_old - m_new to Dst::D0, then Exp<>
    // transforms D0 in-place, then PackTile packs D0 into cb_alpha.
    //
    // OperandKind::Block for both inputs: each of the B_q_t tiles is distinct
    // (per-row running max). The chain iterates over all tiles, reading tile i
    // from each CB. BroadcastDim::None (both CBs have the same shape: B_q_t tiles).
    //
    // BinaryDataFormatReconfig::Input (default): reconfigures unpacker from
    // reduce format to eltwise binary format on both srcA and srcB.
    // PackTileReconfig::Output (default): reconfigures packer from cb_max_new
    // (reduce output) to cb_alpha format.
    eltwise_chain(
        B_q_t,
        BinaryFpu<
            /*CbA=*/cb_max_old,
            /*CbB=*/cb_max_new,
            /*Op=*/BinaryFpuOp::Sub,
            /*Bcast=*/BroadcastDim::None,
            /*APolicy=*/InputLifecycle::Bulk,
            /*BPolicy=*/InputLifecycle::HeldBulk,
            /*Reconfig=*/BinaryDataFormatReconfig::Input,
            /*DstSlot=*/Dst::D0,
            /*AIndex=*/OperandKind::Block,
            /*BIndex=*/OperandKind::Block>{},
        Exp<>{},
        PackTile<cb_alpha, OutputLifecycle::Streaming>{});

    // --- Stage 5 DPRINT: cb_alpha tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0), front of CB (between cb_wait_front
    // and cb_pop_front). After the chain completes, all 4 tiles are in cb_alpha.
    // The read pointer is at tile 0 (nobody has popped). We wait_front tile 0,
    // print it, then leave it (no pop — cb_alpha is consumed by later stages 6, 7).
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_alpha, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_5 alpha:\n{}\n", TileSlice(cb_alpha, 0, sr, true, true));
    }
#endif

    // --- Stage 6: Rescale O by alpha ---
    // O_i = alpha * O_i (broadcast Col: alpha is per-row, O is 2D [B_q_t, D_t])
    //   CbA  = cb_o     (B_q_t * D_t tiles = 16 for D=128, Streaming — per-tile wait+pop)
    //   CbB  = cb_alpha (B_q_t = 4 tiles, HeldBulk — wait upfront, no pop; retained
    //     for phase 7 rescale_l which also needs alpha)
    //   CbOut = cb_o    (in-place: CbA == CbOut)
    //   BroadcastDim::Col: alpha tile[ht] broadcasts across all D_t column tiles
    //   OperandKind::Scalar for A (front-relative streaming, reads tile 0 per iter)
    //   OperandKind::Col   for B (reads tile ht per row, window = Ht = B_q_t)
    //   Shape: EltwiseShape::grid(B_q_t, D_t) — 2D for Col broadcast
    //
    // BinaryDataFormatReconfig::Input (default): reconfigures unpacker from
    // eltwise_chain (alpha phase) to eltwise binary format on both srcA and srcB.
    // PackTileReconfig::Output (default): reconfigures packer from cb_alpha
    // to cb_o format.
    mul<
        /*CbA=*/cb_o,
        /*CbB=*/cb_alpha,
        /*CbOut=*/cb_o,
        /*Bcast=*/BroadcastDim::Col,
        /*ALife=*/InputLifecycle::Streaming,
        /*BLife=*/InputLifecycle::HeldBulk,
        /*OutLife=*/OutputLifecycle::Streaming,
        /*Reconfig=*/BinaryDataFormatReconfig::Input,
        /*OutReconfig=*/PackTileReconfig::Output,
        /*AIdx=*/OperandKind::Scalar,
        /*BIdx=*/OperandKind::Col>(EltwiseShape::grid(B_q_t, D_t));

    // --- Stage 6 DPRINT: cb_o tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0), front of CB (between cb_wait_front
    // and cb_pop_front). After the in-place mul completes, cb_o holds the rescaled
    // O values. We wait_front tile 0, print it, then leave it (no pop — cb_o is
    // a persistent running state CB, consumed by later stages 7, 12).
    // On the first KV-block: O_old=0 and alpha=exp(-inf - finite)=0, so O stays 0.
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_o, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_6 rescale_o:\n{}\n", TileSlice(cb_o, 0, sr, true, true));
    }
#endif

    // --- Stage 7: Rescale l by alpha ---
    // l_i = alpha * l_i → cb_sum_old (in-place, B_q_t = 4 tiles)
    //   CbA  = cb_sum_old (4 tiles, Streaming — per-tile wait+pop)
    //   CbB  = cb_alpha   (4 tiles, Streaming — consumed here; phase 6 used
    //     HeldBulk and did NOT pop, so alpha tiles are still in the CB)
    //   CbOut = cb_sum_old (in-place: CbA == CbOut)
    //   BroadcastDim::None (both operands are Col0 [B_q_t, 1] — same shape per
    //     the Broadcast Verification table; no broadcast needed)
    //   Shape: B_q_t tiles (1D column vector, both operands 4 tiles)
    //
    // BinaryDataFormatReconfig::Input (default): reconfigures unpacker from
    // eltwise mul (rescale_o) format to eltwise binary format on both srcA/srcB.
    // PackTileReconfig::Output (default): reconfigures packer from cb_o to
    // cb_sum_old format.
    mul<
        /*CbA=*/cb_sum_old,
        /*CbB=*/cb_alpha,
        /*CbOut=*/cb_sum_old,
        /*Bcast=*/BroadcastDim::None,
        /*ALife=*/InputLifecycle::Streaming,
        /*BLife=*/InputLifecycle::Streaming>(B_q_t);

    // --- Stage 7 DPRINT: cb_sum_old tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0), front of CB (between cb_wait_front
    // and cb_pop_front). After the in-place mul completes, cb_sum_old holds the
    // rescaled l values. We wait_front tile 0, print it, then leave it (no pop —
    // cb_sum_old is a persistent running state CB, consumed by later stages 11).
    // On the first KV-block: l_old=0 and alpha=exp(-inf - finite)=0, so l stays 0.
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_sum_old, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_7 rescale_l:\n{}\n", TileSlice(cb_sum_old, 0, sr, true, true));
    }
#endif

    // --- Stage 8: Subtract m_new from scores ---
    // S = S - m_new (broadcast Col: m_new is per-row, S is 2D [B_q_t, B_kv_t])
    //   CbA  = cb_scores_masked (B_q_t * B_kv_t = 16 tiles, Streaming — per-tile
    //     wait+pop; phase 4 used WaitUpfrontNoPop which left all 16 tiles intact)
    //   CbB  = cb_max_new (B_q_t = 4 tiles, HeldBulk — wait upfront, no pop;
    //     retained for phase 13 update_m. Phase 5 used HeldBulk and did NOT pop,
    //     so all 4 tiles are still at the front of cb_max_new)
    //   CbOut = cb_scores_masked (in-place: CbA == CbOut)
    //   BroadcastDim::Col: m_new tile[ht] broadcasts across all B_kv_t column tiles
    //   OperandKind::Scalar for A (front-relative streaming, reads tile 0 per iter)
    //   OperandKind::Col   for B (reads tile ht per row, window = Ht = B_q_t)
    //   Shape: EltwiseShape::grid(B_q_t, B_kv_t) — 2D for Col broadcast
    //
    // BinaryDataFormatReconfig::Input (default): reconfigures unpacker from
    // eltwise mul (rescale_l) format to eltwise binary format on both srcA/srcB.
    // PackTileReconfig::Output (default): reconfigures packer from cb_sum_old
    // to cb_scores_masked format.
    sub<
        /*CbA=*/cb_scores_masked,
        /*CbB=*/cb_max_new,
        /*CbOut=*/cb_scores_masked,
        /*Bcast=*/BroadcastDim::Col,
        /*ALife=*/InputLifecycle::Streaming,
        /*BLife=*/InputLifecycle::HeldBulk,
        /*OutLife=*/OutputLifecycle::Streaming,
        /*Reconfig=*/BinaryDataFormatReconfig::Input,
        /*OutReconfig=*/PackTileReconfig::Output,
        /*AIdx=*/OperandKind::Scalar,
        /*BIdx=*/OperandKind::Col>(EltwiseShape::grid(B_q_t, B_kv_t));

    // --- Stage 8 DPRINT: cb_scores_masked tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0), front of CB (between cb_wait_front
    // and cb_pop_front). After the in-place sub completes, cb_scores_masked
    // holds the rescaled scores (S - m_new). We wait_front tile 0, print it,
    // then leave it (no pop — cb_scores_masked is consumed by later stage 9 exp).
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_scores_masked, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_8 subtract_max:\n{}\n", TileSlice(cb_scores_masked, 0, sr, true, true));
    }
#endif

    // --- Stage 9: Exp of rescaled scores ---
    // P = exp(S - m_new) → cb_exp_scores (B_q_t * B_kv_t = 16 tiles)
    //   CbIn  = cb_scores_masked (16 tiles, Streaming — per-tile wait+pop; phase 8
    //     sub left all 16 tiles at the front via in-place Streaming write)
    //   CbOut = cb_exp_scores (16 tiles, Streaming — per-tile reserve+push)
    //   Chain: CopyTile(D0) → Exp(D0) → PackTile(cb_exp_scores)
    //   Exp<> defaults: Approx::Exact (matches math_approx_mode=False), Dst::D0
    //   Shape: num_score_tiles (16) — 1D streaming walk, no broadcast needed
    //
    // CopyTileReconfig::Input (default): reconfigures unpacker from eltwise binary
    // sub format to eltwise unary format.
    // PackTileReconfig::Output (default): reconfigures packer from cb_scores_masked
    // to cb_exp_scores format.
    unary<
        /*SfpuOp=*/Exp<>,
        /*CbIn=*/cb_scores_masked,
        /*CbOut=*/cb_exp_scores>(num_score_tiles);

    // --- Stage 9 DPRINT: cb_exp_scores tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0), front of CB (between cb_wait_front
    // and cb_pop_front). After the unary exp completes, cb_exp_scores holds the
    // exp of rescaled scores (P = exp(S - m_new)). We wait_front tile 0, print
    // it, then leave it (no pop — cb_exp_scores is consumed by later stages 10, 12).
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_exp_scores, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_9 exp:\n{}\n", TileSlice(cb_exp_scores, 0, sr, true, true));
    }
#endif

    // --- Stage 10: Row-sum reduce ---
    // l_blk = sum(cb_exp_scores, dim=-1) → cb_sum_new (B_q_t tiles)
    //   Input: cb_exp_scores (16 tiles, WaitAndPopPerTile — per-tile wait+pop,
    //     consuming the full exp-scores block)
    //   Scaler: cb_scaler_reduce (1 tile at front — the SUM scaler, col-0 fill.
    //     The MAX scaler was popped after Stage 4, so this is tile 0 now.)
    //   Output: cb_sum_new (B_q_t = 4 tiles, per-row sum)
    //   ReduceInputBlockShape::of(B_q_t, B_kv_t, 1) = of(4, 4, 1)
    //
    // SUM REDUCE_ROW uses the matmul path (reduce_uses_matmul<SUM, REDUCE_ROW>
    // = true), which is why the reader prepared a col-0-fill scaler via
    // calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>.
    //
    // WaitAndPopPerTile: waits and pops one tile at a time (streaming), pushing
    // 4 output tiles one at a time. This consumes all 16 exp-score tiles.
    //
    // reconfig_mode=INPUT_AND_OUTPUT (default): reconfigures unpacker from
    // eltwise unary (exp) format to reduce format, and packer from cb_exp_scores
    // to cb_sum_new format.
    reduce<
        /*reduce_type=*/PoolType::SUM,
        /*reduce_dim=*/ReduceDim::REDUCE_ROW,
        /*input_dfb_id=*/cb_exp_scores,
        /*scaler_dfb_id=*/cb_scaler_reduce,
        /*output_dfb_id=*/cb_sum_new,
        /*input_policy=*/ReduceInputPolicy::WaitAndPopPerTile>(ReduceInputBlockShape::of(B_q_t, B_kv_t, 1));

    // --- Stage 10 DPRINT: cb_sum_new tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0), front of CB (between cb_wait_front
    // and cb_pop_front). After the reduce completes, cb_sum_new holds the
    // per-row sum of exp scores (l_blk = sum(P, dim=-1)). We wait_front tile 0,
    // print it, then leave it (no pop — cb_sum_new is consumed by later stage 11).
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_sum_new, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_10 rowsum:\n{}\n", TileSlice(cb_sum_new, 0, sr, true, true));
    }
#endif
}
