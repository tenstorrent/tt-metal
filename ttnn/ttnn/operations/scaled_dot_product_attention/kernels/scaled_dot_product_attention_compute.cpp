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
// DPRINT: cb_scores_masked tile 0, first 4×4 elements — passthrough copy of
// scaled scores. Printed from the unpacker (TRISC0) via DPRINT_UNPACK, front
// of CB (between cb_wait_front and cb_pop_front).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    // CB indices (match op_design.md CB layout).
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_scale_factor = 5;
    constexpr uint32_t cb_scores = 24;
    constexpr uint32_t cb_scores_masked = 25;

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
    // in0_policy=WaitAndRetainOnLastBlock (retain Q),
    // in1_policy=WaitAndPopPerKBlock (consume K per KV-block).
    matmul_block<
        /*transpose=*/true,
        /*packer_l1_acc=*/false,
        /*last_block_target=*/LastBlockTarget::Out,
        /*tile_order=*/OutputCBLayout::SubblockMajor,
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

    // --- Stage 3 DPRINT: cb_scores_masked tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0). After the copy completes, all 16
    // tiles are in cb_scores_masked. We wait_front tile 0 (the first tile
    // pushed by the copy's PackTile), print it, then pop_front 1 tile.
    // The pop keeps the CB advancing for later stages (phase 4 row-max will
    // wait_front the remaining tiles with WaitUpfrontNoPop).
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_scores_masked, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_3 mask:\n{}\n", TileSlice(cb_scores_masked, 0, sr, true, true));
    }
#endif
}
